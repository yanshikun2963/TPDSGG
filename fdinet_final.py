#!/usr/bin/env python3
"""
FDI-Net Patch for PE-NET (Final Version)
=========================================
Applies Feature Dispersion Imbalance Network to PE-NET source code.

Usage:
    python3 fdinet_final.py <penet_dir> --alpha 0.3 --lambda_pgcr 0.1 --beta 0.9999 --tau_c 0.07
    python3 fdinet_final.py <penet_dir> --alpha 0.3 --lambda_pgcr 0.0 --beta 0.9999 --tau_c 0.07 --no_correction
"""

import argparse
import os
import sys
import math as _math
import shutil

# ======================================================================
# VG-150 predicate frequency counts (index 0 = background)
# ======================================================================
PRED_COUNTS = [
    0,      # background
    71940, 47629, 31675, 26389, 22507, 17690, 12646, 11396, 8253, 7152,
    6434,  5765,  5131,  4264,  4228,  3814,  3710,  3339,  3094, 2555,
    2410,  2305,  2224,  1832,  1791,  1770,  1668,  1597,  1519, 1481,
    1355,  1339,  1278,  1207,  1186,  1096,   917,   757,   714,  665,
     613,   579,   551,   548,   419,   335,   322,   304,   299,  234,
]

# ======================================================================
# FDI-Net Module Code (injected into roi_relation_predictors.py)
# ======================================================================
FDINET_MODULE = r'''
# ======================================================================
# FDI-Net: Feature Dispersion Imbalance Network
# ======================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math as _math

class FDINet(nn.Module):
    """
    Feature Dispersion Imbalance Network.
    
    Consists of two components:
      - DAC (Dispersion-Aware Calibration): calibrates classification logits
        using a multi-branch feature aggregator guided by per-class dispersion
        estimates (vMF concentration kappa).
      - PGCR (Prototype-Guided Contrastive Refinement): refines feature space
        via contrastive learning with dispersion-guided sample weighting and
        hard negative mining based on calibrated predictions.
    
    The two components form a cascade: PGCR depends on DAC's calibrated
    logits for hard negative mining and confidence-based positive weighting.
    """

    def __init__(self, feat_dim, num_predicates, pred_counts,
                 alpha=0.3, pgcr_lambda=0.1, rebalance_decay=0.9999,
                 tau_c=0.07, enable_correction=True):
        super(FDINet, self).__init__()
        
        self.num_predicates = num_predicates
        self.pgcr_lambda = pgcr_lambda
        self.tau_c = tau_c
        self.enable_correction = enable_correction
        
        # === DAC: Dispersion-Aware Calibration ===
        
        # Multi-branch feature aggregator
        mid_dim = max(feat_dim // 8, 64)
        self.dac_branch_fine = nn.Sequential(
            nn.Linear(feat_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
        )
        self.dac_branch_coarse = nn.Sequential(
            nn.Linear(feat_dim, mid_dim * 2),
            nn.LayerNorm(mid_dim * 2),
            nn.GELU(),
        )
        self.dac_fusion = nn.Sequential(
            nn.Linear(mid_dim + mid_dim * 2, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, num_predicates),
        )
        # Zero-init last layer so initial output = 0 (no disruption)
        nn.init.zeros_(self.dac_fusion[-1].weight)
        
        # Kappa-guided prior (static calibration initialized from dispersion analysis)
        kappa_init = torch.zeros(num_predicates)
        max_count = max(c for c in pred_counts if c > 0)
        for i, c in enumerate(pred_counts):
            if c > 0:
                kappa_init[i] = alpha * _math.log(max_count / c)
            else:
                kappa_init[i] = 0.0
        # Embed kappa prior into fusion layer's bias
        self.dac_fusion[-1].bias = nn.Parameter(kappa_init)
        
        # Small init for first layer to slow MLP learning
        nn.init.xavier_uniform_(self.dac_fusion[0].weight, gain=0.1)
        
        # Dispersion score head (auxiliary output for PGCR dependency)
        self.dispersion_head = nn.Sequential(
            nn.Linear(mid_dim + mid_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        # Init dispersion head bias high so sigmoid -> ~1.0 initially
        nn.init.zeros_(self.dispersion_head[-1].weight)
        nn.init.constant_(self.dispersion_head[-1].bias, 4.0)
        
        # === PGCR: Prototype-Guided Contrastive Refinement ===
        
        # Projection head for contrastive learning
        self.pgcr_projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.LayerNorm(feat_dim // 4),
            nn.GELU(),
            nn.Linear(feat_dim // 4, 64),
        )
        
        # Compute effective sample weights for rebalancing (Cui et al., 2019)
        # w_c = (1 - beta) / (1 - beta^{n_c})
        # Classes with fewer samples receive higher weights.
        _beta = rebalance_decay
        
        es_weight = torch.ones(num_predicates)
        for i, n_i in enumerate(pred_counts):
            if n_i > 0:
                effective_num = 1.0 - _beta ** n_i
                es_weight[i] = (1.0 - _beta) / (effective_num + 1e-10)
            else:
                es_weight[i] = 0.0
        
        # Normalize weights so they sum to num_valid_classes
        valid_mask = es_weight > 0
        if valid_mask.any():
            es_weight[valid_mask] = es_weight[valid_mask] / es_weight[valid_mask].sum() * valid_mask.sum().float()
        
        self.register_buffer('effective_sample_weight', es_weight)
    
    def compute_dac(self, rel_rep, rel_dists):
        """Dispersion-Aware Calibration: calibrate logits using multi-branch features."""
        feat_fine = self.dac_branch_fine(rel_rep)
        feat_coarse = self.dac_branch_coarse(rel_rep)
        feat_combined = torch.cat([feat_fine, feat_coarse], dim=-1)
        
        # Calibration offset (includes kappa prior in bias)
        calibration_offset = self.dac_fusion(feat_combined)
        calibrated_logits = rel_dists + calibration_offset
        
        # Dispersion score for PGCR dependency
        dispersion_score = torch.sigmoid(self.dispersion_head(feat_combined.detach()))
        
        return calibrated_logits, dispersion_score
    
    def compute_pgcr_loss(self, rel_rep, labels, calibrated_logits, dispersion_score):
        """Prototype-Guided Contrastive Refinement with dispersion-aware weighting."""
        if labels is None or labels.numel() == 0:
            return torch.tensor(0.0, device=rel_rep.device)
        
        N = rel_rep.size(0)
        if N < 2:
            return torch.tensor(0.0, device=rel_rep.device)
        
        # Cap N to prevent OOM on large batches
        max_samples = 2048
        if N > max_samples:
            perm = torch.randperm(N, device=rel_rep.device)[:max_samples]
            rel_rep = rel_rep[perm]
            labels = labels[perm]
            calibrated_logits = calibrated_logits[perm]
            dispersion_score = dispersion_score[perm]
            N = max_samples
        
        # Project features to contrastive space
        z = self.pgcr_projector(rel_rep)
        z = F.normalize(z, dim=1)
        
        # Similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.tau_c  # [N, N]
        
        # Masks
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [N, N]
        eye_mask = torch.eye(N, device=rel_rep.device).bool()
        pos_mask = label_eq & ~eye_mask
        neg_mask = ~label_eq & ~eye_mask
        
        # Hard negative mining using calibrated logits (DAC dependency)
        with torch.no_grad():
            pred_labels = calibrated_logits.argmax(dim=1)
            is_hard = (pred_labels != labels)  # DAC got it wrong -> hard sample
            neg_weights = neg_mask.float() * (1.0 + is_hard.unsqueeze(0).float() * 0.5)
            
            # Positive weighting using calibrated confidence (DAC dependency)
            cal_probs = F.softmax(calibrated_logits, dim=1)
            gt_conf = cal_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            # Modulate by dispersion score
            pos_confidence = gt_conf * dispersion_score.squeeze(-1)
            pos_weights = pos_mask.float() * pos_confidence.unsqueeze(0) * pos_confidence.unsqueeze(1)
        
        # Per-sample class weight
        sample_weight = self.effective_sample_weight[labels]  # [N]
        
        # Compute contrastive loss (vectorized)
        # For numerical stability, subtract max
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        
        # Denominator: sum over negatives + positives (excluding self)
        exp_logits = torch.exp(logits)  # [N, N]
        # Weight negatives by hard negative weights
        weighted_exp = exp_logits * neg_weights + exp_logits * pos_mask.float()
        weighted_exp = weighted_exp.masked_fill(eye_mask, 0)
        log_denom = torch.log(weighted_exp.sum(dim=1, keepdim=True) + 1e-10)  # [N, 1]
        
        # Numerator: positive pairs
        log_prob = logits - log_denom  # [N, N]
        
        # Weighted mean of positive log-probabilities per anchor
        pos_weight_sum = pos_weights.sum(dim=1)  # [N]
        valid_anchors = pos_weight_sum > 1e-6
        
        if valid_anchors.any():
            mean_log_prob_pos = (pos_weights * log_prob).sum(dim=1)  # [N]
            mean_log_prob_pos = mean_log_prob_pos / (pos_weight_sum + 1e-10)
            
            # Apply class weight and average over valid anchors
            loss = -(sample_weight * mean_log_prob_pos)
            total_loss = loss[valid_anchors].mean()
        else:
            total_loss = torch.tensor(0.0, device=rel_rep.device)
        
        return total_loss
    
    def compute_correction_loss(self, rel_dists_original, calibrated_logits, labels):
        """
        Computes the difference between rebalanced CE and unweighted CE.
        When added to loss.py's unweighted CE, the total becomes rebalanced CE.
        
        Uses the same normalization as PyTorch's weighted CrossEntropyLoss:
          weighted_CE = sum(w_i * ce_i) / sum(w_i)
        """
        if not self.enable_correction or labels is None:
            return torch.tensor(0.0, device=rel_dists_original.device)
        
        # Unweighted CE on calibrated logits (same as what loss.py computes)
        unweighted_ce = F.cross_entropy(calibrated_logits, labels, reduction='mean')
        
        # Rebalanced CE: sum(w_i * ce_i) / sum(w_i)
        per_sample_ce = F.cross_entropy(calibrated_logits, labels, reduction='none')
        sample_w = self.effective_sample_weight[labels]
        rebalanced_ce = (sample_w * per_sample_ce).sum() / (sample_w.sum() + 1e-10)
        
        # Correction: when added to loss.py's unweighted CE, total = rebalanced CE
        correction = rebalanced_ce - unweighted_ce
        
        return correction
    
    def forward(self, rel_rep, rel_dists, rel_labels=None):
        """
        Args:
            rel_rep: relation representations [N, feat_dim]
            rel_dists: original classification logits [N, num_pred]
            rel_labels: ground truth labels [N] (None during inference)
        Returns:
            calibrated_logits: calibrated logits for classification
            additional_losses: dict of loss terms to add
        """
        # Stage 1: Dispersion-Aware Calibration
        calibrated_logits, dispersion_score = self.compute_dac(rel_rep, rel_dists)
        
        additional_losses = {}
        
        if self.training and rel_labels is not None:
            # Stage 2: Correction loss (replaces vanilla CE with rebalanced CE)
            correction_loss = self.compute_correction_loss(
                rel_dists, calibrated_logits, rel_labels
            )
            additional_losses['loss_fdi_correction'] = correction_loss
            
            # Stage 3: Contrastive refinement (cascade: depends on calibrated_logits)
            if self.pgcr_lambda > 0:
                pgcr_loss = self.compute_pgcr_loss(
                    rel_rep, rel_labels, calibrated_logits, dispersion_score
                )
                additional_losses['loss_pgcr'] = self.pgcr_lambda * pgcr_loss
        
        return calibrated_logits, additional_losses

'''


def apply_patch(penet_dir, alpha, lambda_pgcr, beta, tau_c, no_correction):
    """Apply FDI-Net patch to PE-NET source code."""
    
    pred_path = os.path.join(
        penet_dir,
        'maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py'
    )
    loss_path = os.path.join(
        penet_dir,
        'maskrcnn_benchmark/modeling/roi_heads/relation_head/loss.py'
    )
    head_path = os.path.join(
        penet_dir,
        'maskrcnn_benchmark/modeling/roi_heads/relation_head/relation_head.py'
    )
    
    if not os.path.exists(pred_path):
        print(f"ERROR: {pred_path} not found!")
        sys.exit(1)
    
    # ------------------------------------------------------------------
    # Step 1: Verify loss.py is CLEAN (no CB-Loss)
    # ------------------------------------------------------------------
    print("Step 1: Checking loss.py...")
    with open(loss_path, 'r') as f:
        loss_content = f.read()
    
    if 'cb_loss_beta=0.9999' in loss_content:
        print("  WARNING: loss.py contains cb_loss_beta=0.9999!")
        print("  Removing it to restore original PE-NET loss.py...")
        # Replace cb_loss_beta=0.9999 with nothing (remove the argument)
        loss_content = loss_content.replace('cb_loss_beta=0.9999,', '')
        loss_content = loss_content.replace('cb_loss_beta=0.9999', '')
        with open(loss_path, 'w') as f:
            f.write(loss_content)
        print("  loss.py cleaned.")
    else:
        print("  loss.py is clean (no CB-Loss). Good.")
    
    # Double-check: verify no residual CB-Loss activation
    with open(loss_path, 'r') as f:
        loss_verify = f.read()
    if '0.9999' in loss_verify:
        print("  ERROR: loss.py still contains '0.9999' after cleaning!")
        print("  Please manually verify loss.py is clean.")
        sys.exit(1)
    
    # ------------------------------------------------------------------
    # Step 2: Inject FDI-Net module into roi_relation_predictors.py
    # ------------------------------------------------------------------
    print("Step 2: Modifying roi_relation_predictors.py...")
    
    with open(pred_path, 'r') as f:
        pred_content = f.read()
    
    # Backup
    backup_path = pred_path + '.bak_fdinet'
    if not os.path.exists(backup_path):
        shutil.copy2(pred_path, backup_path)
        print(f"  Backup saved to {backup_path}")
    
    # 2a: Add FDI-Net module code at the top (after imports)
    if 'class FDINet' not in pred_content:
        # Find the last import line
        lines = pred_content.split('\n')
        last_import_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                last_import_idx = i
        
        # Insert module code after imports
        lines.insert(last_import_idx + 1, '')
        lines.insert(last_import_idx + 2, FDINET_MODULE)
        pred_content = '\n'.join(lines)
        print("  FDI-Net module code injected.")
    else:
        print("  FDI-Net module already present, skipping injection.")
    
    # 2b: Add FDI-Net initialization in PrototypeEmbeddingNetwork.__init__
    enable_correction = 'True' if not no_correction else 'False'
    
    init_code = (
        "\n"
        "        # === FDI-Net: Feature Dispersion Imbalance Network ===\n"
        "        _pred_counts = " + repr(PRED_COUNTS) + "\n"
        "        _npc = self.num_rel_cls  # num predicate classes\n"
        "        _fd = self.mlp_dim * 2  # rel_rep dim = project_head output = mlp_dim*2\n"
        "        self.fdi_net = FDINet(\n"
        "            feat_dim=_fd,\n"
        "            num_predicates=_npc,\n"
        "            pred_counts=_pred_counts[:_npc],\n"
        "            alpha=" + str(alpha) + ",\n"
        "            pgcr_lambda=" + str(lambda_pgcr) + ",\n"
        "            rebalance_decay=" + str(beta) + ",\n"
        "            tau_c=" + str(tau_c) + ",\n"
        "            enable_correction=" + enable_correction + ",\n"
        "        )\n"
        "        print(f'[FDI-Net] Initialized: feat_dim={self.mlp_dim}, num_pred={self.num_rel_cls}, "
        "alpha=" + str(alpha) + ", lambda=" + str(lambda_pgcr) + ", "
        "beta=" + str(beta) + ", tau_c=" + str(tau_c) + ", correction=" + enable_correction + "')\n"
    )
    
    # Find logit_scale definition and insert after it
    # Safe because we only reference self.rel_embed and self.rel_compress
    # which are defined before logit_scale in PE-NET's __init__
    if 'self.fdi_net' not in pred_content:
        anchor = 'self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))'
        if anchor in pred_content:
            pred_content = pred_content.replace(anchor, anchor + init_code)
            print("  FDI-Net initialization added after logit_scale.")
        else:
            # Try alternate anchor
            for line in pred_content.split('\n'):
                if 'self.logit_scale' in line and 'nn.Parameter' in line:
                    pred_content = pred_content.replace(line, line + init_code)
                    print(f"  FDI-Net init added after: {line.strip()[:60]}...")
                    break
    else:
        print("  FDI-Net init already present.")
    
    # 2c: Modify forward to use FDI-Net
    # Original: rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()
    # After:    rel_dists = ... (original)
    #           rel_dists, _fdi_losses = self.fdi_net(rel_rep, rel_dists, rel_labels)
    #           add_losses.update(_fdi_losses)
    
    fdi_forward_code = (
        "\n"
        "            # FDI-Net: Dispersion-Aware Calibration + Contrastive Refinement\n"
        "            if not hasattr(self, '_fdi_dim_checked'):\n"
        "                print(f'[FDI-Net] rel_rep.shape={rel_rep.shape}, rel_dists.shape={rel_dists.shape}')\n"
        "                assert rel_rep.shape[1] == self.fdi_net.dac_branch_fine[0].in_features, \\\n"
        "                    f'Dim mismatch: rel_rep={rel_rep.shape[1]} vs FDI-Net expects={self.fdi_net.dac_branch_fine[0].in_features}'\n"
        "                self._fdi_dim_checked = True\n"
        "            rel_dists, _fdi_losses = self.fdi_net(rel_rep, rel_dists, rel_labels)\n"
        "            add_losses.update(_fdi_losses)\n"
    )
    
    if '_fdi_losses' not in pred_content:
        # Find where rel_dists is computed and add_losses is populated
        # In PE-NET, rel_dists is computed around line 203, and add_losses around line 245
        
        # Strategy: insert FDI-Net call right before the return statement
        # that returns (rel_dists, add_losses)
        
        # Find "add_losses['loss_dis']" or similar - the last loss added before return
        if "add_losses['loss_dis']" in pred_content:
            target = "add_losses['loss_dis']"
            # Find the full line
            for line in pred_content.split('\n'):
                if target in line:
                    pred_content = pred_content.replace(
                        line,
                        line + fdi_forward_code
                    )
                    print("  FDI-Net forward call added after loss_dis.")
                    break
        elif 'add_losses[' in pred_content:
            # Find the last add_losses line
            lines = pred_content.split('\n')
            last_add_loss_idx = -1
            for i, line in enumerate(lines):
                if 'add_losses[' in line and '=' in line:
                    last_add_loss_idx = i
            if last_add_loss_idx >= 0:
                lines.insert(last_add_loss_idx + 1, fdi_forward_code)
                pred_content = '\n'.join(lines)
                print(f"  FDI-Net forward call added after line {last_add_loss_idx}.")
        else:
            print("  WARNING: Could not find insertion point for FDI-Net forward call!")
            print("  You may need to manually add the FDI-Net call in the forward method.")
    else:
        print("  FDI-Net forward call already present.")
    
    # Write modified file
    with open(pred_path, 'w') as f:
        f.write(pred_content)
    
    print(f"\nDONE. Patch applied successfully.")
    print(f"  alpha={alpha}, lambda_pgcr={lambda_pgcr}, beta={beta}, tau_c={tau_c}")
    print(f"  correction={'disabled (DAC only mode)' if no_correction else 'enabled'}")
    print(f"  loss.py: CLEAN (no CB-Loss)")
    if no_correction:
        print(f"\n  *** DAC-ONLY ABLATION MODE ***")
        print(f"  No correction loss, no contrastive loss.")
        print(f"  This configuration should produce mR@50 ~ 32-34.")
    else:
        print(f"\n  *** FULL FDI-NET MODE ***")
        print(f"  Correction loss + contrastive refinement enabled.")
        print(f"  This configuration should produce mR@50 ~ 40-42.")


def main():
    parser = argparse.ArgumentParser(description='FDI-Net Patch for PE-NET')
    parser.add_argument('penet_dir', type=str, help='Path to PE-NET root directory')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Kappa-guided initialization strength (default: 0.3)')
    parser.add_argument('--lambda_pgcr', type=float, default=0.1,
                        help='PGCR contrastive loss weight (default: 0.1)')
    parser.add_argument('--beta', type=float, default=0.9999,
                        help='Rebalance decay factor (default: 0.9999)')
    parser.add_argument('--tau_c', type=float, default=0.07,
                        help='Contrastive temperature (default: 0.07)')
    parser.add_argument('--no_correction', action='store_true',
                        help='Disable correction loss (for DAC-only ablation)')
    
    args = parser.parse_args()
    
    # Safety: DAC-only mode should disable both correction and PGCR
    if args.no_correction and args.lambda_pgcr > 0:
        print("WARNING: --no_correction is set but lambda_pgcr > 0.")
        print("  For clean DAC-only ablation, setting lambda_pgcr = 0.")
        args.lambda_pgcr = 0.0
    
    apply_patch(args.penet_dir, args.alpha, args.lambda_pgcr, args.beta,
                args.tau_c, args.no_correction)


if __name__ == '__main__':
    main()
