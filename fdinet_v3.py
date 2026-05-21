#!/usr/bin/env python3
"""
FDI-Net Joint Training Patch (v3 - Bug Fixed)
==============================================
Patches PE-NET with:
  - DAC: Learnable per-class bias (Sub-mechanism 2)
  - PGCR: Prototype-Guided Contrastive Refinement with CB-Loss weight (Sub-mechanism 1)

KEY FIX vs fdinet_patch.py:
  The previous version had PGCR contrastive loss = 0.0 because:
  1. It tried to use rel_labels AFTER split (list of tensors, not concatenated)
  2. It inserted contrastive code in wrong position

This version computes contrastive loss BEFORE rel_dists.split() at line 207,
using rel_rep_norm (continuous tensor) and properly concatenated rel_labels.

Usage:
  python3 fdinet_v3.py <penet_root> --alpha 0.3 --lambda_pgcr 0.1 --beta 0.9999 --tau_c 0.07

Parameters:
  --alpha       : DAC bias initialization scale (0=zero init, >0=log-freq init)
  --lambda_pgcr : PGCR loss weight (0=disable PGCR, just DAC+CB-Loss)
  --beta        : CB-Loss beta for PGCR class weights (0.9999 recommended)
  --tau_c       : Contrastive temperature (0.07 recommended)
"""

import argparse
import os
import sys
import numpy as np


# VG-150 predicate frequencies (classes 1-50, from VG dataset)
PRED_COUNTS = [
    71940, 47629, 31675, 26389, 22507, 17690, 12646, 11396, 8253, 7152,
    6434, 5765, 5131, 4264, 4228, 3814, 3710, 3339, 3094, 2555,
    2410, 2305, 2224, 1832, 1791, 1770, 1668, 1597, 1519, 1481,
    1355, 1339, 1278, 1207, 1186, 1096, 917, 757, 714, 665,
    613, 579, 551, 548, 419, 335, 322, 304, 299, 234
]


def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)
    print(f"  Written: {path}")


def generate_init_code(alpha, lambda_pgcr, beta, tau_c):
    """Generate the __init__ code to insert after logit_scale"""
    
    # Compute log-freq bias initialization
    counts = np.array(PRED_COUNTS, dtype=np.float64)
    log_freq_bias = np.log(counts.max() / counts)  # higher for tail classes
    # Prepend 0 for background class
    log_freq_bias = np.concatenate([[0.0], log_freq_bias])
    bias_str = ", ".join([f"{v:.4f}" for v in log_freq_bias])
    
    # Compute CB-Loss weights for PGCR
    effective_num = 1.0 - np.power(beta, counts)
    cb_w = (1.0 - beta) / effective_num
    # Prepend background weight (very small)
    bg_count = sum(counts) * 3.0
    bg_effective = 1.0 - beta ** bg_count
    bg_w = (1.0 - beta) / bg_effective
    all_weights = np.concatenate([[bg_w], cb_w])
    all_weights = all_weights / all_weights.sum() * len(all_weights)  # normalize
    weights_str = ", ".join([f"{v:.6f}" for v in all_weights])
    
    code = '''
        # ============================================================
        # FDI-Net: Distribution-Aware Calibration (DAC) + 
        #          Prototype-Guided Contrastive Refinement (PGCR)
        # ============================================================
        
        # DAC: Learnable per-class calibration bias
        _alpha = ''' + str(alpha) + '''
        if _alpha > 0:
            _log_freq_init = torch.tensor([''' + bias_str + '''], dtype=torch.float32)
            self.predicate_bias = nn.Parameter(_log_freq_init * _alpha)
        else:
            self.predicate_bias = nn.Parameter(torch.zeros(self.num_rel_cls))
        
        # PGCR parameters
        self.pgcr_lambda = ''' + str(lambda_pgcr) + '''
        self.pgcr_tau = ''' + str(tau_c) + '''
        
        # PGCR: CB-Loss weights as class-balancing coefficients for contrastive loss
        self.register_buffer('pgcr_cb_weights', torch.tensor([''' + weights_str + '''], dtype=torch.float32))
        
        print(f"[FDI-Net] DAC alpha={_alpha}, PGCR lambda={self.pgcr_lambda}, beta=''' + str(beta) + ''', tau_c={self.pgcr_tau}")
        print(f"[FDI-Net] DAC bias range: [{self.predicate_bias.min().item():.4f}, {self.predicate_bias.max().item():.4f}]")
        print(f"[FDI-Net] PGCR CB-weight range: [{self.pgcr_cb_weights.min().item():.4f}, {self.pgcr_cb_weights.max().item():.4f}]")
'''
    return code


def generate_forward_code():
    """Generate the forward code: DAC bias + PGCR contrastive loss.
    
    This replaces the SINGLE line:
        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()
    
    With DAC bias addition AND PGCR contrastive loss computation.
    
    CRITICAL: This must happen BEFORE rel_dists.split() at line 207,
    because after split rel_dists becomes a list of tensors.
    """
    
    code = '''        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  #  <r_norm, c_norm> / tau
        
        # DAC: Add learnable per-class calibration bias
        rel_dists = rel_dists + self.predicate_bias.unsqueeze(0)
        
        # PGCR: Compute contrastive loss BEFORE rel_dists.split()
        # Must use rel_rep_norm (continuous tensor) and concatenate rel_labels here
        if self.training and self.pgcr_lambda > 0:
            # Concatenate rel_labels (they come in as a list of tensors)
            _rel_labels_cat = cat(rel_labels, dim=0) if isinstance(rel_labels, (list, tuple)) else rel_labels
            
            # Supervised contrastive loss on rel_rep_norm
            # For each sample i with label y_i:
            #   L_i = -w_{y_i} * (1/|P(i)|) * sum_{j in P(i)} log(exp(sim_ij/tau) / sum_k exp(sim_ik/tau))
            # where P(i) = {j : y_j == y_i, j != i}
            
            _batch_size = rel_rep_norm.size(0)
            
            if _batch_size > 1:
                # Cosine similarity matrix between all pairs
                _sim_matrix = rel_rep_norm @ rel_rep_norm.t()  # [B, B]
                _sim_matrix = _sim_matrix / self.pgcr_tau  # scale by temperature
                
                # Mask: positive pairs (same class, excluding self)
                _labels_col = _rel_labels_cat.unsqueeze(1)  # [B, 1]
                _labels_row = _rel_labels_cat.unsqueeze(0)  # [1, B]
                _pos_mask = (_labels_col == _labels_row).float()  # [B, B]
                _self_mask = 1.0 - torch.eye(_batch_size, device=rel_rep_norm.device)
                _pos_mask = _pos_mask * _self_mask  # exclude self-pairs
                
                # Only compute loss for foreground samples (label > 0)
                _fg_mask = (_rel_labels_cat > 0).float()  # [B]
                
                # Number of positives per sample
                _num_pos = _pos_mask.sum(dim=1).clamp(min=1.0)  # [B]
                
                # Log-sum-exp for stability
                _logits_max, _ = _sim_matrix.max(dim=1, keepdim=True)
                _logits = _sim_matrix - _logits_max.detach()  # for numerical stability
                
                # Denominator: sum over all samples except self
                _exp_logits = torch.exp(_logits) * _self_mask
                _log_prob = _logits - torch.log(_exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-8))
                
                # Mean of log-prob over positive pairs
                _mean_log_prob_pos = (_pos_mask * _log_prob).sum(dim=1) / _num_pos  # [B]
                
                # CB-Loss class weights as contrastive balancing coefficients
                _sample_weights = self.pgcr_cb_weights[_rel_labels_cat]  # [B]
                
                # Final contrastive loss (only foreground, weighted by CB)
                _pgcr_loss = -(_sample_weights * _fg_mask * _mean_log_prob_pos).sum() / _fg_mask.sum().clamp(min=1.0)
                
                add_losses.update({"pgcr_loss": self.pgcr_lambda * _pgcr_loss})
            else:
                add_losses.update({"pgcr_loss": torch.tensor(0.0, device=rel_rep_norm.device)})
        elif self.training:
            add_losses.update({"pgcr_loss": torch.tensor(0.0, device=rel_rep_norm.device)})'''
    
    return code


def patch_predictors(filepath, alpha, lambda_pgcr, beta, tau_c):
    content = read_file(filepath)
    
    # ============================================================
    # PATCH 1: Add DAC + PGCR parameters in __init__
    # Insert after logit_scale definition
    # ============================================================
    old_init = "        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))"
    init_code = generate_init_code(alpha, lambda_pgcr, beta, tau_c)
    new_init = old_init + init_code
    
    if old_init not in content:
        print("  ERROR: Cannot find logit_scale in __init__")
        return False
    content = content.replace(old_init, new_init, 1)  # replace only first occurrence
    print("  PATCH 1: DAC + PGCR parameters added in __init__")
    
    # ============================================================
    # PATCH 2: Replace rel_dists computation line with DAC + PGCR
    # This is the CRITICAL fix: compute everything BEFORE split()
    # ============================================================
    old_forward = "        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  #  <r_norm, c_norm> / τ"
    new_forward = generate_forward_code()
    
    if old_forward not in content:
        print("  ERROR: Cannot find rel_dists computation in forward")
        return False
    content = content.replace(old_forward, new_forward, 1)
    print("  PATCH 2: DAC bias + PGCR contrastive loss added in forward (BEFORE split)")
    
    # ============================================================
    # PATCH 3: Ensure 'cat' is imported (it should already be, but verify)
    # ============================================================
    if "from maskrcnn_benchmark.modeling.utils import cat" not in content:
        if "import torch" in content:
            content = content.replace(
                "import torch",
                "import torch\nfrom maskrcnn_benchmark.modeling.utils import cat",
                1
            )
            print("  PATCH 3: Added 'cat' import")
    
    write_file(filepath, content)
    return True


def verify_loss_py(penet_root):
    """Verify that loss.py has CB-Loss beta=0.9999"""
    loss_path = os.path.join(penet_root,
        "maskrcnn_benchmark/modeling/roi_heads/relation_head/loss.py")
    if os.path.exists(loss_path):
        content = read_file(loss_path)
        if "cb_loss_beta=0.9999" in content:
            print("  VERIFIED: CB-Loss beta=0.9999 present in loss.py")
            print("  NOTE: CB-Loss in loss.py affects the MAIN classification loss (L_e_sim)")
            print("  NOTE: PGCR's CB weights are SEPARATE and only affect contrastive loss")
            return True
        else:
            print("  WARNING: cb_loss_beta=0.9999 NOT found in loss.py")
            print("  The main CE loss will use uniform weights (no CB-Loss)")
            return True  # still proceed
    else:
        print(f"  ERROR: {loss_path} not found")
        return False


def main():
    parser = argparse.ArgumentParser(description='FDI-Net Joint Training Patch v3')
    parser.add_argument('penet_root', help='Path to PE-NET root')
    parser.add_argument('--alpha', type=float, default=0.3,
                       help='DAC bias init scale (0=zero, >0=log-freq)')
    parser.add_argument('--lambda_pgcr', type=float, default=0.1,
                       help='PGCR contrastive loss weight (0=disable)')
    parser.add_argument('--beta', type=float, default=0.9999,
                       help='CB-Loss beta for PGCR weights')
    parser.add_argument('--tau_c', type=float, default=0.07,
                       help='Contrastive temperature')
    args = parser.parse_args()
    
    pred_path = os.path.join(args.penet_root,
        "maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py")
    
    if not os.path.exists(pred_path):
        print(f"ERROR: {pred_path} not found")
        sys.exit(1)
    
    print(f"FDI-Net v3 Patch")
    print(f"  alpha={args.alpha}, lambda_pgcr={args.lambda_pgcr}, beta={args.beta}, tau_c={args.tau_c}")
    print(f"  Target: {pred_path}")
    print()
    
    success = patch_predictors(pred_path, args.alpha, args.lambda_pgcr, args.beta, args.tau_c)
    if not success:
        print("\nPATCH FAILED")
        sys.exit(1)
    
    print()
    verify_loss_py(args.penet_root)
    
    print()
    print("=" * 60)
    print("DONE. FDI-Net v3 patch applied successfully.")
    print()
    print("What this patch does:")
    print("  1. DAC: Adds learnable per-class bias to cosine logits")
    print("  2. PGCR: Adds supervised contrastive loss with CB-weighted")
    print("     class balancing coefficients")
    print("  3. CB-Loss in loss.py (main CE loss) is NOT modified")
    print()
    print("Total loss = CE(rel_dists, labels, weight=cb_main)")
    print("           + lambda * SupCon(rel_rep_norm, labels, weight=cb_pgcr)")
    print("           + L21 + dist_loss2 + loss_dis (unchanged)")
    print("=" * 60)


if __name__ == '__main__':
    main()
