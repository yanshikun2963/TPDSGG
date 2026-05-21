#!/usr/bin/env python3
"""
FDI-Net v4 Patch — loss.py-SAFE version (FINAL FIXED)
======================================================
This script ONLY modifies roi_relation_predictors.py.
It does NOT touch loss.py at all.

You must manually ensure loss.py has cb_loss_beta=0.9999
(use the proven loss.py that produced mR@50=41.2%).

Usage:
  python3 fdinet_v4.py <penet_root> --alpha 0.3 --lambda_pgcr 0.1 \
      --beta 0.9999 --tau_c 0.07 --residual_scale 1.0
"""

import argparse
import os
import sys
import math as _math
import shutil

# ============================================================
# Predicate frequency counts (VG-150, 50 predicates)
# ============================================================
PRED_COUNTS = [
    0,      # background
    71940, 47629, 31675, 26389, 22507, 17690, 12646, 11396,
    8253,  7152,  6434,  5765,  5131,  4264,  4228,  3814,
    3710,  3339,  3094,  2555,  2410,  2305,  2224,  1832,
    1791,  1770,  1668,  1597,  1519,  1481,  1355,  1339,
    1278,  1207,  1186,  1096,  917,   757,   714,   665,
    613,   579,   551,   548,   419,   335,   322,   304,
    299,   234,
]


def generate_fdinet_module(alpha, lambda_pgcr, beta, tau_c, residual_scale):
    """Generate the FDINet nn.Module class definition."""

    # Pre-compute kappa_init values (51 values: background=0 + 50 predicates)
    counts = PRED_COUNTS[1:]  # skip background for computation
    max_c = max(counts)
    kappa_vals = [0.0] + [alpha * _math.log(max_c / max(c, 1)) for c in counts]  # [0] = background
    kappa_str = ", ".join([f"{v:.4f}" for v in kappa_vals])

    # Pre-compute CB-Loss weights for PGCR contrastive learning (51 values: bg + 50 predicates)
    cb_weights = [0.0]  # background weight = 0 (excluded by fg_mask anyway)
    for c in counts:
        eff = 1.0 - beta ** c
        cb_weights.append((1.0 - beta) / max(eff, 1e-10))
    # Normalize the 50 predicate weights (skip background)
    s = sum(cb_weights[1:])
    n = len(cb_weights[1:])
    cb_weights = [cb_weights[0]] + [w / s * n for w in cb_weights[1:]]
    cb_str = ", ".join([f"{v:.6f}" for v in cb_weights])

    # NOTE: Using doubled braces {{ }} for Python format string escaping
    code = f'''
# ============================================================
# FDI-Net: Feature Dispersion Imbalance Network
# ============================================================
import torch as _fdi_torch
import torch.nn as _fdi_nn
import torch.nn.functional as _fdi_F

class FDINet(_fdi_nn.Module):
    """
    FDI-Net: Dispersion-Aware Calibration (DAC) +
             Prototype-Guided Contrastive Refinement (PGCR)
    """
    def __init__(self, feat_dim, num_predicates):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_predicates = num_predicates
        self.pgcr_lambda = {lambda_pgcr}
        self.tau_c = {tau_c}
        self.residual_scale = {residual_scale}

        # --- DAC: Multi-branch feature aggregation ---
        _half = feat_dim // 2
        _quarter = feat_dim // 4
        # Fine branch: preserves detail
        self.dac_branch_fine = _fdi_nn.Sequential(
            _fdi_nn.Linear(feat_dim, _half),
            _fdi_nn.LayerNorm(_half),
            _fdi_nn.GELU(),
        )
        # Coarse branch: captures global pattern
        self.dac_branch_coarse = _fdi_nn.Sequential(
            _fdi_nn.Linear(feat_dim, _quarter),
            _fdi_nn.LayerNorm(_quarter),
            _fdi_nn.GELU(),
        )
        # Fusion: combines branches -> per-class calibration offset
        _cat_dim = _half + _quarter
        self.dac_fusion = _fdi_nn.Sequential(
            _fdi_nn.Linear(_cat_dim, _cat_dim // 2),
            _fdi_nn.GELU(),
            _fdi_nn.Linear(_cat_dim // 2, num_predicates),
        )
        # Zero-init last layer weight so initial output = bias = kappa_init
        _fdi_nn.init.zeros_(self.dac_fusion[-1].weight)
        # Small init for earlier layers to control MLP learning speed
        _fdi_nn.init.xavier_uniform_(self.dac_fusion[0].weight, gain=0.1)
        # Set bias of last layer = kappa_init (static calibration prior)
        _kappa_init = _fdi_torch.tensor([{kappa_str}])
        self.dac_fusion[-1].bias = _fdi_nn.Parameter(_kappa_init)

        # --- DAC: Dispersion score head (auxiliary output for PGCR) ---
        self.dac_dispersion_head = _fdi_nn.Sequential(
            _fdi_nn.Linear(_cat_dim, 64),
            _fdi_nn.GELU(),
            _fdi_nn.Linear(64, 1),
        )
        # Init bias=4.0 so sigmoid(4)~0.98 initially
        _fdi_nn.init.constant_(self.dac_dispersion_head[-1].bias, 4.0)

        # --- PGCR: Projection head for contrastive learning ---
        self.pgcr_projector = _fdi_nn.Sequential(
            _fdi_nn.Linear(feat_dim, feat_dim // 4),
            _fdi_nn.LayerNorm(feat_dim // 4),
            _fdi_nn.GELU(),
            _fdi_nn.Linear(feat_dim // 4, 64),
        )

        # --- PGCR: CB-Loss weights for contrastive sample weighting ---
        _pgcr_w = _fdi_torch.tensor([{cb_str}])
        self.register_buffer("pgcr_cb_weights", _pgcr_w)

        self._dim_checked = False

    def forward(self, rel_rep_norm, rel_dists, rel_labels=None):
        """
        Args:
            rel_rep_norm: [N, feat_dim] L2-normalized relation features
            rel_dists:    [N, num_pred] original cosine logits
            rel_labels:   [N] ground-truth labels (can be list or tensor, None during inference)
        Returns:
            calibrated_logits: [N, num_pred]
            additional_losses: dict
        """
        # Runtime dimension check (once)
        if not self._dim_checked:
            if rel_rep_norm.shape[1] != self.feat_dim:
                print(f"[FDI-Net] Dim mismatch: rel_rep={{rel_rep_norm.shape[1]}} vs expected={{self.feat_dim}}")
            self._dim_checked = True

        additional_losses = {{}}

        # === DAC: Dispersion-Aware Calibration ===
        _fine = self.dac_branch_fine(rel_rep_norm)
        _coarse = self.dac_branch_coarse(rel_rep_norm)
        _feat_cat = _fdi_torch.cat([_fine, _coarse], dim=1)
        _dac_offset = self.dac_fusion(_feat_cat)  # [N, num_pred]

        # Separate static bias from dynamic MLP output
        _static_bias = self.dac_fusion[-1].bias.detach().unsqueeze(0)  # [1, num_pred]
        # Apply residual_scale only to dynamic part (W*x), keep static bias intact
        _dac_offset = self.residual_scale * (_dac_offset - _static_bias) + _static_bias

        calibrated_logits = rel_dists + _dac_offset

        # DAC auxiliary: dispersion score (for PGCR cascade dependency)
        _dac_dispersion = _fdi_torch.sigmoid(
            self.dac_dispersion_head(_feat_cat.detach())
        ).squeeze(-1)  # [N]

        # === PGCR: Prototype-Guided Contrastive Refinement ===
        if self.training and rel_labels is not None and self.pgcr_lambda > 0:
            # Handle rel_labels as list (PE-NET passes list before cat)
            if isinstance(rel_labels, (list, tuple)):
                _labels_con = _fdi_torch.cat(rel_labels, dim=0).long()
            else:
                _labels_con = rel_labels.long()
            _N = rel_rep_norm.shape[0]

            # Subsample if too many (memory safety)
            _max_samples = 4096
            if _N > _max_samples:
                _perm = _fdi_torch.randperm(_N, device=rel_rep_norm.device)[:_max_samples]
                _rep_con = rel_rep_norm[_perm]
                _labels_con = _labels_con[_perm]
                _logits_con = calibrated_logits[_perm]
                _disp_con = _dac_dispersion[_perm]
            else:
                _rep_con = rel_rep_norm
                _logits_con = calibrated_logits
                _disp_con = _dac_dispersion

            _M = _rep_con.shape[0]
            # Only use foreground samples (label > 0)
            _fg_mask = (_labels_con > 0).float()

            # Project features
            _z = self.pgcr_projector(_rep_con)
            _z = _fdi_F.normalize(_z, dim=1)

            # Similarity matrix
            _sim = _fdi_torch.mm(_z, _z.t()) / self.tau_c  # [M, M]
            # Numerical stability
            _logits_max, _ = _sim.max(dim=1, keepdim=True)
            _sim = _sim - _logits_max.detach()

            # Masks
            _self_mask = 1.0 - _fdi_torch.eye(_M, device=_z.device)
            _label_eq = (_labels_con.unsqueeze(0) == _labels_con.unsqueeze(1)).float()
            _fg_row = _fg_mask.unsqueeze(1)  # [M, 1]
            _fg_col = _fg_mask.unsqueeze(0)  # [1, M]
            _pos_mask = _label_eq * _self_mask * _fg_row * _fg_col

            # --- Cascade dependency: hard negative mining from DAC ---
            with _fdi_torch.no_grad():
                _pred_labels = _logits_con.argmax(dim=1)
                _is_hard = (_pred_labels != _labels_con).float()
                _neg_weights = (1.0 - _pos_mask) * _self_mask * (1.0 + _is_hard.unsqueeze(0) * 0.5)

                # --- Cascade dependency: positive confidence from DAC ---
                _cal_conf = _fdi_F.softmax(_logits_con.detach(), dim=1)
                _gt_conf = _cal_conf.gather(1, _labels_con.unsqueeze(1)).squeeze(1)  # [M]
                _pos_conf = _gt_conf * _disp_con
                _pos_weights = _pos_mask * _pos_conf.unsqueeze(0) * _pos_conf.unsqueeze(1)

            # Denominator: sum of exp(sim) for all negatives
            _exp_sim = _fdi_torch.exp(_sim) * _neg_weights
            _neg_sum = _exp_sim.sum(dim=1, keepdim=True)  # [M, 1]

            # Log-prob of positives
            _log_prob = _sim - _fdi_torch.log(_neg_sum + _fdi_torch.exp(_sim) + 1e-10)

            # Weighted mean of positive log-probs
            _pos_weight_sum = _pos_weights.sum(dim=1)  # [M]
            _mean_log_prob_pos = (_pos_weights * _log_prob).sum(dim=1) / _pos_weight_sum.clamp(min=1e-6)

            # CB-Loss sample weights
            _sample_weights = self.pgcr_cb_weights[_labels_con.clamp(min=0, max=self.pgcr_cb_weights.shape[0]-1)]

            # Valid mask: foreground with at least one positive pair
            _valid = _fg_mask * (_pos_weight_sum > 1e-6).float()

            if _valid.sum() > 0:
                _pgcr_loss = self.pgcr_lambda * (
                    -(_sample_weights * _valid * _mean_log_prob_pos).sum()
                    / _valid.sum().clamp(min=1.0)
                )
            else:
                _pgcr_loss = _fdi_torch.tensor(0.0, device=_z.device)

            additional_losses["pgcr_loss"] = _pgcr_loss
        elif self.training:
            additional_losses["pgcr_loss"] = _fdi_torch.tensor(0.0, device=rel_dists.device)

        return calibrated_logits, additional_losses
'''
    return code


def generate_init_code():
    """Generate code to initialize FDI-Net in PrototypeEmbeddingNetwork.__init__"""
    return '''
        # === FDI-Net initialization ===
        _fdi_feat_dim = self.mlp_dim * 2  # rel_rep dimension after project_head
        _fdi_num_pred = self.num_rel_cls   # number of predicate classes
        self.fdi_net = FDINet(_fdi_feat_dim, _fdi_num_pred)
'''


def generate_forward_code():
    """Generate code that replaces the rel_dists computation line.

    This replaces:
        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()
    With:
        original computation + DAC calibration + store losses on self for later injection

    CRITICAL: We store _fdi_losses on self (self._fdi_losses_cache) because
    add_losses doesn't exist yet at this point in PE-NET's forward().
    The inject code (PATCH 4) will read from self._fdi_losses_cache later.
    """
    return '''        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()
        # === FDI-Net: DAC calibration + PGCR contrastive ===
        rel_dists, self._fdi_losses_cache = self.fdi_net(rel_rep_norm, rel_dists, rel_labels)
'''


def generate_inject_code():
    """Generate code to inject cached PGCR losses into add_losses.

    This is inserted AFTER add_losses.update({"loss_dis": ...}) where
    add_losses dict already exists.

    Reads from self._fdi_losses_cache which was stored by the forward code.

    CRITICAL: Indentation must be 12 spaces (same level as the
    add_losses.update lines inside PE-NET's if self.training: block).
    """
    # Each line must have exactly 12 spaces of indentation
    return (
        '            # === FDI-Net: inject PGCR losses ===\n'
        '            if hasattr(self, "_fdi_losses_cache") and self._fdi_losses_cache:\n'
        '                add_losses.update(self._fdi_losses_cache)\n'
    )


def patch_predictors(penet_root, args):
    """Apply all patches to roi_relation_predictors.py"""
    pred_path = os.path.join(
        penet_root,
        "maskrcnn_benchmark", "modeling", "roi_heads",
        "relation_head", "roi_relation_predictors.py"
    )

    if not os.path.exists(pred_path):
        print(f"ERROR: {pred_path} not found!")
        return False

    # Create backup
    bak_path = pred_path + ".bak_fdinet"
    if not os.path.exists(bak_path):
        shutil.copy2(pred_path, bak_path)
        print(f"  Backup saved to {bak_path}")

    with open(pred_path, "r") as f:
        content = f.read()
    lines = content.split("\n")

    # Check if already patched
    if "class FDINet" in content:
        print("  Already patched (FDINet class found). Skipping.")
        return True

    # ==========================================
    # PATCH 1: Inject FDINet module definition
    # ==========================================
    # Must insert BEFORE the @registry decorator, not between decorator and class
    class_line = None
    for i, line in enumerate(lines):
        if "class PrototypeEmbeddingNetwork" in line and "nn.Module" in line:
            class_line = i
            break

    if class_line is None:
        print("  ERROR: Cannot find 'class PrototypeEmbeddingNetwork'")
        return False

    # Check if there's a decorator above the class line — insert before it
    insert_line = class_line
    for j in range(class_line - 1, max(class_line - 5, -1), -1):
        if lines[j].strip().startswith("@"):
            insert_line = j  # insert before the decorator
            break

    module_code = generate_fdinet_module(
        args.alpha, args.lambda_pgcr, args.beta, args.tau_c, args.residual_scale
    )
    lines.insert(insert_line, module_code)
    print(f"  PATCH 1: FDI-Net module code injected before line {insert_line}.")

    content = "\n".join(lines)
    lines = content.split("\n")

    # ==========================================
    # PATCH 2: Add FDI-Net init after logit_scale
    # ==========================================
    init_line = None
    for i, line in enumerate(lines):
        if "self.logit_scale" in line and "nn.Parameter" in line:
            init_line = i
            break

    if init_line is None:
        print("  ERROR: Cannot find 'self.logit_scale = nn.Parameter'")
        return False

    init_code = generate_init_code()
    lines.insert(init_line + 1, init_code)
    print(f"  PATCH 2: FDI-Net init added after logit_scale (line {init_line + 1}).")

    content = "\n".join(lines)
    lines = content.split("\n")

    # ==========================================
    # PATCH 3: Replace forward rel_dists line
    # ==========================================
    forward_line = None
    for i, line in enumerate(lines):
        if "rel_rep_norm" in line and "predicate_proto_norm" in line and "logit_scale" in line:
            forward_line = i
            break

    if forward_line is None:
        print("  ERROR: Cannot find rel_dists computation line in forward()")
        return False

    forward_code = generate_forward_code()
    lines[forward_line] = forward_code
    print(f"  PATCH 3: Forward code replaced at line {forward_line}.")

    content = "\n".join(lines)
    lines = content.split("\n")

    # ==========================================
    # PATCH 4: Inject PGCR loss into add_losses
    # ==========================================
    # PE-NET uses: add_losses.update({"loss_dis": loss_sum})
    # We need to find this ONLY inside PrototypeEmbeddingNetwork class

    # First, find the class boundaries
    proto_class_start = None
    proto_class_end = None
    for i, line in enumerate(lines):
        if "class PrototypeEmbeddingNetwork" in line:
            proto_class_start = i
        elif proto_class_start is not None and line.strip().startswith("class ") and i > proto_class_start + 10:
            proto_class_end = i
            break

    if proto_class_end is None:
        proto_class_end = len(lines)

    inject_line = None

    if proto_class_start is not None:
        # Strategy 1: Find add_losses.update({"loss_dis"...})
        for i in range(proto_class_start, proto_class_end):
            if "loss_dis" in lines[i] and "add_losses" in lines[i]:
                inject_line = i
                break

        # Strategy 2: Find any add_losses.update({" line
        if inject_line is None:
            for i in range(proto_class_start, proto_class_end):
                if "add_losses.update" in lines[i] and "loss" in lines[i]:
                    inject_line = i  # keep searching to get the LAST one
            # inject_line now points to the last add_losses.update line

        # Strategy 3: Find return line with add_losses and search backwards
        if inject_line is None:
            for i in range(proto_class_start, proto_class_end):
                if "return" in lines[i] and "add_losses" in lines[i]:
                    for j in range(i - 1, max(proto_class_start, i - 50), -1):
                        if "add_losses" in lines[j] and lines[j].strip() != "":
                            inject_line = j
                            break
                    break

    if inject_line is None:
        print("  ERROR: Cannot find add_losses in PrototypeEmbeddingNetwork.")
        if proto_class_start is not None:
            print(f"  Searched lines {proto_class_start}-{proto_class_end}")
            print("  All add_losses lines found:")
            for i in range(proto_class_start, proto_class_end):
                if "add_losses" in lines[i]:
                    print(f"    Line {i}: {lines[i].rstrip()}")
        return False

    inject_code = generate_inject_code()
    lines.insert(inject_line + 1, inject_code)
    print(f"  PATCH 4: PGCR loss injection added after line {inject_line}.")
    print(f"    Anchor: {lines[inject_line].strip()[:70]}")

    # Write patched file
    content = "\n".join(lines)
    with open(pred_path, "w") as f:
        f.write(content)

    print("\n  All 4 patches applied successfully!")
    return True


def verify_loss_py(penet_root):
    """Read-only check of loss.py — does NOT modify it."""
    loss_path = os.path.join(
        penet_root,
        "maskrcnn_benchmark", "modeling", "roi_heads",
        "relation_head", "loss.py"
    )
    if not os.path.exists(loss_path):
        print(f"  WARNING: {loss_path} not found")
        return

    with open(loss_path, "r") as f:
        content = f.read()

    if "cb_loss_beta=0.9999" in content:
        print("  loss.py: CB-Loss FOUND (cb_loss_beta=0.9999). Good.")
    elif "0.9999" in content:
        print("  loss.py: Found 0.9999 reference. Likely has CB-Loss.")
    else:
        print("  *** WARNING: loss.py has NO CB-Loss! ***")
        print("  *** You must use the proven loss.py with cb_loss_beta=0.9999 ***")
        print("  *** Otherwise mR@50 will only reach ~31 instead of ~41 ***")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FDI-Net v4 Patch (loss.py-safe)")
    parser.add_argument("penet_root", help="Path to penet-main directory")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--lambda_pgcr", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.9999)
    parser.add_argument("--tau_c", type=float, default=0.07)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    args = parser.parse_args()

    print("=" * 60)
    print("FDI-Net v4 Patch (loss.py-safe version)")
    print("=" * 60)
    print(f"  alpha={args.alpha}, lambda_pgcr={args.lambda_pgcr}")
    print(f"  beta={args.beta}, tau_c={args.tau_c}, residual_scale={args.residual_scale}")

    # Step 1: Check loss.py (READ ONLY)
    print("\nStep 1: Checking loss.py (read-only)...")
    verify_loss_py(args.penet_root)

    # Step 2: Patch predictors.py
    print("\nStep 2: Patching roi_relation_predictors.py...")
    success = patch_predictors(args.penet_root, args)

    if success:
        print("\n" + "=" * 60)
        print("DONE. Patch applied successfully.")
        print(f"  Reminder: Ensure loss.py has cb_loss_beta=0.9999!")
        print("=" * 60)
    else:
        print("\nFAILED. See errors above.")
        sys.exit(1)
