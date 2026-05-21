"""
Automatically patch maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py
to add A1 (EMA Memory Bank) and B1 (Prototype Alignment) components.

Idempotent: safe to run multiple times.
Usage:
    python3 patch_predictor.py
"""
import os
import sys

PROJECT_DIR = "/root/autodl-tmp/penet-main"
TARGET = os.path.join(PROJECT_DIR, "maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py")

# ============================================================
# Patch 1: __init__ 结尾追加 component setup
# 锚点: "self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES"
# ============================================================
INIT_ANCHOR = "        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES"

INIT_INSERT = '''

        # ============================================================
        # New components: A1 (EMA Memory Bank), B1 (Prototype Alignment)
        # ============================================================
        import os as _os
        self.register_buffer('_comp_iter', torch.tensor(0, dtype=torch.long))

        _rcfg = config.MODEL.ROI_RELATION_HEAD
        self.use_a1 = getattr(_rcfg, 'USE_A1_MEMORY', False)
        self.use_b1 = getattr(_rcfg, 'USE_B1_ALIGN', False)

        # A1/B1 shared class statistics
        if self.use_a1 or self.use_b1:
            _feat_dim = self.mlp_dim * 2  # 4096 (after project_head)
            self.register_buffer('class_mean', torch.zeros(self.num_rel_cls, _feat_dim))
            self.register_buffer('class_var', torch.ones(self.num_rel_cls, _feat_dim))
            self.register_buffer('class_count', torch.zeros(self.num_rel_cls))

            _stats_path = ''
            if self.use_a1:
                _stats_path = getattr(_rcfg, 'A1_STATS_INIT_PATH', '')
            if not _stats_path and self.use_b1:
                _stats_path = getattr(_rcfg, 'B1_STATS_INIT_PATH', '')
            if _stats_path and _os.path.exists(_stats_path):
                _stats = torch.load(_stats_path, map_location='cpu')
                self.class_mean.copy_(_stats['class_mean'])
                self.class_var.copy_(_stats['class_var'])
                self.class_count.copy_(_stats['class_count'])
                print(f"[Component] Loaded class stats from {_stats_path}")
                print(f"[Component]   class_count: min={self.class_count.min().item():.0f}, max={self.class_count.max().item():.0f}")
            else:
                print(f"[Component] No stats init; will accumulate EMA during training (warmup applies).")
'''

# ============================================================
# Patch 2: forward training branch 末尾追加 component logic
# 锚点: 'add_losses.update({"loss_dis": loss_sum})'
# ============================================================
FORWARD_ANCHOR = '            add_losses.update({"loss_dis": loss_sum})     # Le_euc = max(0, (g+) - (g-) + gamma1)'

FORWARD_INSERT = '''

            # ============================================================
            # New components: A1 (EMA Memory Bank), B1 (Prototype Alignment)
            # ============================================================
            if self.use_a1 or self.use_b1:
                self._comp_iter += 1
                with torch.no_grad():
                    self._update_class_stats(rel_rep, rel_labels)
                if self.use_a1:
                    _loss_a1 = self._a1_synth_loss(predicate_proto_norm)
                    if _loss_a1 is not None:
                        add_losses.update({"loss_a1_aug": _loss_a1})
                if self.use_b1:
                    _loss_b1 = self._b1_align_loss(predicate_proto)
                    if _loss_b1 is not None:
                        add_losses.update({"loss_b1_align": _loss_b1})'''

# ============================================================
# Patch 3: helper methods (insert before 'def refine_obj_labels')
# 锚点: "    def refine_obj_labels(self, roi_features, proposals):"
# ============================================================
METHODS_ANCHOR = "    def refine_obj_labels(self, roi_features, proposals):"

METHODS_INSERT = '''    # ============================================================
    # Helper methods for new components (A1, B1)
    # ============================================================
    def _update_class_stats(self, rel_rep, rel_labels):
        """Update per-class mean and variance via EMA."""
        _rcfg = self.cfg.MODEL.ROI_RELATION_HEAD
        momentum = getattr(_rcfg, 'A1_EMA_MOMENTUM', 0.95)
        rel_rep_d = rel_rep.detach()
        for c in torch.unique(rel_labels):
            c_int = int(c.item())
            if c_int == 0:
                continue
            mask = rel_labels == c
            n_c = int(mask.sum().item())
            if n_c == 0:
                continue
            batch_mean = rel_rep_d[mask].mean(0)
            if n_c > 1:
                batch_var = rel_rep_d[mask].var(0, unbiased=False) + 1e-3
            else:
                batch_var = torch.ones_like(batch_mean)

            if self.class_count[c_int].item() < 5:
                self.class_mean[c_int] = batch_mean
                self.class_var[c_int] = batch_var
            else:
                self.class_mean[c_int].mul_(momentum).add_(batch_mean, alpha=1.0 - momentum)
                self.class_var[c_int].mul_(momentum).add_(batch_var, alpha=1.0 - momentum)
            self.class_count[c_int] += n_c

    def _a1_synth_loss(self, predicate_proto_norm):
        """A1: Generate synthetic features for tail classes; return CE loss (or None during warmup)."""
        _rcfg = self.cfg.MODEL.ROI_RELATION_HEAD
        warmup = getattr(_rcfg, 'A1_WARMUP_ITER', 3000)
        if int(self._comp_iter.item()) < warmup:
            return None

        n_per_class = getattr(_rcfg, 'A1_N_AUG_PER_CLASS', 8)
        aug_w = getattr(_rcfg, 'A1_AUG_WEIGHT', 0.3)

        rel_prop = list(_rcfg.REL_PROP)
        freq = [0.0] + rel_prop
        med = sorted(rel_prop)[len(rel_prop) // 2]
        tail_classes = [c for c in range(1, self.num_rel_cls)
                       if freq[c] < med and self.class_count[c].item() >= 20]
        if len(tail_classes) == 0:
            return None

        device = predicate_proto_norm.device
        aug_feats_list = []
        aug_labels_list = []
        for c in tail_classes:
            mean = self.class_mean[c].to(device)
            std = self.class_var[c].to(device).clamp(min=1e-6).sqrt()
            eps = torch.randn(n_per_class, mean.shape[0], device=device)
            samples = mean + 0.5 * eps * std
            aug_feats_list.append(samples)
            aug_labels_list.append(torch.full((n_per_class,), c, dtype=torch.long, device=device))

        aug_feats = torch.cat(aug_feats_list, dim=0)
        aug_labels = torch.cat(aug_labels_list, dim=0)

        aug_norm = aug_feats / (aug_feats.norm(dim=1, keepdim=True) + 1e-8)
        # NOTE: do NOT detach predicate_proto_norm here. If we detach, the only
        # learnable tensor in aug_logits is logit_scale (a single scalar), and
        # the synthetic loss becomes effectively useless. With aug_w=0.3 the
        # influence on prototypes is already bounded.
        aug_logits = aug_norm @ predicate_proto_norm.t() * self.logit_scale.exp()
        aug_logits = aug_logits.clamp(-20, 20)
        loss = F.cross_entropy(aug_logits, aug_labels) * aug_w
        return loss

    def _b1_align_loss(self, predicate_proto):
        """B1: Cosine alignment between MLP-prototypes and empirical class-mean."""
        _rcfg = self.cfg.MODEL.ROI_RELATION_HEAD
        warmup = getattr(_rcfg, 'B1_WARMUP_ITER', 5000)
        if int(self._comp_iter.item()) < warmup:
            return None

        min_count = getattr(_rcfg, 'B1_MIN_COUNT', 100)
        align_w = getattr(_rcfg, 'B1_ALIGN_WEIGHT', 0.2)

        active = (self.class_count >= min_count).clone()
        active[0] = False
        if not active.any():
            return None
        idx = active.nonzero(as_tuple=False).squeeze(-1)

        mlp_proto = predicate_proto[idx]
        emp_proto = self.class_mean[idx].detach()

        mlp_norm = mlp_proto / (mlp_proto.norm(dim=-1, keepdim=True) + 1e-8)
        emp_norm = emp_proto / (emp_proto.norm(dim=-1, keepdim=True) + 1e-8)

        loss = (1 - (mlp_norm * emp_norm).sum(dim=-1)).mean() * align_w
        return loss

'''


def main():
    if not os.path.exists(TARGET):
        print(f"[ERROR] Target file not found: {TARGET}")
        sys.exit(1)

    with open(TARGET, 'r') as f:
        content = f.read()

    # backup
    backup = TARGET + ".backup_components"
    if not os.path.exists(backup):
        with open(backup, 'w') as f:
            f.write(content)
        print(f"[ok] Backed up to {backup}")

    # Check idempotency
    if "self.use_a1 = getattr" in content:
        print("[skip] Already patched (use_a1 flag found). Nothing to do.")
        return

    # ===== Patch 1: __init__ =====
    if INIT_ANCHOR not in content:
        print(f"[ERROR] init anchor not found:\n  {INIT_ANCHOR}")
        sys.exit(1)
    content = content.replace(INIT_ANCHOR, INIT_ANCHOR + INIT_INSERT, 1)
    print("[ok] Patched __init__")

    # ===== Patch 2: forward =====
    if FORWARD_ANCHOR not in content:
        print(f"[ERROR] forward anchor not found:\n  {FORWARD_ANCHOR}")
        sys.exit(1)
    content = content.replace(FORWARD_ANCHOR, FORWARD_ANCHOR + FORWARD_INSERT, 1)
    print("[ok] Patched forward")

    # ===== Patch 3: helper methods =====
    if METHODS_ANCHOR not in content:
        print(f"[ERROR] methods anchor not found:\n  {METHODS_ANCHOR}")
        sys.exit(1)
    content = content.replace(METHODS_ANCHOR, METHODS_INSERT + METHODS_ANCHOR, 1)
    print("[ok] Inserted helper methods")

    with open(TARGET, 'w') as f:
        f.write(content)

    print(f"\n[done] Patched {TARGET}")
    print(f"[note] Original backup: {backup}")


if __name__ == "__main__":
    main()
