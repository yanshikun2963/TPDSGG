"""
patch_cptr_v2.py: CPTR (Confusion-Pair Targeted Repulsion) -- IMPROVED.

Fixes vs v1 (which collapsed R from 63 -> 53):
  FIX 1  Detach the HEAD prototype. v1 used predicate_proto_norm[head]
         directly in the loss, so the head prototype received gradient and
         got pushed -> head decision boundaries broke -> R collapse. v2
         detaches it: only the TAIL prototype is moved (consistent with
         TEPA's tail-only philosophy).
  FIX 2  CPTR_WEIGHT default 0.2 -> 0.05 (4x gentler).
  FIX 3  New CPTR_WARMUP_ITER (default 5000): CPTR loss only kicks in after
         TEPA has settled, so it does not fight TEPA during early alignment.

Self-reverting + idempotent: if an earlier CPTR patch is detected, the
pre-CPTR state is restored from .backup_cptr first, then v2 is applied.
So you can just run this directly on a machine that already ran v1.

Run from project root:
    cd /root/autodl-tmp/penet-main
    python3 patch_cptr_v2.py
"""
import os
import sys

PROJECT_DIR = "/root/autodl-tmp/penet-main"
PREDICTOR = os.path.join(PROJECT_DIR,
    "maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py")
DEFAULTS = os.path.join(PROJECT_DIR,
    "maskrcnn_benchmark/config/defaults.py")

DEFAULTS_ADD = """

# ============================================================
# CPTR: Confusion-Pair Targeted Repulsion (v2)
# ============================================================
_C.MODEL.ROI_RELATION_HEAD.USE_CPTR = False
_C.MODEL.ROI_RELATION_HEAD.CPTR_PAIRS_PATH = ""
_C.MODEL.ROI_RELATION_HEAD.CPTR_WEIGHT = 0.05
_C.MODEL.ROI_RELATION_HEAD.CPTR_MARGIN = 0.3
_C.MODEL.ROI_RELATION_HEAD.CPTR_K_SHARPNESS = 10.0
_C.MODEL.ROI_RELATION_HEAD.CPTR_WARMUP_ITER = 5000
"""

# ------------------------------------------------------------
# predictor.py - Patch A: load CPTR pairs in __init__
# Anchor: tail of the v1-TEPA-patch B1 init block
# ------------------------------------------------------------
PREDICTOR_INIT_OLD = """            else:
                print(f"[Component] No stats init; will accumulate EMA during training (warmup applies).")"""

PREDICTOR_INIT_NEW = """            else:
                print(f"[Component] No stats init; will accumulate EMA during training (warmup applies).")

        # ============================================================
        # CPTR: Confusion-Pair Targeted Repulsion (v2)
        # ============================================================
        self.use_cptr = getattr(_rcfg, 'USE_CPTR', False)
        if self.use_cptr:
            import numpy as _np
            _cptr_path = getattr(_rcfg, 'CPTR_PAIRS_PATH', '')
            assert _cptr_path and _os.path.exists(_cptr_path), \\
                f"[CPTR] pairs file missing: {_cptr_path}"
            _cptr_pairs = _np.load(_cptr_path)  # (P, 3): [tail, head, weight]
            self.register_buffer('cptr_tail', torch.tensor(_cptr_pairs[:, 0], dtype=torch.long))
            self.register_buffer('cptr_head', torch.tensor(_cptr_pairs[:, 1], dtype=torch.long))
            self.register_buffer('cptr_weight', torch.tensor(_cptr_pairs[:, 2], dtype=torch.float32))
            print("=" * 60)
            print(f"[CPTR] ENABLED (v2: head-detached, gentle weight, warmup)")
            print(f"[CPTR]   loaded {len(_cptr_pairs)} confused (tail, head) pairs")
            print(f"[CPTR]   weight={getattr(_rcfg, 'CPTR_WEIGHT', 0.05)}, "
                  f"margin={getattr(_rcfg, 'CPTR_MARGIN', 0.3)}, "
                  f"k_sharp={getattr(_rcfg, 'CPTR_K_SHARPNESS', 10.0)}, "
                  f"warmup={getattr(_rcfg, 'CPTR_WARMUP_ITER', 5000)}")
            print("=" * 60)
        else:
            print("[CPTR] DISABLED")"""

# ------------------------------------------------------------
# predictor.py - Patch B: add CPTR loss inside training branch
# Anchor: line right after l21_loss block (### end marker)
# ------------------------------------------------------------
PREDICTOR_FWD_OLD = """            add_losses.update({\"l21_loss\": l21})  # Le_sim = ||S||_{2,1}
            ### end"""

PREDICTOR_FWD_NEW = """            add_losses.update({\"l21_loss\": l21})  # Le_sim = ||S||_{2,1}
            ### end

            # ============================================================
            # CPTR v2: Confusion-Pair Targeted Repulsion
            # Pushes the TAIL prototype away from the prototypes of HEAD
            # predicates it tends to be confused with. The head prototype is
            # DETACHED (FIX 1): only the tail prototype moves, so head
            # decision boundaries -- and therefore R@50 -- are preserved.
            # Gated by a warmup (FIX 3) so TEPA settles first.
            # ============================================================
            if self.use_cptr and self.cptr_tail.numel() > 0:
                _rcfg_cptr = self.cfg.MODEL.ROI_RELATION_HEAD
                _cptr_warmup = getattr(_rcfg_cptr, 'CPTR_WARMUP_ITER', 5000)
                if int(self._comp_iter.item()) >= _cptr_warmup:
                    _cptr_w = getattr(_rcfg_cptr, 'CPTR_WEIGHT', 0.05)
                    _cptr_m = getattr(_rcfg_cptr, 'CPTR_MARGIN', 0.3)
                    _cptr_k = getattr(_rcfg_cptr, 'CPTR_K_SHARPNESS', 10.0)
                    # FIX 1: detach the head side -> one-directional push
                    _proto_tail = predicate_proto_norm[self.cptr_tail]
                    _proto_head = predicate_proto_norm[self.cptr_head].detach()
                    _cos = (_proto_tail * _proto_head).sum(dim=-1)
                    # smoothed hinge: penalize cos(tail, head) > margin
                    _hinge = F.softplus(_cptr_k * (_cos - _cptr_m)) / _cptr_k
                    _cptr_loss = (_hinge * self.cptr_weight).mean() * _cptr_w
                    add_losses.update({\"loss_cptr\": _cptr_loss})"""


def _restore_or_backup(path, backup_suffix):
    """If a prior patch backup exists, restore from it (revert old patch);
    otherwise create the backup. Returns the (pre-patch) file content."""
    backup = path + backup_suffix
    if os.path.exists(backup):
        with open(backup) as f:
            content = f.read()
        with open(path, 'w') as f:
            f.write(content)
        print(f"[revert] {os.path.basename(path)} restored from {os.path.basename(backup)}")
    else:
        with open(path) as f:
            content = f.read()
        with open(backup, 'w') as f:
            f.write(content)
        print(f"[backup] {os.path.basename(path)} -> {os.path.basename(backup)}")
    return content


def patch_defaults():
    content = _restore_or_backup(DEFAULTS, ".backup_cptr")
    with open(DEFAULTS, 'a') as f:
        f.write(DEFAULTS_ADD)
    print("[ok]   defaults.py: CPTR v2 flags added (CPTR_WEIGHT=0.05, +CPTR_WARMUP_ITER)")


def patch_predictor():
    content = _restore_or_backup(PREDICTOR, ".backup_cptr")

    assert PREDICTOR_INIT_OLD in content, \
        "predictor.py: CPTR init anchor not found (is the TEPA v1 patch applied?)"
    content = content.replace(PREDICTOR_INIT_OLD, PREDICTOR_INIT_NEW, 1)

    assert PREDICTOR_FWD_OLD in content, \
        "predictor.py: CPTR forward anchor (l21_loss block) not found"
    content = content.replace(PREDICTOR_FWD_OLD, PREDICTOR_FWD_NEW, 1)

    with open(PREDICTOR, 'w') as f:
        f.write(content)
    print("[ok]   predictor.py: CPTR v2 init + head-detached loss applied")


if __name__ == "__main__":
    print("=" * 60)
    print("  CPTR Patch v2  (head-detached + gentle weight + warmup)")
    print("=" * 60)
    patch_defaults()
    patch_predictor()
    print("\n[done] CPTR v2 patch complete.")
    print("\nKey changes vs v1:")
    print("  FIX 1: head prototype detached -> only tail prototype moves")
    print("  FIX 2: CPTR_WEIGHT 0.2 -> 0.05")
    print("  FIX 3: CPTR_WARMUP_ITER 5000 (CPTR loss starts after TEPA settles)")
