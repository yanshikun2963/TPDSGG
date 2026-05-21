"""
patch_cptr.py: Add CPTR (Confusion-Pair Targeted Repulsion) component.

Modifies (idempotent):
1. maskrcnn_benchmark/config/defaults.py            -> add USE_CPTR and CPTR_* flags
2. maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py
                                                    -> load confusion pairs in __init__
                                                    -> add CPTR loss after l21_loss block

Run from project root:
    python3 patch_cptr.py
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
# CPTR: Confusion-Pair Targeted Repulsion
# ============================================================
_C.MODEL.ROI_RELATION_HEAD.USE_CPTR = False
_C.MODEL.ROI_RELATION_HEAD.CPTR_PAIRS_PATH = ""
_C.MODEL.ROI_RELATION_HEAD.CPTR_WEIGHT = 0.2
_C.MODEL.ROI_RELATION_HEAD.CPTR_MARGIN = 0.3
_C.MODEL.ROI_RELATION_HEAD.CPTR_K_SHARPNESS = 10.0
"""

# ============================================================
# predictor.py - Patch A: load CPTR pairs in __init__
# Anchor: tail of the v1-patch B1 init block
# ============================================================
PREDICTOR_INIT_OLD = """            else:
                print(f"[Component] No stats init; will accumulate EMA during training (warmup applies).")"""

PREDICTOR_INIT_NEW = """            else:
                print(f"[Component] No stats init; will accumulate EMA during training (warmup applies).")

        # ============================================================
        # CPTR: Confusion-Pair Targeted Repulsion
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
            print(f"[CPTR] ENABLED")
            print(f"[CPTR]   loaded {len(_cptr_pairs)} confused (tail, head) pairs")
            print(f"[CPTR]   weight={getattr(_rcfg, 'CPTR_WEIGHT', 0.2)}, "
                  f"margin={getattr(_rcfg, 'CPTR_MARGIN', 0.3)}, "
                  f"k_sharp={getattr(_rcfg, 'CPTR_K_SHARPNESS', 10.0)}")
            print("=" * 60)
        else:
            print("[CPTR] DISABLED")"""

# ============================================================
# predictor.py - Patch B: add CPTR loss inside training branch
# Anchor: line right after l21_loss block (### end marker)
# ============================================================
PREDICTOR_FWD_OLD = """            add_losses.update({\"l21_loss\": l21})  # Le_sim = ||S||_{2,1}
            ### end"""

PREDICTOR_FWD_NEW = """            add_losses.update({\"l21_loss\": l21})  # Le_sim = ||S||_{2,1}
            ### end

            # ============================================================
            # CPTR: Confusion-Pair Targeted Repulsion
            # Pushes apart prototypes of (tail, head) pairs that the baseline
            # was likely to confuse, based on (s, o) distribution overlap.
            # ============================================================
            if self.use_cptr:
                _rcfg_cptr = self.cfg.MODEL.ROI_RELATION_HEAD
                _cptr_w = getattr(_rcfg_cptr, 'CPTR_WEIGHT', 0.2)
                _cptr_m = getattr(_rcfg_cptr, 'CPTR_MARGIN', 0.3)
                _cptr_k = getattr(_rcfg_cptr, 'CPTR_K_SHARPNESS', 10.0)
                # predicate_proto_norm is computed above (outside this if-block)
                # and has gradients flowing to W_pred. Cosine similarity between
                # tail and head prototypes; smoothed hinge penalizes cos > margin.
                _cos = (predicate_proto_norm[self.cptr_tail] *
                        predicate_proto_norm[self.cptr_head]).sum(dim=-1)
                _hinge = F.softplus(_cptr_k * (_cos - _cptr_m)) / _cptr_k
                _cptr_loss = (_hinge * self.cptr_weight).mean() * _cptr_w
                add_losses.update({\"loss_cptr\": _cptr_loss})"""


def patch_defaults():
    with open(DEFAULTS) as f:
        content = f.read()
    if "USE_CPTR" in content:
        print("[skip] defaults.py: USE_CPTR already present")
        return
    backup = DEFAULTS + ".backup_cptr"
    if not os.path.exists(backup):
        with open(backup, 'w') as f:
            f.write(content)
    with open(DEFAULTS, 'a') as f:
        f.write(DEFAULTS_ADD)
    print("[ok]   defaults.py: added CPTR flags")


def patch_predictor():
    with open(PREDICTOR) as f:
        content = f.read()
    if "self.use_cptr" in content:
        print("[skip] predictor.py: CPTR already patched")
        return
    backup = PREDICTOR + ".backup_cptr"
    if not os.path.exists(backup):
        with open(backup, 'w') as f:
            f.write(content)

    assert PREDICTOR_INIT_OLD in content, "predictor.py: CPTR init anchor not found"
    content = content.replace(PREDICTOR_INIT_OLD, PREDICTOR_INIT_NEW, 1)

    assert PREDICTOR_FWD_OLD in content, "predictor.py: CPTR forward anchor not found"
    content = content.replace(PREDICTOR_FWD_OLD, PREDICTOR_FWD_NEW, 1)

    with open(PREDICTOR, 'w') as f:
        f.write(content)
    print("[ok]   predictor.py: CPTR init + loss added")


if __name__ == "__main__":
    print("=" * 60)
    print("  CPTR Patch")
    print("=" * 60)
    patch_defaults()
    patch_predictor()
    print("\n[done] CPTR patch complete.")
    print("\nNew cfg options (defaults shown):")
    print("  USE_CPTR=False")
    print("  CPTR_PAIRS_PATH=''")
    print("  CPTR_WEIGHT=0.2  CPTR_MARGIN=0.3  CPTR_K_SHARPNESS=10.0")
