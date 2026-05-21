"""
patch_stage2.py: 升级 B1 -> TEPA + RPR (Two-Stage DPL Framework, Stage 2)

在已经 patched B1 的 roi_relation_predictors.py 基础上:
1. _b1_align_loss 加入 tail-only filter (TEPA)
2. l21_loss 加入条件性 PR boost (RPR)
3. defaults.py 加入新 config: B1_TAIL_ONLY, B1_PR_BOOST

Idempotent: 重复跑无副作用。

Usage:
    cd /root/autodl-tmp/penet-main
    python3 patch_stage2.py
"""
import os
import sys
import subprocess

PROJECT_DIR = "/root/autodl-tmp/penet-main"
PREDICTOR = os.path.join(PROJECT_DIR, "maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py")
DEFAULTS = os.path.join(PROJECT_DIR, "maskrcnn_benchmark/config/defaults.py")

# ============================================================
# Patch 1: defaults.py - add new cfg flags
# ============================================================
DEFAULTS_ADD = """

# ============================================================
# Stage 2 (DPL framework): TEPA + RPR settings
# ============================================================
_C.MODEL.ROI_RELATION_HEAD.B1_TAIL_ONLY = True   # TEPA: only align tail classes (freq < median)
_C.MODEL.ROI_RELATION_HEAD.B1_PR_BOOST = 1.0     # RPR: l21_loss multiplier (1.0=disabled, >1.0=stronger PR)
"""

# ============================================================
# Patch 2: predictor.py - _b1_align_loss tail-only (TEPA)
# Anchor: the part after warmup check, before "if not active.any()"
# ============================================================
B1_OLD = """        min_count = getattr(_rcfg, 'B1_MIN_COUNT', 100)
        align_w = getattr(_rcfg, 'B1_ALIGN_WEIGHT', 0.2)

        active = (self.class_count >= min_count).clone()
        active[0] = False
        if not active.any():"""

B1_NEW = """        min_count = getattr(_rcfg, 'B1_MIN_COUNT', 100)
        align_w = getattr(_rcfg, 'B1_ALIGN_WEIGHT', 0.2)
        tail_only = getattr(_rcfg, 'B1_TAIL_ONLY', True)

        active = (self.class_count >= min_count).clone()
        active[0] = False
        if tail_only:
            # TEPA: only align tail classes (freq < median frequency)
            # to preserve head prototype discriminability established by PR
            rel_prop = list(_rcfg.REL_PROP)
            freq = [0.0] + list(rel_prop)  # index 0 = background
            median_freq = sorted(rel_prop)[len(rel_prop) // 2]
            for _c in range(1, self.num_rel_cls):
                if freq[_c] >= median_freq:
                    active[_c] = False
        if not active.any():"""

# ============================================================
# Patch 3: predictor.py - l21_loss with conditional PR boost (RPR)
# Anchor: line 247-248 of original PE-Net predictor.py
# Note: line 247 has TWO trailing spaces (intentional, preserve for exact match)
# ============================================================
L21_OLD = """            l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (51*51)  
            add_losses.update({\"l21_loss\": l21})  # Le_sim = ||S||_{2,1}"""

L21_NEW = """            # RPR: Reinforced Prototype Regularization
            # Compensates for B1's tendency to collapse prototype space (mlp_proto -> emp_mean
            # pulls prototypes toward each other since empirical means cluster). Boost only
            # activates when B1 is on; preserves original PE-Net behavior otherwise.
            _pr_boost = getattr(self.cfg.MODEL.ROI_RELATION_HEAD, 'B1_PR_BOOST', 1.0) if getattr(self, 'use_b1', False) else 1.0
            l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (51*51) * _pr_boost
            add_losses.update({\"l21_loss\": l21})  # Le_sim = ||S||_{2,1}"""


def patch_defaults():
    """Add B1_TAIL_ONLY and B1_PR_BOOST cfg flags to defaults.py"""
    if not os.path.exists(DEFAULTS):
        print(f"[ERROR] defaults.py not found: {DEFAULTS}")
        sys.exit(1)
    with open(DEFAULTS) as f:
        content = f.read()
    if "B1_TAIL_ONLY" in content:
        print("[skip] defaults.py: B1_TAIL_ONLY already present")
        return
    # backup
    backup = DEFAULTS + ".backup_stage2"
    if not os.path.exists(backup):
        with open(backup, 'w') as f:
            f.write(content)
        print(f"       backup -> {backup}")
    with open(DEFAULTS, 'a') as f:
        f.write(DEFAULTS_ADD)
    print("[ok]   defaults.py: added B1_TAIL_ONLY and B1_PR_BOOST")


def patch_predictor():
    """Add TEPA filter to _b1_align_loss, and RPR boost to l21_loss"""
    if not os.path.exists(PREDICTOR):
        print(f"[ERROR] predictor.py not found: {PREDICTOR}")
        sys.exit(1)
    with open(PREDICTOR) as f:
        content = f.read()
    # backup
    backup = PREDICTOR + ".backup_stage2"
    if not os.path.exists(backup):
        with open(backup, 'w') as f:
            f.write(content)
        print(f"       backup -> {backup}")

    # Patch 2: TEPA tail-only filter
    if 'tail_only = getattr' in content:
        print("[skip] predictor.py: TEPA already patched")
    elif B1_OLD in content:
        content = content.replace(B1_OLD, B1_NEW, 1)
        print("[ok]   predictor.py: TEPA tail-only filter applied")
    else:
        print("[ERROR] predictor.py: TEPA anchor (B1_OLD) not found.")
        print("        Make sure patch_predictor.py (v1) was applied first.")
        sys.exit(1)

    # Patch 3: RPR conditional boost on l21_loss
    if 'B1_PR_BOOST' in content:
        print("[skip] predictor.py: RPR already patched")
    elif L21_OLD in content:
        content = content.replace(L21_OLD, L21_NEW, 1)
        print("[ok]   predictor.py: RPR l21 conditional boost applied")
    else:
        print("[ERROR] predictor.py: RPR anchor (L21_OLD) not found.")
        print("        The l21_loss block may have been modified externally.")
        sys.exit(1)

    # Atomic write only after both patches succeed
    with open(PREDICTOR, 'w') as f:
        f.write(content)


# Smoke test as a separate Python file (simpler than shell quoting)
SMOKE_TEST_CODE = """\
import sys
sys.path.insert(0, '.')
try:
    from maskrcnn_benchmark.config import cfg
    print('  USE_B1_ALIGN default:', cfg.MODEL.ROI_RELATION_HEAD.USE_B1_ALIGN)
    print('  B1_TAIL_ONLY default:', cfg.MODEL.ROI_RELATION_HEAD.B1_TAIL_ONLY)
    print('  B1_PR_BOOST  default:', cfg.MODEL.ROI_RELATION_HEAD.B1_PR_BOOST)
    print('  B1_ALIGN_WEIGHT def:', cfg.MODEL.ROI_RELATION_HEAD.B1_ALIGN_WEIGHT)
    print('  B1_WARMUP_ITER def:', cfg.MODEL.ROI_RELATION_HEAD.B1_WARMUP_ITER)
    print('[ok] cfg loads successfully')
except Exception as e:
    print(f'[ERROR] cfg load failed: {e}')
    sys.exit(1)

try:
    from maskrcnn_benchmark.modeling.roi_heads.relation_head.roi_relation_predictors import PrototypeEmbeddingNetwork
    print('[ok] predictor.py imports successfully')
except Exception as e:
    print(f'[ERROR] predictor import failed: {e}')
    sys.exit(1)
"""


def smoke_test():
    """Run smoke test by writing to temp file and executing."""
    print("\n[smoke test] Verifying cfg flags and predictor module...")
    tmp_path = os.path.join(PROJECT_DIR, "_smoke_test_stage2.py")
    with open(tmp_path, 'w') as f:
        f.write(SMOKE_TEST_CODE)
    try:
        result = subprocess.run(
            ["python3", tmp_path],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=60,
        )
        print(result.stdout, end='')
        if result.returncode != 0:
            print(f"[ERROR] smoke test failed:\n{result.stderr}")
            return False
        return True
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    print("=" * 60)
    print("  Stage 2 patch: TEPA (tail-only B1) + RPR (l21 boost)")
    print("=" * 60)
    patch_defaults()
    patch_predictor()
    ok = smoke_test()
    if ok:
        print("\n" + "=" * 60)
        print("[done] Stage 2 patch complete and verified.")
        print("=" * 60)
        print("\nNew cfg options (defaults shown):")
        print("  MODEL.ROI_RELATION_HEAD.B1_TAIL_ONLY  True   # tail-only alignment")
        print("  MODEL.ROI_RELATION_HEAD.B1_PR_BOOST   1.0    # PR boost (>1.0 to activate)")
        print("\nUsage in training command (override defaults via CLI):")
        print("  MODEL.ROI_RELATION_HEAD.USE_B1_ALIGN True \\")
        print("  MODEL.ROI_RELATION_HEAD.B1_TAIL_ONLY True \\")
        print("  MODEL.ROI_RELATION_HEAD.B1_PR_BOOST 1.0 \\")
        print("  MODEL.ROI_RELATION_HEAD.B1_ALIGN_WEIGHT 0.2 \\")
        print("  MODEL.ROI_RELATION_HEAD.B1_WARMUP_ITER 28000 \\")
    else:
        print("\n[done] Patches applied but smoke test failed.")
        print("       Inspect predictor.py and defaults.py manually.")
        sys.exit(1)
