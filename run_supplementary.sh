#!/bin/bash
# ==========================================================
# FDI-Net Supplementary Experiments - 5 Machines (FINAL)
# ==========================================================
# A group: Baseline training for kappa analysis (3 machines)
#   A1: MOTIFS baseline    (clean loss.py, clean predictors.py)
#   A2: VCTree baseline    (clean loss.py, clean predictors.py)
#   A3: PE-NET baseline    (clean loss.py, clean predictors.py)
# C group: CB-Loss reweighting on other backbones (2 machines)
#   C1: MOTIFS + CB-Loss   (loss_correct.py, clean predictors.py)
#   C2: VCTree + CB-Loss   (loss_correct.py, clean predictors.py)
# ==========================================================

PENET_DIR="/root/autodl-tmp/penet-main"
LOSS_DIR="$PENET_DIR/maskrcnn_benchmark/modeling/roi_heads/relation_head"
PRED_FILE="$LOSS_DIR/roi_relation_predictors.py"
PRED_BAK="$PRED_FILE.bak_fdinet"
LOSS_FILE="$LOSS_DIR/loss.py"

# ----------------------------------------------------------
# UNIVERSAL FIX (run on EVERY machine first)
# ----------------------------------------------------------
fix_all() {
    cd "$PENET_DIR"
    echo "========== Applying fixes =========="

    python3 << 'PYEOF'
import os, re

root = "."
count = 0
for dirpath, dirs, files in os.walk(root):
    dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git', 'checkpoints', 'Datasets')]
    for fn in files:
        if not fn.endswith('.py'):
            continue
        fp = os.path.join(dirpath, fn)
        try:
            c = open(fp).read()
        except:
            continue
        changed = False

        # Fix apex imports
        if 'from apex import amp' in c:
            c = re.sub(
                r'(try:\s*\n\s*)*from apex import amp(\s*\n\s*except[^\n]*\n\s*(pass|raise[^\n]*))*',
                'try:\n    from apex import amp\nexcept ImportError:\n    pass',
                c
            )
            changed = True

        # Fix amp.float_function / amp.half_function usage (not decorator)
        if 'amp.float_function(' in c:
            c = re.sub(r'amp\.float_function\(([^)]+)\)', r'\1', c)
            changed = True
        if 'amp.half_function(' in c:
            c = re.sub(r'amp\.half_function\(([^)]+)\)', r'\1', c)
            changed = True

        # Fix amp decorators
        if '@amp.float_function' in c:
            c = re.sub(r'\s*@amp\.float_function\n', '\n', c)
            changed = True
        if '@amp.half_function' in c:
            c = re.sub(r'\s*@amp\.half_function\n', '\n', c)
            changed = True

        # Fix torch._six
        if 'from torch._six import' in c:
            c = c.replace('from torch._six import string_classes', 'string_classes = (str,)')
            changed = True
        if 'torch._six.PY37' in c:
            c = c.replace('if torch._six.PY37:', 'if True:')
            changed = True
        if 'torch._six.PY3' in c and 'PY37' not in c:
            c = c.replace('torch._six.PY3', 'True')
            changed = True

        # Fix _download_url_to_file
        if '_download_url_to_file' in c and 'download_url_to_file' in c:
            c = c.replace('_download_url_to_file', 'download_url_to_file')
            changed = True

        # Fix scipy.misc.imread
        if 'from scipy.misc import imread' in c:
            c = c.replace('from scipy.misc import imread', 'from imageio import imread')
            changed = True

        if changed:
            open(fp, 'w').write(c)
            count += 1
            print(f"  Fixed: {fp}")

print(f"  Total files fixed: {count}")
PYEOF

    echo "========== Fixes complete =========="
}

# ----------------------------------------------------------
# Common environment setup
# ----------------------------------------------------------
setup_env() {
    cd "$PENET_DIR"
    export CUDA_VISIBLE_DEVICES=0
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
    export OMP_NUM_THREADS=4
    export PYTHONPATH="$PENET_DIR:$PYTHONPATH"
}

# ----------------------------------------------------------
# Ensure predictors.py is clean (no FDI-Net or any other patch)
# ----------------------------------------------------------
ensure_clean_predictors() {
    if [ -f "$PRED_BAK" ]; then
        cp "$PRED_BAK" "$PRED_FILE"
        echo "  Restored clean predictors.py from .bak_fdinet"
    fi

    if grep -qE "FDINet|dac_branch|pgcr_projector|predicate_bias|_fdi_|fdi_net" "$PRED_FILE"; then
        echo "  ERROR: predictors.py still contains patch code!"
        grep -nE "FDINet|dac_branch|pgcr_projector|predicate_bias|_fdi_|fdi_net" "$PRED_FILE" | head -5
        echo "  Please manually restore the original PE-NET predictors.py"
        return 1
    fi
    echo "  predictors.py: CLEAN"
    return 0
}

# ----------------------------------------------------------
# Set loss.py to clean version (no CB-Loss)
# ----------------------------------------------------------
set_clean_loss() {
    if [ -f "$PENET_DIR/loss_clean.py" ]; then
        cp "$PENET_DIR/loss_clean.py" "$LOSS_FILE"
    else
        echo "  ERROR: loss_clean.py not found in $PENET_DIR"
        return 1
    fi

    if grep -q "cb_loss_beta=0.9999" "$LOSS_FILE"; then
        echo "  ERROR: loss.py STILL has CB-Loss!"
        return 1
    fi
    echo "  loss.py: CLEAN (no CB-Loss)"
    return 0
}

# ----------------------------------------------------------
# Set loss.py to CB-Loss version
# ----------------------------------------------------------
set_cbloss() {
    if [ -f "$PENET_DIR/loss_correct.py" ]; then
        cp "$PENET_DIR/loss_correct.py" "$LOSS_FILE"
    else
        echo "  ERROR: loss_correct.py not found in $PENET_DIR"
        return 1
    fi

    if ! grep -q "cb_loss_beta=0.9999" "$LOSS_FILE"; then
        echo "  ERROR: loss.py does NOT have CB-Loss!"
        return 1
    fi
    echo "  loss.py: CB-Loss PRESENT"
    return 0
}

# ----------------------------------------------------------
# Generic training function
# ----------------------------------------------------------
run_training() {
    local predictor="$1"
    local output_dir="$2"

    if [ -d "$output_dir" ]; then
        echo "  Clearing old checkpoint directory: $output_dir"
        rm -rf "$output_dir"
    fi
    mkdir -p "$output_dir"

    echo ""
    echo "  Starting training: PREDICTOR=$predictor"
    echo "  OUTPUT_DIR=$output_dir"
    echo "  Estimated time: ~8 hours"
    echo ""

    stdbuf -oL python3 -u tools/relation_train_net.py \
      --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
      MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
      MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
      MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
      MODEL.ROI_RELATION_HEAD.PREDICTOR "$predictor" \
      DTYPE "float32" \
      SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 1 \
      SOLVER.MAX_ITER 60000 SOLVER.BASE_LR 1e-3 \
      SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
      MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
      SOLVER.STEPS "(28000, 48000)" \
      SOLVER.VAL_PERIOD 3000 SOLVER.CHECKPOINT_PERIOD 3000 \
      GLOVE_DIR "$PENET_DIR/Datasets/VG" \
      MODEL.PRETRAINED_DETECTOR_CKPT "$PENET_DIR/Datasets/VG/model_final.pth" \
      OUTPUT_DIR "$output_dir" \
      SOLVER.PRE_VAL False SOLVER.GRAD_NORM_CLIP 5.0 \
      INPUT.MIN_SIZE_TRAIN "(600,)" INPUT.MAX_SIZE_TRAIN 1000 \
      INPUT.MIN_SIZE_TEST 600 INPUT.MAX_SIZE_TEST 1000 \
      DATALOADER.NUM_WORKERS 4 \
      2>&1 | tee -a "$output_dir/train_$(date +%Y%m%d_%H%M%S).log"
}

# ===========================================================
# A GROUP: Baseline (clean loss, clean predictors)
# ===========================================================

train_A1() {
    setup_env
    echo ""; echo "========================================"
    echo "  Machine A1: MOTIFS baseline"
    echo "========================================"
    ensure_clean_predictors || return 1
    set_clean_loss || return 1
    run_training "MotifPredictor" "./checkpoints/MOTIFS_baseline"
}

train_A2() {
    setup_env
    echo ""; echo "========================================"
    echo "  Machine A2: VCTree baseline"
    echo "========================================"
    ensure_clean_predictors || return 1
    set_clean_loss || return 1
    run_training "VCTreePredictor" "./checkpoints/VCTree_baseline"
}

train_A3() {
    setup_env
    echo ""; echo "========================================"
    echo "  Machine A3: PE-NET baseline"
    echo "========================================"
    ensure_clean_predictors || return 1
    set_clean_loss || return 1
    run_training "PrototypeEmbeddingNetwork" "./checkpoints/PENET_baseline"
}

# ===========================================================
# C GROUP: CB-Loss reweighting (loss_correct, clean predictors)
# ===========================================================

train_C1() {
    setup_env
    echo ""; echo "========================================"
    echo "  Machine C1: MOTIFS + CB-Loss"
    echo "========================================"
    ensure_clean_predictors || return 1
    set_cbloss || return 1
    run_training "MotifPredictor" "./checkpoints/MOTIFS_CBLoss"
}

train_C2() {
    setup_env
    echo ""; echo "========================================"
    echo "  Machine C2: VCTree + CB-Loss"
    echo "========================================"
    ensure_clean_predictors || return 1
    set_cbloss || return 1
    run_training "VCTreePredictor" "./checkpoints/VCTree_CBLoss"
}

# ===========================================================
echo "==========================================="
echo "  Supplementary Experiments Script (FINAL)"
echo "==========================================="
echo ""
echo "  Machine A1: fix_all && train_A1    # MOTIFS baseline"
echo "  Machine A2: fix_all && train_A2    # VCTree baseline"
echo "  Machine A3: fix_all && train_A3    # PE-NET baseline"
echo "  Machine C1: fix_all && train_C1    # MOTIFS + CB-Loss"
echo "  Machine C2: fix_all && train_C2    # VCTree + CB-Loss"
echo ""
echo "  Required files in $PENET_DIR:"
echo "    loss_clean.py      (original PE-NET loss.py)"
echo "    loss_correct.py    (loss.py with cb_loss_beta=0.9999)"
echo "==========================================="
