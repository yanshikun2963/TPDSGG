#!/bin/bash
# CAPE-SGG Ablation Experiment Launcher
# Usage: bash scripts/run_ablation.sh <ablation_id> <gpu_id>
# Example: bash scripts/run_ablation.sh A3 0
#          bash scripts/run_ablation.sh A7 1

ABLATION=${1:?Usage: run_ablation.sh <A1|A2|A3|A4|A7|A8|fasa005|fasa02|fasa05|heads2|heads8> <gpu>}
GPU=${2:-0}

export PYTHONPATH=$(pwd):$PYTHONPATH
mkdir -p logs

CONFIG="configs/cape_sgg_predcls.yaml"
COMMON="--config-file $CONFIG \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR CAPEPrototypeEmbeddingNetwork \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    SOLVER.IMS_PER_BATCH 12 \
    SOLVER.MAX_ITER 50000 \
    SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    GLOVE_DIR ./datasets/vg/"

echo "=== Starting ablation: $ABLATION on GPU $GPU ==="

case $ABLATION in
    A1)
        DESC="PE-NET + APT only (no CLIP Bridge, no CAPM, no FASA)"
        EXTRA="MODEL.ROI_RELATION_HEAD.CAPE.USE_CLIP_BRIDGE False \
               MODEL.ROI_RELATION_HEAD.CAPE.USE_CAPM False \
               MODEL.ROI_RELATION_HEAD.CAPE.USE_FASA False"
        ;;
    A2)
        DESC="+ CLIP Bridge (no CAPM, no FASA)"
        EXTRA="MODEL.ROI_RELATION_HEAD.CAPE.USE_CAPM False \
               MODEL.ROI_RELATION_HEAD.CAPE.USE_FASA False"
        ;;
    A3)
        DESC="+ CLIP Bridge + CAPM (no FASA)"
        EXTRA="MODEL.ROI_RELATION_HEAD.CAPE.USE_FASA False"
        ;;
    A4)
        DESC="+ CLIP Bridge + FASA (no CAPM)"
        EXTRA="MODEL.ROI_RELATION_HEAD.CAPE.USE_CAPM False"
        ;;
    A5)
        DESC="Full CAPE-SGG (same as main experiment)"
        EXTRA=""
        ;;
    A7)
        DESC="Full but CAPM→ContextBias (logit-space bias)"
        EXTRA="MODEL.ROI_RELATION_HEAD.CAPE.CAPM_VARIANT context_bias"
        ;;
    A8)
        DESC="Full but CAPM→LearnableGate (no CLIP semantics in gate)"
        EXTRA="MODEL.ROI_RELATION_HEAD.CAPE.CAPM_VARIANT learnable_gate"
        ;;
    fasa005)
        DESC="FASA weight = 0.05"
        EXTRA="MODEL.ROI_RELATION_HEAD.CAPE.FASA_WEIGHT 0.05"
        ;;
    fasa02)
        DESC="FASA weight = 0.2"
        EXTRA="MODEL.ROI_RELATION_HEAD.CAPE.FASA_WEIGHT 0.2"
        ;;
    fasa05)
        DESC="FASA weight = 0.5"
        EXTRA="MODEL.ROI_RELATION_HEAD.CAPE.FASA_WEIGHT 0.5"
        ;;
    heads2)
        DESC="CAPM heads = 2"
        EXTRA="MODEL.ROI_RELATION_HEAD.CAPE.CAPM_NUM_HEADS 2"
        ;;
    heads8)
        DESC="CAPM heads = 8"
        EXTRA="MODEL.ROI_RELATION_HEAD.CAPE.CAPM_NUM_HEADS 8"
        ;;
    *)
        echo "Unknown ablation: $ABLATION"
        echo "Available: A1 A2 A3 A4 A5 A7 A8 fasa005 fasa02 fasa05 heads2 heads8"
        exit 1
        ;;
esac

OUTDIR="./output/ablation_${ABLATION}"
LOGFILE="logs/ablation_${ABLATION}.log"

echo "Description: $DESC"
echo "Output: $OUTDIR"
echo "Log: $LOGFILE"
echo ""

CUDA_VISIBLE_DEVICES=$GPU nohup python tools/relation_train_net.py \
    $COMMON \
    $EXTRA \
    OUTPUT_DIR "$OUTDIR" \
    > "$LOGFILE" 2>&1 &

PID=$!
echo "Started PID: $PID"
echo "Monitor: tail -f $LOGFILE"
echo "Check results: grep 'SGG eval' $LOGFILE | tail -10"
