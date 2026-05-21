#!/usr/bin/env bash
# ============================================================
# Machine 2: B1 (EMA Prototype Alignment) + CB Loss
# ============================================================
# 单阶段端到端训练，从 baseline ckpt resume
# 时长: ~14h (30k iter)
# ============================================================

set -e

PROJECT_DIR="/root/autodl-tmp/penet-main"
CONFIG_FILE="configs/e2e_relation_X_101_32_8_FPN_1x.yaml"
BASELINE_CKPT="/root/autodl-tmp/penet-main/checkpoints/PENET_baseline/model_final.pth"
GLOVE_DIR="${PROJECT_DIR}/datasets/vg/"
PRETRAINED_DET_CKPT="${PROJECT_DIR}/checkpoints/pretrained_faster_rcnn/model_final.pth"
OUTPUT_DIR="${PROJECT_DIR}/checkpoints/M2_B1_CB"

export CUDA_VISIBLE_DEVICES=0
export NUM_GPU=1

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

echo "==============================================="
echo "[Machine 2] B1 EMA Prototype Alignment + CB Loss"
echo "==============================================="

python3 -u tools/relation_train_net.py \
    --config-file "${CONFIG_FILE}" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    MODEL.ROI_RELATION_HEAD.USE_CB_LOSS False \
    MODEL.ROI_RELATION_HEAD.CB_LOSS_BETA 0.9999 \
    MODEL.ROI_RELATION_HEAD.USE_B1_ALIGN True \
    MODEL.ROI_RELATION_HEAD.B1_ALIGN_WEIGHT 0.2 \
    MODEL.ROI_RELATION_HEAD.B1_WARMUP_ITER 5000 \
    MODEL.ROI_RELATION_HEAD.B1_MIN_COUNT 100 \
    MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
    MODEL.WEIGHT "${BASELINE_CKPT}" \
    GLOVE_DIR "${GLOVE_DIR}" \
    MODEL.PRETRAINED_DETECTOR_CKPT "${PRETRAINED_DET_CKPT}" \
    SOLVER.IMS_PER_BATCH 8 \
    TEST.IMS_PER_BATCH ${NUM_GPU} \
    SOLVER.BASE_LR 5e-4 \
    SOLVER.MAX_ITER 60000 \
    SOLVER.STEPS "(30000, 50000)" \
    SOLVER.VAL_PERIOD 2500 \
    SOLVER.CHECKPOINT_PERIOD 2500 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.GRAD_NORM_CLIP 5.0 \
    SOLVER.PRE_VAL False \
    DTYPE "float32" \
    OUTPUT_DIR "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "[Done] Best ckpt: ${OUTPUT_DIR}/model_best.pth"
