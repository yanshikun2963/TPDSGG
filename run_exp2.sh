#!/usr/bin/env bash
# ============================================================
# Exp 2: 早激活 (warmup=28000) + 中等力度 (align_w=0.2) + tail-only
# 用途: B1 在 lr 第一次衰减时激活, 有 32k iter 充分发挥作用
# 预期: mR ~35-36, R ~60-62, l21_loss 轻微上升
# ============================================================
set -e
cd /root/autodl-tmp/penet-main
rm -rf checkpoints/M_exp2_early_moderate

python3 -u tools/relation_train_net.py \
    --config-file configs/e2e_relation_X_101_32_8_FPN_1x.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    MODEL.ROI_RELATION_HEAD.USE_B1_ALIGN True \
    MODEL.ROI_RELATION_HEAD.B1_TAIL_ONLY True \
    MODEL.ROI_RELATION_HEAD.B1_PR_BOOST 1.0 \
    MODEL.ROI_RELATION_HEAD.B1_ALIGN_WEIGHT 0.2 \
    MODEL.ROI_RELATION_HEAD.B1_WARMUP_ITER 28000 \
    MODEL.ROI_RELATION_HEAD.B1_MIN_COUNT 100 \
    MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
    MODEL.WEIGHT "" \
    GLOVE_DIR ./datasets/vg/ \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    SOLVER.IMS_PER_BATCH 8 \
    TEST.IMS_PER_BATCH 1 \
    SOLVER.BASE_LR 0.001 \
    SOLVER.MAX_ITER 60000 \
    SOLVER.STEPS "(28000, 48000)" \
    SOLVER.VAL_PERIOD 3000 \
    SOLVER.CHECKPOINT_PERIOD 3000 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.GRAD_NORM_CLIP 5.0 \
    SOLVER.PRE_VAL False \
    DTYPE "float32" \
    OUTPUT_DIR ./checkpoints/M_exp2_early_moderate
