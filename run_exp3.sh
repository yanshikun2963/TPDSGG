#!/usr/bin/env bash
# ============================================================
# Exp 3: 晚激活 (warmup=48000) + 加强力度 (align_w=0.3) + tail-only
# 用途: B1 在 lr 第二次衰减时激活, 模型几乎完全收敛, 最安全
# 预期: mR ~33-35, R ~62-63, l21_loss 几乎不动
# ============================================================
set -e
cd /root/autodl-tmp/penet-main
rm -rf checkpoints/M_exp3_late_strong

python3 -u tools/relation_train_net.py \
    --config-file configs/e2e_relation_X_101_32_8_FPN_1x.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    MODEL.ROI_RELATION_HEAD.USE_B1_ALIGN True \
    MODEL.ROI_RELATION_HEAD.B1_TAIL_ONLY True \
    MODEL.ROI_RELATION_HEAD.B1_PR_BOOST 1.0 \
    MODEL.ROI_RELATION_HEAD.B1_ALIGN_WEIGHT 0.3 \
    MODEL.ROI_RELATION_HEAD.B1_WARMUP_ITER 48000 \
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
    OUTPUT_DIR ./checkpoints/M_exp3_late_strong
