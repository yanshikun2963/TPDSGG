#!/usr/bin/env bash
# ============================================================
# Exp 1: 复现机器3失败 - 早激活 (warmup=10000) + full B1
# 用途: 验证我们的失败假设 - B1 在高 lr 阶段激活会崩塌
# 预期: mR 在 iter 12000-18000 之间崩塌
# ============================================================
set -e
cd /root/autodl-tmp/penet-main
rm -rf checkpoints/M_exp1_reproduce_fail

python3 -u tools/relation_train_net.py \
    --config-file configs/e2e_relation_X_101_32_8_FPN_1x.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    MODEL.ROI_RELATION_HEAD.USE_B1_ALIGN True \
    MODEL.ROI_RELATION_HEAD.B1_TAIL_ONLY False \
    MODEL.ROI_RELATION_HEAD.B1_PR_BOOST 1.0 \
    MODEL.ROI_RELATION_HEAD.B1_ALIGN_WEIGHT 0.2 \
    MODEL.ROI_RELATION_HEAD.B1_WARMUP_ITER 10000 \
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
    OUTPUT_DIR ./checkpoints/M_exp1_reproduce_fail
