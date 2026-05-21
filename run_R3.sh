#!/usr/bin/env bash
# ============================================================
# R3: Resume from baseline + tail-only TEPA + align_w=0.4 (激进)
# 用途: 补偿 tail-only 类数减半 -> 力度 2x 拉回到接近 full B1 效果
# 预期: mR 36-37, R 58-60
# 时长: ~2.5-3h (5090 单卡独享)
# ============================================================
set -e
cd /root/autodl-tmp/penet-main
rm -rf checkpoints/M_R3_tailonly_w04

python3 -u tools/relation_train_net.py \
    --config-file configs/e2e_relation_X_101_32_8_FPN_1x.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    MODEL.ROI_RELATION_HEAD.USE_B1_ALIGN True \
    MODEL.ROI_RELATION_HEAD.B1_TAIL_ONLY True \
    MODEL.ROI_RELATION_HEAD.B1_PR_BOOST 1.0 \
    MODEL.ROI_RELATION_HEAD.B1_ALIGN_WEIGHT 0.4 \
    MODEL.ROI_RELATION_HEAD.B1_WARMUP_ITER 5000 \
    MODEL.ROI_RELATION_HEAD.B1_MIN_COUNT 100 \
    MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
    MODEL.WEIGHT checkpoints/PENET_baseline/model_final.pth \
    GLOVE_DIR ./datasets/vg/ \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    SOLVER.IMS_PER_BATCH 8 \
    TEST.IMS_PER_BATCH 1 \
    SOLVER.BASE_LR 5e-5 \
    SOLVER.MAX_ITER 30000 \
    SOLVER.STEPS "(15000, 25000)" \
    SOLVER.VAL_PERIOD 2500 \
    SOLVER.CHECKPOINT_PERIOD 2500 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.GRAD_NORM_CLIP 5.0 \
    SOLVER.PRE_VAL False \
    DTYPE "float32" \
    OUTPUT_DIR ./checkpoints/M_R3_tailonly_w04
