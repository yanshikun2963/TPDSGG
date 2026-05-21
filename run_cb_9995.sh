#!/usr/bin/env bash
# ============================================================
# Machine B: TEPA (R3) + CPTR v2  --  主推: 过滤pairs + weight 0.10
# 二维 sweep 的一个臂. 共用: head-detach + warmup=5000.
# 变量: CPTR_WEIGHT=0.10, pairs=cptr_pairs_filtered.npy
# 主指标 mR@50, 约束 R@50 > 60. 时长 ~3.5h, 5090 单卡.
#
# 前置 (按顺序, 每台 CPTR 机器都要做):
#   python3 build_freq_table.py        # -> datasets/vg/soo_freq.npy
#   python3 build_confusion_pairs.py   # -> cptr_pairs.npy + cptr_pairs_filtered.npy
#   python3 patch_cptr_v2.py           # 应用 CPTR v2 patch (自动回退旧 patch)
# ============================================================
set -e
cd /root/autodl-tmp/penet-main
rm -rf checkpoints/M_cb_9995

if [ ! -f "datasets/vg/cptr_pairs_filtered.npy" ]; then
    echo "[ERROR] datasets/vg/cptr_pairs_filtered.npy missing."
    echo "        Run: python3 build_freq_table.py && python3 build_confusion_pairs.py"
    exit 1
fi

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
    MODEL.ROI_RELATION_HEAD.USE_CPTR True \
    MODEL.ROI_RELATION_HEAD.USE_CB_LOSS True \
    MODEL.ROI_RELATION_HEAD.CB_LOSS_BETA 0.9995 \
    MODEL.ROI_RELATION_HEAD.CPTR_PAIRS_PATH datasets/vg/cptr_pairs_filtered.npy \
    MODEL.ROI_RELATION_HEAD.CPTR_WEIGHT 0.10 \
    MODEL.ROI_RELATION_HEAD.CPTR_MARGIN 0.3 \
    MODEL.ROI_RELATION_HEAD.CPTR_K_SHARPNESS 10.0 \
    MODEL.ROI_RELATION_HEAD.CPTR_WARMUP_ITER 5000 \
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
    OUTPUT_DIR ./checkpoints/M_cb_9995
