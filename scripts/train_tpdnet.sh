#!/usr/bin/env bash
# Train TPD-Net (TEPA + CPTR + class-balanced loss) on Visual Genome.
# Prerequisites:
#   1. A trained PE-Net baseline checkpoint at
#      checkpoints/PENET_baseline/model_final.pth
#   2. Confusion pair file at datasets/vg/cptr_pairs_filtered.npy, produced by
#      python datasets/prepare_freq_table.py
#      python datasets/prepare_confusion_pairs.py
set -e

MODEL_NAME="TPDNet_VG_PredCls"

if [ ! -f "datasets/vg/cptr_pairs_filtered.npy" ]; then
    echo "[ERROR] datasets/vg/cptr_pairs_filtered.npy not found."
    echo "        Please run datasets/prepare_freq_table.py and"
    echo "        datasets/prepare_confusion_pairs.py first."
    exit 1
fi

python3 -u tools/relation_train_net.py \
    --config-file configs/e2e_relation_X_101_32_8_FPN_1x.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    MODEL.ROI_RELATION_HEAD.USE_TEPA True \
    MODEL.ROI_RELATION_HEAD.TEPA_TAIL_ONLY True \
    MODEL.ROI_RELATION_HEAD.TEPA_WEIGHT 0.4 \
    MODEL.ROI_RELATION_HEAD.TEPA_WARMUP_ITER 5000 \
    MODEL.ROI_RELATION_HEAD.TEPA_MIN_COUNT 100 \
    MODEL.ROI_RELATION_HEAD.USE_CPTR True \
    MODEL.ROI_RELATION_HEAD.CPTR_PAIRS_PATH datasets/vg/cptr_pairs_filtered.npy \
    MODEL.ROI_RELATION_HEAD.CPTR_WEIGHT 0.10 \
    MODEL.ROI_RELATION_HEAD.CPTR_MARGIN 0.3 \
    MODEL.ROI_RELATION_HEAD.CPTR_K_SHARPNESS 10.0 \
    MODEL.ROI_RELATION_HEAD.CPTR_WARMUP_ITER 5000 \
    MODEL.ROI_RELATION_HEAD.USE_CB_LOSS True \
    MODEL.ROI_RELATION_HEAD.CB_LOSS_BETA 0.9995 \
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
    OUTPUT_DIR ./checkpoints/${MODEL_NAME}
