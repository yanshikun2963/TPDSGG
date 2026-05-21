#!/bin/bash
# ============================================
# Phase 0 - Reweight beta=0.5 + Train-LA tau=0.15
# Stronger dual debiasing than penet6 (which uses tau=0.10)
# Tests whether stronger LA is compatible with Reweight
# ============================================
bash setup_data.sh
export CUDA_VISIBLE_DEVICES=0
export NUM_GUP=1

MODEL_NAME="phase0_reweight_b05_trainla_t015"
mkdir -p ./output/${MODEL_NAME}/

echo "========================================"
echo "Starting: ${MODEL_NAME}"
echo "Reweight beta=0.5, Train-LA tau=0.15"
echo "========================================"

python3 \
  tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
  MODEL.ROI_RELATION_HEAD.REWEIGHT_BETA 0.5 \
  MODEL.ROI_RELATION_HEAD.TRAIN_LA_TAU 0.15 \
  DTYPE "float32" \
  SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH $NUM_GUP \
  SOLVER.MAX_ITER 60000 SOLVER.BASE_LR 1e-3 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 256 \
  SOLVER.STEPS "(28000, 48000)" SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 10000 GLOVE_DIR ./datasets/vg/ \
  MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR ./output/${MODEL_NAME} \
  SOLVER.PRE_VAL False \
  SOLVER.GRAD_NORM_CLIP 5.0 \
  2>&1 | tee ./output/${MODEL_NAME}/train_log.txt

echo "Training complete: ${MODEL_NAME}"
