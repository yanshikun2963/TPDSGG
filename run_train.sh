#!/bin/bash
# ============================================================
# Day1: DRW beta=0.30 at 30K
# REWEIGHT_BETA=0.30 DRW_ITER=30000
# OUTPUT: ./output/day1_drw030_30k
# Clean PE-NET code + reweight only
# ============================================================

cd /root/autodl-tmp/penet-main

export PYTHONPATH=$(pwd):$PYTHONPATH
export LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])")/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=4

mkdir -p output

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python3 -u tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
    MODEL.ROI_RELATION_HEAD.REWEIGHT_BETA 0.30 \
    MODEL.ROI_RELATION_HEAD.DEFERRED_REWEIGHT_ITER 30000 \
    DTYPE "float32" \
    SOLVER.IMS_PER_BATCH 8 \
    TEST.IMS_PER_BATCH 1 \
    SOLVER.MAX_ITER 60000 \
    SOLVER.BASE_LR 1e-3 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 256 \
    SOLVER.STEPS "(28000, 48000)" \
    SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 10000 \
    GLOVE_DIR ./datasets/vg/ \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    OUTPUT_DIR ./output/day1_drw030_30k \
    SOLVER.PRE_VAL False \
    SOLVER.GRAD_NORM_CLIP 5.0 \
    INPUT.MIN_SIZE_TRAIN "(600,)" \
    INPUT.MAX_SIZE_TRAIN 800 \
    2>&1 | tee ./output/day1_drw030_30k.log
