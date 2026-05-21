#!/usr/bin/env bash
# Evaluate a trained TPD-Net checkpoint on Visual Genome PredCls.
set -e

MODEL_NAME="TPDNet_VG_PredCls"

python3 -u tools/relation_test_net.py \
    --config-file configs/e2e_relation_X_101_32_8_FPN_1x.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
    TEST.IMS_PER_BATCH 1 \
    DTYPE "float32" \
    GLOVE_DIR ./datasets/vg/ \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.WEIGHT ./checkpoints/${MODEL_NAME}/model_final.pth \
    OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
    TEST.ALLOW_LOAD_FROM_CACHE False
