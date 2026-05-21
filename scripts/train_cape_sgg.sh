#!/bin/bash
# CAPE-SGG Training Script
# Usage:
#   bash scripts/train_cape_sgg.sh predcls 0      # PredCls on GPU 0
#   bash scripts/train_cape_sgg.sh sgcls 1         # SGCls on GPU 1
#   bash scripts/train_cape_sgg.sh sgdet 2         # SGDet on GPU 2

MODE=${1:-predcls}
GPU=${2:-0}

export PYTHONPATH=$(pwd):$PYTHONPATH

CONFIG_FILE="configs/cape_sgg_predcls.yaml"
GLOVE_DIR="./datasets/vg/"
CLIP_EMBED_PATH="./datasets/vg/clip_embeddings.pt"

# Precompute CLIP embeddings if not already done
if [ ! -f "$CLIP_EMBED_PATH" ]; then
    echo "=== Precomputing CLIP embeddings ==="
    python tools/clip_precompute.py --output_dir ./datasets/vg/
    if [ $? -ne 0 ]; then
        echo "CLIP precompute failed. Trying with OpenAI CLIP fallback..."
        python tools/clip_precompute.py --use_openai_clip --output_dir ./datasets/vg/
    fi
fi

echo "=== Training CAPE-SGG: Mode=$MODE, GPU=$GPU ==="

# LR schedule notes:
# PE-NET baseline: batch=8, BASE_LR=0.001 (effective=0.008), steps=(28000,48000), MAX_ITER=60000
# Linear scaling rule: when batch increases by k, LR increases by k, milestones decrease by k.
# - batch=12: BASE_LR=0.001 (eff=0.012), steps=(19000,32000), MAX_ITER=40000
# - batch=8:  BASE_LR=0.001 (eff=0.008),  steps=(28000,48000), MAX_ITER=60000

if [ "$MODE" == "predcls" ]; then
    CUDA_VISIBLE_DEVICES=$GPU torchrun \
        --master_port 10025 --nproc_per_node=1 \
        tools/relation_train_net.py \
        --config-file "$CONFIG_FILE" \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
        MODEL.ROI_RELATION_HEAD.PREDICTOR "CAPEPrototypeEmbeddingNetwork" \
        MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
        MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
        SOLVER.IMS_PER_BATCH 12 \
        TEST.IMS_PER_BATCH 2 \
        SOLVER.BASE_LR 0.001 \
        SOLVER.WARMUP_FACTOR 0.01 \
        SOLVER.WARMUP_ITERS 1000 \
        SOLVER.MAX_ITER 40000 \
        SOLVER.SCHEDULE.TYPE "WarmupMultiStepLR" \
        SOLVER.STEPS "(19000, 32000)" \
        SOLVER.VAL_PERIOD 2000 \
        SOLVER.CHECKPOINT_PERIOD 2000 \
        GLOVE_DIR "$GLOVE_DIR" \
        OUTPUT_DIR "./output/cape_sgg_predcls"

elif [ "$MODE" == "sgcls" ]; then
    CUDA_VISIBLE_DEVICES=$GPU torchrun \
        --master_port 10026 --nproc_per_node=1 \
        tools/relation_train_net.py \
        --config-file "$CONFIG_FILE" \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        MODEL.ROI_RELATION_HEAD.PREDICTOR "CAPEPrototypeEmbeddingNetwork" \
        MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
        MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
        SOLVER.IMS_PER_BATCH 12 \
        TEST.IMS_PER_BATCH 2 \
        SOLVER.BASE_LR 0.001 \
        SOLVER.WARMUP_FACTOR 0.01 \
        SOLVER.WARMUP_ITERS 1000 \
        SOLVER.MAX_ITER 40000 \
        SOLVER.SCHEDULE.TYPE "WarmupMultiStepLR" \
        SOLVER.STEPS "(19000, 32000)" \
        SOLVER.VAL_PERIOD 2000 \
        GLOVE_DIR "$GLOVE_DIR" \
        OUTPUT_DIR "./output/cape_sgg_sgcls"

elif [ "$MODE" == "sgdet" ]; then
    # SGDet uses batch=8 (same as PE-NET), so milestones are unchanged
    CUDA_VISIBLE_DEVICES=$GPU torchrun \
        --master_port 10027 --nproc_per_node=1 \
        tools/relation_train_net.py \
        --config-file "$CONFIG_FILE" \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        MODEL.ROI_RELATION_HEAD.PREDICTOR "CAPEPrototypeEmbeddingNetwork" \
        MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
        MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
        SOLVER.IMS_PER_BATCH 8 \
        TEST.IMS_PER_BATCH 2 \
        SOLVER.BASE_LR 0.001 \
        SOLVER.WARMUP_FACTOR 0.01 \
        SOLVER.WARMUP_ITERS 1000 \
        SOLVER.MAX_ITER 60000 \
        SOLVER.SCHEDULE.TYPE "WarmupMultiStepLR" \
        SOLVER.STEPS "(28000, 48000)" \
        SOLVER.VAL_PERIOD 2000 \
        GLOVE_DIR "$GLOVE_DIR" \
        OUTPUT_DIR "./output/cape_sgg_sgdet"

else
    echo "Unknown mode: $MODE. Use predcls, sgcls, or sgdet."
    exit 1
fi

echo "=== Training complete for $MODE ==="
