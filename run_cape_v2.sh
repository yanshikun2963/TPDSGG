#!/bin/bash
# penet9: CAPE-Full: Gate+FiLM+LoRA pipeline (aggressive)
export PYTHONPATH=$(pwd):$PYTHONPATH
mkdir -p output
python -m torch.distributed.launch --master_port 10099 --nproc_per_node=1 \
  tools/relation_train_net.py \
  --config-file configs/cape_sgg_predcls.yaml \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR "CAPEv2PrototypeNetwork" \
  MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
  SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 2 \
  SOLVER.BASE_LR 0.001 SOLVER.MAX_ITER 60000 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  SOLVER.STEPS "(28000, 48000)" \
  SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 10000 \
  SOLVER.GRAD_NORM_CLIP 5.0 GLOVE_DIR ./datasets/vg/ \
  MODEL.RPN.POST_NMS_TOP_N_TRAIN 1000 \
  OUTPUT_DIR ./output/v2_cape_full \
  2>&1 | tee ./output/v2_cape_full_log.txt
