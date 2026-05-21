#!/usr/bin/env python3
"""
Phase 0 代码修复脚本 - 在AutoDL上直接执行
用法: cd /root/autodl-tmp/penet-main && python fix_phase0.py
"""
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
REL_DIR = os.path.join(BASE, "maskrcnn_benchmark", "modeling", "roi_heads", "relation_head")

def fix_defaults():
    """在defaults.py中添加REWEIGHT_BETA和TRAIN_LA_TAU配置"""
    path = os.path.join(BASE, "maskrcnn_benchmark", "config", "defaults.py")
    with open(path, "r") as f:
        content = f.read()
    
    if "REWEIGHT_BETA" in content:
        print("[defaults.py] 已有REWEIGHT_BETA，跳过")
        return
    
    marker = "0.00647, 0.00084, 0.01077, 0.00132, 0.00069, 0.00376, 0.00214, 0.11424, 0.01205, 0.02958]"
    if marker not in content:
        print("[defaults.py] ERROR: 找不到REL_PROP末尾标记，请检查文件")
        sys.exit(1)
    
    replacement = marker + """

# Phase0 debiasing configs
_C.MODEL.ROI_RELATION_HEAD.REWEIGHT_BETA = 0.0    # 0=no reweight, 0.3/0.5/0.7=inverse freq reweight
_C.MODEL.ROI_RELATION_HEAD.TRAIN_LA_TAU = 0.0     # 0=no logit adjustment, 0.10/0.15=Train-LA strength"""
    
    content = content.replace(marker, replacement)
    with open(path, "w") as f:
        f.write(content)
    print("[defaults.py] ✅ 已添加 REWEIGHT_BETA 和 TRAIN_LA_TAU")


def fix_loss():
    """修复loss.py: 拆分criterion_loss为obj用和rel用，添加Reweight支持"""
    path = os.path.join(REL_DIR, "loss.py")
    with open(path, "r") as f:
        content = f.read()
    
    if "criterion_loss_rel" in content:
        print("[loss.py] 已有criterion_loss_rel，跳过")
        return
    
    # 1. 修改__init__签名：添加reweight_beta参数
    old_init = """        predicate_proportion,
    ):"""
    new_init = """        predicate_proportion,
        reweight_beta=0.0,
    ):"""
    if old_init in content:
        content = content.replace(old_init, new_init)
    
    # 2. 替换loss函数初始化（核心修改）
    old_loss_init = """        if self.use_label_smoothing:
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
        else:
            self.criterion_loss = nn.CrossEntropyLoss()"""
    
    new_loss_init = """        if self.use_label_smoothing:
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
            self.criterion_loss_rel = self.criterion_loss  # same for label smoothing
        else:
            # Object refinement loss is ALWAYS unweighted (151 object classes)
            self.criterion_loss = nn.CrossEntropyLoss()
            
            # Relation loss: optionally weighted (51 predicate classes)
            if reweight_beta > 0:
                raw_weight = torch.FloatTensor([0.5,] + predicate_proportion)
                inv_freq = 1.0 / (raw_weight + 1e-8)
                reweight = inv_freq ** reweight_beta
                reweight[0] = 1.0
                fg_mean = reweight[1:].mean()
                reweight[1:] = reweight[1:] / fg_mean
                reweight = torch.clamp(reweight, max=20.0)
                print(f"[Reweight] beta={reweight_beta}")
                print(f"[Reweight] bg_weight={reweight[0]:.3f}")
                print(f"[Reweight] fg_weight range: [{reweight[1:].min():.3f}, {reweight[1:].max():.3f}]")
                print(f"[Reweight] fg_weight mean: {reweight[1:].mean():.3f}")
                print(f"[Reweight] top-5 weights: {reweight.topk(5).values.tolist()}")
                self.criterion_loss_rel = nn.CrossEntropyLoss(weight=reweight.cuda())
            else:
                self.criterion_loss_rel = nn.CrossEntropyLoss()"""
    
    if old_loss_init not in content:
        print("[loss.py] ERROR: 找不到原始loss初始化代码块")
        sys.exit(1)
    content = content.replace(old_loss_init, new_loss_init)
    
    # 3. 修改loss计算：relation用criterion_loss_rel
    old_loss_call = "        loss_relation = self.criterion_loss(relation_logits, rel_labels.long())"
    new_loss_call = "        loss_relation = self.criterion_loss_rel(relation_logits, rel_labels.long())"
    content = content.replace(old_loss_call, new_loss_call)
    
    # 4. 修改make_roi_relation_loss_evaluator：传入reweight_beta
    old_factory = """    loss_evaluator = RelationLossComputation(
        cfg.MODEL.ATTRIBUTE_ON,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
        cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
    )"""
    new_factory = """    loss_evaluator = RelationLossComputation(
        cfg.MODEL.ATTRIBUTE_ON,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
        cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
        reweight_beta=cfg.MODEL.ROI_RELATION_HEAD.REWEIGHT_BETA,
    )"""
    content = content.replace(old_factory, new_factory)
    
    with open(path, "w") as f:
        f.write(content)
    print("[loss.py] ✅ 已拆分criterion_loss/criterion_loss_rel + 添加Reweight")


def fix_predictors():
    """修复roi_relation_predictors.py: 添加Train-LA和config打印"""
    path = os.path.join(REL_DIR, "roi_relation_predictors.py")
    with open(path, "r") as f:
        content = f.read()
    
    if "train_la_tau" in content:
        print("[predictors.py] 已有train_la_tau，跳过")
        return
    
    # 1. 在nms_thresh后添加Train-LA初始化和config打印
    old_nms = "        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES"
    new_nms = """        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        # Train-LA: Training-time Logit Adjustment
        self.train_la_tau = config.MODEL.ROI_RELATION_HEAD.TRAIN_LA_TAU
        if self.train_la_tau > 0:
            rel_prop = config.MODEL.ROI_RELATION_HEAD.REL_PROP
            log_prior = torch.log(torch.FloatTensor([0.5,] + rel_prop) + 1e-8)
            self.register_buffer('log_prior', log_prior)
            effective_shift = self.train_la_tau * log_prior
            print(f"[Train-LA] tau={self.train_la_tau}")
            print(f"[Train-LA] log_prior range: [{log_prior.min():.3f}, {log_prior.max():.3f}]")
            print(f"[Train-LA] Effective logit shift range: [{effective_shift.min():.3f}, {effective_shift.max():.3f}]")

        # Print full config summary
        rw_beta = config.MODEL.ROI_RELATION_HEAD.REWEIGHT_BETA
        la_tau = config.MODEL.ROI_RELATION_HEAD.TRAIN_LA_TAU
        use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        bspi = config.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE
        print("=" * 60)
        print(f"[Phase0 Config] REWEIGHT_BETA={rw_beta}, TRAIN_LA_TAU={la_tau}")
        print(f"[Phase0 Config] PREDICT_USE_BIAS={use_bias}, BATCH_SIZE_PER_IMAGE={bspi}")
        print(f"[Phase0 Config] mode={self.mode}")
        print("=" * 60)"""
    
    if old_nms not in content:
        print("[predictors.py] ERROR: 找不到nms_thresh行")
        sys.exit(1)
    content = content.replace(old_nms, new_nms, 1)  # 只替换第一个匹配
    
    # 2. 在rel_dists计算后、split前添加Train-LA logit调整
    old_split = """        ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  #  <r_norm, c_norm> / τ
        # the rel_dists will be used to calculate the Le_sim with the ce_loss

        entity_dists = entity_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)"""
    
    new_split = """        ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  #  <r_norm, c_norm> / τ
        # the rel_dists will be used to calculate the Le_sim with the ce_loss

        ### Train-LA: Training-time Logit Adjustment (only during training)
        if self.training and self.train_la_tau > 0:
            rel_dists = rel_dists + self.train_la_tau * self.log_prior.unsqueeze(0)

        entity_dists = entity_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)"""
    
    if old_split not in content:
        print("[predictors.py] ERROR: 找不到rel_dists split代码块")
        sys.exit(1)
    content = content.replace(old_split, new_split)
    
    with open(path, "w") as f:
        f.write(content)
    print("[predictors.py] ✅ 已添加Train-LA + config打印")


if __name__ == "__main__":
    print("=" * 50)
    print("Phase 0 代码修复脚本")
    print("=" * 50)
    
    # 验证目录
    if not os.path.exists(os.path.join(REL_DIR, "loss.py")):
        print(f"ERROR: 找不到 {REL_DIR}/loss.py")
        print("请确保在penet-main目录下执行: python fix_phase0.py")
        sys.exit(1)
    
    fix_defaults()
    fix_loss()
    fix_predictors()
    
    print("")
    print("=" * 50)
    print("全部修复完成！验证方法：")
    print("  grep 'REWEIGHT_BETA' maskrcnn_benchmark/config/defaults.py")
    print("  grep 'criterion_loss_rel' maskrcnn_benchmark/modeling/roi_heads/relation_head/loss.py")
    print("  grep 'train_la_tau' maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py")
    print("=" * 50)
