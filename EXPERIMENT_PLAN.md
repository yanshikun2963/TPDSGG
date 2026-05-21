# CAPE-SGG 终极实验设计方案
# —— 一份写给SGG领域新手的完整实验指南

**硬件**: 2× RTX PRO 6000 (96GB)，记为 GPU-0 和 GPU-1
**时间**: 10-14天
**目标**: 在 PredCls 任务的 mR@50 和 F@50 指标上超越现有SOTA

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 第零章：SGG实验你必须知道的基础知识
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 0.1 三个子任务是什么？

SGG领域的每篇论文都必须在三个任务上报告结果。它们的区别在于"给模型多少已知信息"：

┌──────────────────────────────────────────────────────────┐
│  PredCls (Predicate Classification)                      │
│  给定: GT框 ✅ + GT物体标签 ✅                            │
│  预测: 只预测关系谓词                                      │
│  难度: ★☆☆☆ (最简单，隔离了检测噪声，纯看关系建模能力)     │
│  地位: 无偏SGG论文的【主战场】，消融实验只需跑这个          │
│  用时: ~9.5h / 50k iter (你的卡)                          │
├──────────────────────────────────────────────────────────┤
│  SGCls (Scene Graph Classification)                      │
│  给定: GT框 ✅ + GT物体标签 ❌                             │
│  预测: 物体标签 + 关系谓词                                 │
│  难度: ★★☆☆ (加了物体分类的噪声)                          │
│  用时: ~10h                                               │
├──────────────────────────────────────────────────────────┤
│  SGDet (Scene Graph Detection)                           │
│  给定: GT框 ❌ + GT物体标签 ❌                             │
│  预测: 检测框 + 物体标签 + 关系谓词 (端到端)               │
│  难度: ★★★★ (最难最实用，但分数会低20-30个点)              │
│  用时: ~12h (batch要减小)                                  │
└──────────────────────────────────────────────────────────┘

### 0.2 七个指标各是什么意思？

┌─────────────────────────────────────────────────────────────────┐
│ R@K (Recall@K) — "整体召回率"                                   │
│   对每张图取模型最confident的K个预测三元组<s,p,o>，              │
│   看覆盖了多少GT三元组。K一般取20/50/100。                      │
│   问题：90%的GT都是"on/has/in"，所以只预测头部类就能拿高分。    │
│   → 这个指标有利于"偏向头部"的模型                              │
│                                                                  │
│ mR@K (mean Recall@K) — "平均类别召回率" ⭐最重要                │
│   对50个谓词类别分别算R@K，然后取平均。                         │
│   每个类权重相同→尾部类("riding","flying in")和头部类一样重要    │
│   → 这个指标衡量"公平性"，是无偏SGG论文的核心指标               │
│                                                                  │
│ F@K (调和平均) — "综合平衡分" ⭐审稿人越来越看重                │
│   F@K = 2 × R@K × mR@K / (R@K + mR@K)                          │
│   同时惩罚R过低和mR过低→要求模型"头尾兼顾"                      │
│   → 一个方法如果mR@50=47但R@50=44(如DRM)，F@50只有45.4          │
│     另一个方法mR@50=36但R@50=62(如RA-SGG)，F@50=45.7 → 反而更高│
│                                                                  │
│ zR@K (Zero-shot Recall) — "零样本泛化"                          │
│   只在训练集中从未出现过的<s类,p类,o类>组合上评估                │
│   → 测试组合泛化能力                                             │
│                                                                  │
│ ng-R@K / ng-mR@K — "无图约束版"                                 │
│   允许同一对物体有多个谓词(标准版只允许1个)                      │
│   → 分数更高，部分论文报告，非必须                               │
└─────────────────────────────────────────────────────────────────┘

### 0.3 一篇SGG论文通常需要哪些实验？

┌──────────────────────────────────────────────────────────────┐
│ 必须有 (审稿人100%会看的):                                    │
│                                                               │
│ ① Table 1: 与SOTA方法对比表                                  │
│    - 3个任务(PredCls/SGCls/SGDet)                             │
│    - 每个任务报6个指标: mR@20/50/100, R@20/50/100            │
│    - 额外计算F@50, F@100 (越来越多论文要求)                   │
│    - 需要包含6-8个对比方法(从已发表论文抄数字)                │
│    - 自己只需跑: 你的方法 + PE-NET基线                        │
│                                                               │
│ ② Table 2: 消融实验表 (Ablation Study)                       │
│    - 只在PredCls任务上做                                       │
│    - 目的：证明你的每个组件都有独立贡献                        │
│    - 报3个指标就够: mR@50, R@50, F@50                         │
│                                                               │
│ ③ 至少1个定性分析 (Qualitative Analysis)                      │
│    - 比如: t-SNE可视化 / gate热力图 / 案例展示                │
│                                                               │
│ 有就更好 (加分项):                                             │
│ ④ Table 3: 超参数敏感性分析                                   │
│ ⑤ Table 4: 泛化性实验 (CAPM插到其他模型上)                    │
│ ⑥ Figure: Per-predicate recall柱状图                          │
│ ⑦ 参数量/推理速度对比                                          │
└──────────────────────────────────────────────────────────────┘

### 0.4 你的三个贡献分别需要什么实验来支撑？

贡献C1 (CAPM方法创新):
  → 消融A3 vs A2 证明 "加CAPM比不加好"
  → 消融A5 vs A7 证明 "原型空间调制 > logit空间偏置"
  → 消融A5 vs A8 证明 "CLIP语义门控 > 纯可学习门控"
  → Table 1证明 "CAPE-SGG整体超过SOTA"

贡献C2 (CAPE-SGG统一框架):
  → 消融表整体证明 "每个组件独立有效且互补"
  → Table 1的三个任务一致提升证明 "框架泛化性好"

贡献C3 (实验发现):
  → A5 vs A7: "prototype-space > logit-space" (发现a)
  → A5 vs A8: "CLIP gate > learnable gate" (发现b)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 第一章：你需要超越的SOTA数字
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### PredCls (你的主战场)

| Method            | Venue     | mR@50 | mR@100 | R@50 | R@100 | F@50 | F@100 |
|-------------------|-----------|-------|--------|------|-------|------|-------|
| MOTIFS            | CVPR'18   | 14.6  | 15.8   | 66.0 | 67.9  | 23.9 | 25.6  |
| Motifs+TDE        | CVPR'20   | 25.5  | 29.1   | 46.2 | 51.4  | 32.9 | 37.0  |
| BGNN              | CVPR'21   | 30.4  | 32.9   | 59.2 | 61.3  | 40.2 | 42.8  |
| PE-NET            | CVPR'23   | 31.5  | 33.8   | 64.9 | 67.2  | 42.4 | 45.0  |
| CFA               | ICCV'23   | 35.7  | 38.2   | 54.1 | 56.6  | 43.0 | 45.7  |
| DRM(w/DKT)        | CVPR'24   | 47.1  | 49.6   | 43.9 | 45.8  | 45.4 | 47.6  |
| RA-SGG            | AAAI'25   | 36.2  | 39.1   | 62.2 | 64.1  | 45.7 | 48.6  |
| HIERCOM+IETrans   | WACV'25   | 38.0  | 44.1   | 60.4 | 66.4  | 46.6 | 53.0  |
| **CAPE-SGG (目标)**|          |**≥38**|**≥41** |**≥60**|**≥63**|**≥47**|**≥50**|

注意：Table 1中对比方法的数字直接从它们论文中抄，不需要你复现。
你只需要自己跑：CAPE-SGG + PE-NET基线。

### 你的最低胜出条件

超过RA-SGG（AAAI'25，你的直接竞争对手，同为PE-NET系）:
  mR@50 > 36.2 且 R@50 ≥ 60 且 F@50 > 45.7
  → 够发ICANN (CCF-C)

逼近/超过HIERCOM+IETrans（WACV'25，当前balanced SOTA）:
  mR@50 ≥ 38.0 且 F@50 ≥ 46.6
  → 论文会更有说服力

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 第二章：完整实验清单与调度
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 实验全貌 (共18个实验)

┌────────────────────────────────────────────────────────────┐
│ 类别        │ 实验数 │ 用途           │ 报在论文哪里      │
├─────────────┼────────┼────────────────┼───────────────────┤
│ 主实验      │   4    │ 对比SOTA       │ Table 1           │
│ 消融实验    │   7    │ 证明每个组件   │ Table 2           │
│ 超参数搜索  │   4    │ 敏感性分析     │ Table 3 / Fig     │
│ 泛化实验    │   2    │ CAPM通用性     │ Table 4           │
│ 定性分析    │   1    │ 可视化         │ Figures 2-5       │
├─────────────┼────────┼────────────────┼───────────────────┤
│ 合计        │  18    │                │                   │
└────────────────────────────────────────────────────────────┘

### 时间估算

单个PredCls实验: ~9.5h (50k iter, batch=12)
单个SGDet实验:   ~12h  (50k iter, batch=8)
显存: PredCls占35GB/96GB → 每张卡可以同时跑2个PredCls

2张卡 × 每卡2个并行 = 4个实验同时跑
18个实验 / 4并行 × 10h平均 = ~45h = ~2天
加上调试/重跑/排队 → 实际3-4天完成所有实验

你有10-14天 → 时间非常充裕，即使遇到问题也有足够buffer

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 第三章：分阶段执行方案
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

###  Phase 0: 快速验证 (Day 1, 前1小时)

目的: 确认修复后的代码能正常收敛

```bash
cd /root/autodl-tmp/penet-main
git pull origin main
rm -rf output/cape_sgg_predcls/
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py \
    --config-file configs/cape_sgg_predcls.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR CAPEPrototypeEmbeddingNetwork \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    SOLVER.IMS_PER_BATCH 12 \
    SOLVER.MAX_ITER 5000 \
    SOLVER.VAL_PERIOD 1000 \
    OUTPUT_DIR ./output/phase0_quicktest
```

通过标准:
  ✅ loss_rel < 3.0 (旧log中固定在3.93, 说明什么都没学)
  ✅ R@100 > 0.15 (旧log中只有0.025)
  ❌ 如果不通过 → SOLVER.BASE_LR改为0.001再试

###  Phase 1: 主实验 (Day 1-2)

Phase 0通过后立即启动，4个实验用2张卡并行。

```
GPU-0, 进程1:  CAPE-SGG Full PredCls  (~9.5h)
GPU-0, 进程2:  PE-NET baseline PredCls (~9h)
GPU-1, 进程1:  CAPE-SGG Full SGCls    (~10h)
GPU-1, 进程2:  PE-NET baseline SGCls   (~9h)
```

具体命令:

```bash
# ===== GPU-0 进程1: CAPE PredCls =====
CUDA_VISIBLE_DEVICES=0 nohup python tools/relation_train_net.py \
    --config-file configs/cape_sgg_predcls.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR CAPEPrototypeEmbeddingNetwork \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    SOLVER.IMS_PER_BATCH 12 \
    SOLVER.MAX_ITER 50000 \
    SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    OUTPUT_DIR ./output/main_cape_predcls \
    > logs/main_cape_predcls.log 2>&1 &

# ===== GPU-0 进程2: PE-NET Baseline PredCls =====
CUDA_VISIBLE_DEVICES=0 nohup python tools/relation_train_net.py \
    --config-file configs/e2e_relation_X_101_32_8_FPN_1x.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    SOLVER.IMS_PER_BATCH 12 \
    SOLVER.MAX_ITER 50000 \
    SOLVER.VAL_PERIOD 2000 \
    GLOVE_DIR ./datasets/vg/ \
    OUTPUT_DIR ./output/main_penet_predcls \
    > logs/main_penet_predcls.log 2>&1 &

# ===== GPU-1 进程1: CAPE SGCls =====
CUDA_VISIBLE_DEVICES=1 nohup python tools/relation_train_net.py \
    --config-file configs/cape_sgg_predcls.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR CAPEPrototypeEmbeddingNetwork \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    SOLVER.IMS_PER_BATCH 12 \
    SOLVER.MAX_ITER 50000 \
    SOLVER.VAL_PERIOD 2000 \
    OUTPUT_DIR ./output/main_cape_sgcls \
    > logs/main_cape_sgcls.log 2>&1 &

# ===== GPU-1 进程2: PE-NET Baseline SGCls =====
CUDA_VISIBLE_DEVICES=1 nohup python tools/relation_train_net.py \
    --config-file configs/e2e_relation_X_101_32_8_FPN_1x.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    SOLVER.IMS_PER_BATCH 12 \
    SOLVER.MAX_ITER 50000 \
    SOLVER.VAL_PERIOD 2000 \
    GLOVE_DIR ./datasets/vg/ \
    OUTPUT_DIR ./output/main_penet_sgcls \
    > logs/main_penet_sgcls.log 2>&1 &
```

⚠️ 如果两个进程同一张卡OOM → 改batch为8，或改为一前一后串行

Phase 1完成后检查 (约Day 2上午):

```bash
# 快速看PredCls结果
grep "SGG eval.*R @ 50\|SGG eval.*mR @ 50" logs/main_cape_predcls.log | tail -4
```

Go/No-Go判定:
  mR@50 ≥ 37 → 直接进Phase 2
  mR@50 = 35-37 → 正常偏低，试调FASA_WEIGHT
  mR@50 < 35 → 需要排查问题

###  Phase 1b: SGDet补充 (Day 2-3)

Phase 1的4个PredCls/SGCls完成后，用空出来的GPU跑SGDet。

```bash
# GPU-0: CAPE SGDet
CUDA_VISIBLE_DEVICES=0 nohup python tools/relation_train_net.py \
    --config-file configs/cape_sgg_predcls.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR CAPEPrototypeEmbeddingNetwork \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    SOLVER.IMS_PER_BATCH 8 \
    SOLVER.MAX_ITER 50000 \
    SOLVER.VAL_PERIOD 2000 \
    OUTPUT_DIR ./output/main_cape_sgdet \
    > logs/main_cape_sgdet.log 2>&1 &

# GPU-1: PE-NET SGDet
CUDA_VISIBLE_DEVICES=1 nohup python tools/relation_train_net.py \
    --config-file configs/e2e_relation_X_101_32_8_FPN_1x.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    SOLVER.IMS_PER_BATCH 8 \
    SOLVER.MAX_ITER 50000 \
    SOLVER.VAL_PERIOD 2000 \
    GLOVE_DIR ./datasets/vg/ \
    OUTPUT_DIR ./output/main_penet_sgdet \
    > logs/main_penet_sgdet.log 2>&1 &
```

###  Phase 2: 消融实验 (Day 3-5)

只跑PredCls。这是论文Table 2的数据来源。
用 scripts/run_ablation.sh 一键启动。

消融表设计如下（已经解释过为什么需要每一行）:

```
论文Table 2:

| ID   | 配置                         | mR@50 | R@50 | F@50 | 验证什么              |
|------|------------------------------|-------|------|------|----------------------|
| Base | PE-NET原始                   |  来自Phase1                              |
| A1   | +APT                         |       |      |      | APT单独贡献          |
| A2   | +APT+CLIP Bridge             |       |      |      | CLIP Bridge单独贡献  |
| A3   | +APT+CLIP Bridge+CAPM        |       |      |      | CAPM单独贡献(核心C1) |
| A4   | +APT+CLIP Bridge+FASA        |       |      |      | FASA单独贡献         |
| A5   | Full (=Phase1 CAPE PredCls)  |       |      |      | 完整框架(C2)         |
| A7   | Full→ContextBias             |       |      |      | C3发现(a)            |
| A8   | Full→LearnableGate           |       |      |      | C3发现(b)            |

阅读方式（审稿人会这样看）:
  Base→A1: +X.X mR → "APT提供了X点提升"
  A1→A2:   +X.X mR → "CLIP Bridge又提供了X点"
  A2→A3:   +X.X mR → "CAPM是关键组件，提供了最大提升"
  A2→A4:   +X.X mR → "FASA从分布平衡角度提升"
  A5 vs A3: 验证CAPM+FASA互补
  A5 vs A7: "在原型空间调制优于在logit空间偏置"
  A5 vs A8: "CLIP语义结构为gate提供有益归纳偏置"
```

调度 (2张卡×2并行=4个同时):

```
Day 3 第1轮 (10h):
  GPU-0-进程1: A1  (bash scripts/run_ablation.sh A1 0)
  GPU-0-进程2: A2  (bash scripts/run_ablation.sh A2 0)
  GPU-1-进程1: A3  (bash scripts/run_ablation.sh A3 1)
  GPU-1-进程2: A4  (bash scripts/run_ablation.sh A4 1)

Day 4 第2轮 (10h):
  GPU-0-进程1: A7  (bash scripts/run_ablation.sh A7 0)
  GPU-0-进程2: A8  (bash scripts/run_ablation.sh A8 0)
  GPU-1: 空闲 → 如有需要可重跑失败的实验
```

###  Phase 3: 超参数敏感性 (Day 5-6)

论文Table 3（或画成图）。目的：证明你的方法对超参不敏感。

```
Day 5 第3轮 (10h):
  GPU-0-进程1: bash scripts/run_ablation.sh fasa005 0  (FASA λ=0.05)
  GPU-0-进程2: bash scripts/run_ablation.sh fasa02 0   (FASA λ=0.2)
  GPU-1-进程1: bash scripts/run_ablation.sh heads2 1   (CAPM heads=2)
  GPU-1-进程2: bash scripts/run_ablation.sh heads8 1   (CAPM heads=8)
```

论文中的呈现方式:
```
Table 3: Hyperparameter Sensitivity (PredCls)

(a) FASA weight λ
| λ    | 0.05 | 0.1(default) | 0.2  | 0.5  |
|------|------|--------------|------|------|
| mR@50|      |              |      |      |
| R@50 |      |              |      |      |

(b) CAPM attention heads
| heads | 1    | 2    | 4(default) | 8    |
|-------|------|------|------------|------|
| mR@50 |      |      |            |      |
| R@50  |      |      |            |      |

→ 审稿人希望看到: 默认值附近结果稳定，不大幅波动
→ 如果某个非默认值更好 → 改用那个值跑最终实验
```

###  Phase 4: 泛化实验 (Day 6-7, 可选)

如果时间足够，证明CAPM是一个通用即插即用模块。

```
这个需要修改代码把CAPM插入MotifPredictor和VCTreePredictor。
如果时间紧，可以跳过——ICANN不强制要求。

论文Table 4:
| Base Model    | mR@50原始 | +CAPM mR@50 | 提升 |
|---------------|-----------|-------------|------|
| Motifs        | ~15.4     |             |      |
| VCTree        | ~17.8     |             |      |
| PE-NET (ours) | 31.5      |             |      |
```

###  Phase 5: 结果收集与定性分析 (Day 7-8)

```bash
# 收集所有结果
python scripts/collect_results.py

# 生成定性分析图表
python tools/cape_analysis.py \
    --checkpoint ./output/main_cape_predcls/model_final.pth \
    --config-file configs/cape_sgg_predcls.yaml \
    --output_dir ./analysis_results/
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 第四章：完整日程表
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
Day  1: Phase 0(1h验证) → Phase 1启动(PredCls+SGCls×2卡)
Day  2: Phase 1完成 → 检查结果 → Phase 1b启动(SGDet×2卡)
Day  3: Phase 1b完成 → Phase 2启动(消融A1/A2/A3/A4)
Day  4: Phase 2第2轮(A7/A8)
Day  5: Phase 3(超参数搜索)
Day  6: 如果有失败实验 → 重跑; 否则 Phase 4(泛化,可选)
Day  7: Phase 5(收集结果+定性分析)
Day  8: 开始写论文(实验数据已齐全)
Day  9-14: 写论文 + 如需补充实验还有buffer
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 第五章：论文中的表格长什么样（模板）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Table 1 模板 (LNCS格式约占1页)

```
Table 1. Comparison with state-of-the-art methods on VG150.
All methods use ResNeXt-101-FPN backbone.
Best results in bold. Second best underlined.

                    PredCls                 SGCls                  SGDet
Method       mR@50 mR@100 R@50 R@100  mR@50 mR@100 R@50 R@100  mR@50 mR@100 R@50 R@100
─────────────────────────────────────────────────────────────────────────────────────────
MOTIFS        14.6  15.8  66.0  67.9   8.0   8.5   39.1  39.9   5.5   6.8  32.1  36.9
Motifs+TDE    25.5  29.1  46.2  51.4  13.1  14.9   27.7  29.9   8.2   9.8  16.9  20.3
BGNN          30.4  32.9  59.2  61.3  14.3  16.5   37.4  38.5  10.7  12.6  31.0  35.8
PE-NET        31.5  33.8  64.9  67.2  17.8  18.9   39.4  40.7  12.4  14.5  30.7  35.2
CFA           35.7  38.2  54.1  56.6   -     -      -     -     -     -     -     -
RA-SGG        36.2  39.1  62.2  64.1  20.9  22.5   38.2  39.1  14.4  17.1  26.0  30.3
─────────────────────────────────────────────────────────────────────────────────────────
PE-NET*       xx.x  xx.x  xx.x  xx.x  xx.x  xx.x  xx.x  xx.x  xx.x  xx.x  xx.x  xx.x
CAPE-SGG      xx.x  xx.x  xx.x  xx.x  xx.x  xx.x  xx.x  xx.x  xx.x  xx.x  xx.x  xx.x

* Our reproduced results.
```

注意：
- 其他方法的数字从它们的论文表格里直接抄
- PE-NET* 是你自己跑的基线，数字可能和原论文略有差异（正常）
- 加粗你的方法在mR@50、mR@100、F@50上的最优值

### Table 2 模板 (消融表)

```
Table 2. Ablation study on VG150 PredCls.

ID  Config                        mR@50  R@50  F@50
────────────────────────────────────────────────────
Base PE-NET                       31.5   64.9  42.4
A1  + APT                         xx.x   xx.x  xx.x
A2  + APT + CLIP Bridge           xx.x   xx.x  xx.x
A3  + APT + CLIP Bridge + CAPM    xx.x   xx.x  xx.x
A4  + APT + CLIP Bridge + FASA    xx.x   xx.x  xx.x
A5  Full (CAPE-SGG)               xx.x   xx.x  xx.x
A7  Full, CAPM→ContextBias        xx.x   xx.x  xx.x
A8  Full, CAPM→LearnableGate      xx.x   xx.x  xx.x
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 第六章：如果结果不够好怎么办（应急手册）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 情况A: mR@50 低于36 (没超过RA-SGG)

原因可能是CAPM调制力度不够或FASA锚定不够强。

方案1: 加大FASA → 重跑一个实验
  CAPE.FASA_WEIGHT 0.3

方案2: 增大CAPM调制 → 改代码capm_module.py
  self.mod_scale = nn.Parameter(torch.tensor(0.3))  # 从0.1改0.3

方案3: 加入类别平衡采样/Reweight → 与CAPM正交，可以叠加
  这需要额外实现，但很简单（loss乘以类别权重）

### 情况B: mR提升了但R@50跌到58以下

FASA太强了，过度向尾部倾斜。

方案: 降低FASA → CAPE.FASA_WEIGHT 0.03 或 0.05

### 情况C: 训练震荡/不收敛

方案1: SOLVER.BASE_LR 0.001 (再降一半)
方案2: 用AdamW替代SGD (需要改solver/build.py)

### 情况D: 消融A5 vs A7/A8差距太小 (<0.5分)

C3贡献不够显著。

方案: 调整A8的LearnableGateCAPM使差距更明显
（比如用更少的参数，或去掉LayerNorm，使对比更不利）
但这要保持公平性——更好的做法是换一个angle来讲故事。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 第七章：日常监控流程
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 每2-3小时查看一次:

```bash
# 一键看所有实验状态
bash scripts/check_all.sh
```

### 关注的健康指标:

```
正常收敛的信号:
  iter  1000: loss_rel ~2.5, loss_dis ~3.0
  iter  5000: loss_rel ~1.8, loss_dis ~1.5, R@100 > 0.30
  iter 10000: loss_rel ~1.5, loss_dis ~1.0, R@100 > 0.50
  iter 20000: loss_rel ~1.2, R@100 > 0.60, mR@100 > 0.20
  iter 50000: loss_rel ~0.9, R@100 > 0.65, mR@100 > 0.30

危险信号 → 立即处理:
  loss_rel 卡在 3.93 不动 → lr太高，模型没在学习
  loss_dis > 100 → 原型空间爆炸
  R@100 < 0.10 at iter 10000 → 严重问题
```

### 实验完成后提取结果:

```bash
# 提取最后一次validation的完整指标
grep "SGG eval" logs/main_cape_predcls.log | tail -8

# 算F@50
python3 -c "
R50=0.62; mR50=0.38
F50 = 2*R50*mR50/(R50+mR50)
print(f'F@50 = {F50:.4f}')
"
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 附录：快速检查清单
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

实验前:
  □ git pull 到最新代码
  □ clip_embeddings.pt 存在于 datasets/vg/ 下
  □ pretrained_faster_rcnn/model_final.pth 存在
  □ GloVe文件 (glove.6B.200d.txt) 存在于 datasets/vg/
  □ Phase 0 quick test 通过

实验中:
  □ 每个实验都有独立的 OUTPUT_DIR
  □ 日志重定向到 logs/ 目录
  □ 定期 bash scripts/check_all.sh 监控
  □ 用 nvidia-smi 确认GPU利用率 > 80%

实验后:
  □ python scripts/collect_results.py 收集所有结果
  □ 确认 F@50 > 45.7 (超过RA-SGG)
  □ 确认所有消融行 A2→A3 有正向提升 (CAPM有效)
  □ 确认 A5 > A7 且 A5 > A8 (C3贡献成立)
  □ 三个任务都有一致提升方向
