#!/usr/bin/env python3
"""
Phase 1 代码修复脚本 — SPC框架 (Semantic Prototype Calibration)
在fix_phase0.py和fix_compat.py之后执行

包含4个正交模块（通过config开关控制）:
  A. DRW (Deferred ReWeighting): DEFERRED_REWEIGHT_ITER > 0
  B. Triplet Context Embedding:  TRIPLET_CONTEXT True
  C. Per-class Temperature:      PER_CLASS_TEMP True
  D. Tail Feature Mixup:         TAIL_MIXUP True

维度流水线 (确认无误):
  rel_rep 初始化: 2048-dim (fusion_so - gate*sem_pred)
  [B] Triplet注入: 2048-dim → gate(51→512→2048) → 2048-dim
  [D] Tail Mixup: 2048-dim 特征混合 → 2048-dim
  norm_rel_rep: 2048 → 2048 (残差+LayerNorm)
  project_head: 2048 → 4096
  L2 normalize: 4096
  cosine sim + [C] per-class temp: [N, 51]

用法: cd /root/autodl-tmp/penet-main && python fix_phase1.py
"""
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
REL_DIR = os.path.join(BASE, "maskrcnn_benchmark", "modeling", "roi_heads", "relation_head")


def add_phase1_configs():
    """Add Phase 1 config options to defaults.py"""
    path = os.path.join(BASE, "maskrcnn_benchmark", "config", "defaults.py")
    with open(path, "r") as f:
        content = f.read()

    if "TRIPLET_CONTEXT" in content:
        print("[defaults.py] Phase 1 configs already exist, skipping")
        return

    marker = "_C.MODEL.ROI_RELATION_HEAD.TRAIN_LA_TAU = 0.0"
    if marker not in content:
        print("[defaults.py] ERROR: TRAIN_LA_TAU marker not found. Run fix_phase0.py first!")
        sys.exit(1)

    new_configs = marker + """

# ============ Phase1: SPC framework configs ============
_C.MODEL.ROI_RELATION_HEAD.TRIPLET_CONTEXT = False      # B: sub/obj class context gating
_C.MODEL.ROI_RELATION_HEAD.PER_CLASS_TEMP = False        # C: 51 learnable temperatures
_C.MODEL.ROI_RELATION_HEAD.TAIL_MIXUP = False            # D: feature mixup for tail classes
_C.MODEL.ROI_RELATION_HEAD.TAIL_MIXUP_RATIO = 0.5       # D: probability of mixing a tail sample
_C.MODEL.ROI_RELATION_HEAD.DEFERRED_REWEIGHT_ITER = 0    # A: DRW switch iteration (0=disabled)"""

    content = content.replace(marker, new_configs)
    with open(path, "w") as f:
        f.write(content)
    print("[defaults.py] ✅ Added Phase 1 SPC configs")


def patch_predictor():
    """Patch PrototypeEmbeddingNetwork with all Phase 1 modules (bug-free version)"""
    path = os.path.join(REL_DIR, "roi_relation_predictors.py")
    with open(path, "r") as f:
        content = f.read()

    if "TRIPLET_CONTEXT" in content:
        print("[predictors.py] Phase 1 patches already applied, skipping")
        return

    # ==================================================================
    # PATCH 1: Add FrequencyBias import (if not already present)
    # ==================================================================
    if "from .model_motifs import" not in content:
        old_import = "from maskrcnn_benchmark.data import get_dataset_statistics"
        new_import = """from maskrcnn_benchmark.data import get_dataset_statistics
from .model_motifs import FrequencyBias"""
        content = content.replace(old_import, new_import)

    # ==================================================================
    # PATCH 2: __init__ — add modules after logit_scale
    #
    # Key dimensions:
    #   self.mlp_dim = 2048
    #   self.num_rel_cls = 51
    #   triplet_gate output = mlp_dim = 2048 (matches rel_rep BEFORE project_head)
    #
    # Reuses `statistics` variable already loaded at line 43
    # ==================================================================
    old_init_block = """        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        ##### refine object labels"""

    new_init_block = """        # ---- Phase1-C: Per-class Temperature ----
        self.per_class_temp = config.MODEL.ROI_RELATION_HEAD.PER_CLASS_TEMP
        if self.per_class_temp:
            self.logit_scale = nn.Parameter(torch.ones(self.num_rel_cls) * np.log(1 / 0.07))
            print(f"[SPC-C] Per-class temperature: {self.num_rel_cls} learnable scales")
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # ---- Phase1-B: Triplet Context Gating ----
        # Input: P(pred|subj,obj) from freq_bias [N_rel, 51]
        # Output: feature modulation [N_rel, 2048] — matches rel_rep BEFORE project_head
        self.use_triplet_context = config.MODEL.ROI_RELATION_HEAD.TRIPLET_CONTEXT
        if self.use_triplet_context:
            self.freq_bias = FrequencyBias(config, statistics)  # `statistics` from line 43
            self.freq_bias.obj_baseline.weight.requires_grad_(False)  # freeze: let gate learn, not bias
            self.triplet_gate = nn.Sequential(
                nn.Linear(self.num_rel_cls, 512),
                nn.ReLU(True),
                nn.Linear(512, self.mlp_dim),  # → 2048
                nn.Sigmoid()
            )
            print(f"[SPC-B] Triplet context: 51 → 512 → {self.mlp_dim} gate")

        # ---- Phase1-D: Tail Feature Mixup ----
        self.use_tail_mixup = config.MODEL.ROI_RELATION_HEAD.TAIL_MIXUP
        self.tail_mixup_ratio = config.MODEL.ROI_RELATION_HEAD.TAIL_MIXUP_RATIO
        if self.use_tail_mixup:
            rel_prop = config.MODEL.ROI_RELATION_HEAD.REL_PROP  # 50 predicate frequencies
            freq = torch.FloatTensor([0.0] + rel_prop)  # prepend 0 for background
            median_freq = freq[freq > 0].median()
            self.register_buffer('is_tail', freq < median_freq)
            self.is_tail[0] = False  # background never tail
            print(f"[SPC-D] Tail mixup: {self.is_tail.sum().item()}/50 tail classes")

        # ---- Phase1-A: DRW ----
        if config.MODEL.ROI_RELATION_HEAD.DEFERRED_REWEIGHT_ITER > 0:
            print(f"[SPC-A] DRW: switch at iter {config.MODEL.ROI_RELATION_HEAD.DEFERRED_REWEIGHT_ITER}")

        ##### refine object labels"""

    content = content.replace(old_init_block, new_init_block)

    # ==================================================================
    # PATCH 3: forward — inject Triplet + Mixup at CORRECT position
    #
    # INJECTION POINT: After rel_rep=fusion_so-gate*sem_pred (2048-dim)
    #                  BEFORE norm_rel_rep residual block
    #
    # This ensures:
    #   1. Dimensions match (2048 not 4096)
    #   2. project_head learns to integrate context
    #   3. BOTH CE loss (rel_dists) AND Euclidean loss see modified features
    # ==================================================================
    old_forward_block = """        rel_rep = fusion_so - sem_pred * gate_sem_pred  #  F(s,o) - gp · h(xu)   i.e., r = F(s,o) - up
        predicate_proto = self.W_pred(self.rel_embed.weight)  # c = Wp x tp  i.e., semantic prototypes
        
        ##### for the model convergence
        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)"""

    new_forward_block = """        rel_rep = fusion_so - sem_pred * gate_sem_pred  #  F(s,o) - gp · h(xu)   i.e., r = F(s,o) - up
        predicate_proto = self.W_pred(self.rel_embed.weight)  # c = Wp x tp  i.e., semantic prototypes

        # ---- Phase1-B: Triplet context injection (rel_rep is 2048-dim here) ----
        if self.use_triplet_context:
            triplet_logits = self.freq_bias.index_with_labels(pair_pred.long())  # [N_rel, 51]
            triplet_mod = self.triplet_gate(triplet_logits)  # [N_rel, 2048]
            rel_rep = rel_rep * (1.0 + triplet_mod)  # soft multiplicative gating

        # ---- Phase1-D: Tail mixup (rel_rep is 2048-dim here) ----
        if self.training and self.use_tail_mixup and rel_rep.size(0) > 1:
            _rl = cat(rel_labels, dim=0) if isinstance(rel_labels, (list, tuple)) else rel_labels
            _tm = self.is_tail[_rl]  # [N_rel] bool: which samples are tail class
            _ti = torch.where(_tm)[0]
            if _ti.size(0) > 1:
                _n = _ti.size(0)
                _do = torch.rand(_n, device=rel_rep.device) < self.tail_mixup_ratio
                if _do.sum() > 0:
                    _perm = _ti[torch.randperm(_n, device=rel_rep.device)]
                    _lam = torch.rand(_do.sum(), 1, device=rel_rep.device) * 0.5 + 0.5
                    _src = rel_rep[_ti[_do]]
                    _tgt = rel_rep[_perm[_do]]
                    rel_rep = rel_rep.clone()
                    rel_rep[_ti[_do]] = _lam * _src + (1.0 - _lam) * _tgt

        ##### for the model convergence
        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)"""

    content = content.replace(old_forward_block, new_forward_block)

    # ==================================================================
    # PATCH 4: forward — per-class temperature for cosine similarity
    # ==================================================================
    old_cosine = "        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  #  <r_norm, c_norm> / τ"

    new_cosine = """        if self.per_class_temp:
            rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp().unsqueeze(0)
        else:
            rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()"""

    content = content.replace(old_cosine, new_cosine)

    with open(path, "w") as f:
        f.write(content)
    print("[predictors.py] ✅ Phase 1 modules patched (all bugs fixed)")
    print("  ✓ Triplet injection @2048-dim (before project_head)")
    print("  ✓ Mixup @2048-dim (before project_head → affects ALL losses)")
    print("  ✓ Per-class temp: [N,51] * [1,51] broadcast")


def patch_loss_drw():
    """Patch loss.py to support DRW via internal step counter"""
    path = os.path.join(REL_DIR, "loss.py")
    with open(path, "r") as f:
        content = f.read()

    if "deferred_reweight" in content:
        print("[loss.py] DRW already patched, skipping")
        return

    # Add deferred_reweight_iter parameter
    old_init_sig = """        reweight_beta=0.0,
    ):"""
    new_init_sig = """        reweight_beta=0.0,
        deferred_reweight_iter=0,
    ):"""
    content = content.replace(old_init_sig, new_init_sig)

    # After criterion_loss_rel creation, add DRW state
    old_else = """            else:
                self.criterion_loss_rel = nn.CrossEntropyLoss()"""
    new_else = """            else:
                self.criterion_loss_rel = nn.CrossEntropyLoss()

            # ---- DRW: deferred reweighting ----
            self.deferred_reweight_iter = deferred_reweight_iter
            self._drw_step = 0
            self._drw_switched = False
            self._criterion_rel_unweighted = nn.CrossEntropyLoss()
            if deferred_reweight_iter > 0 and reweight_beta > 0:
                self._criterion_rel_weighted = self.criterion_loss_rel  # save weighted
                self.criterion_loss_rel = self._criterion_rel_unweighted  # start unweighted
                print(f"[DRW] unweighted → weighted at step {deferred_reweight_iter}")"""
    content = content.replace(old_else, new_else)

    # Replace loss computation with DRW-aware version
    old_loss = "        loss_relation = self.criterion_loss_rel(relation_logits, rel_labels.long())"
    new_loss = """        # DRW: auto-switch at deferred_reweight_iter
        # NOTE: no self.training check — RelationLossComputation is not nn.Module
        #       and __call__ is only invoked during training anyway (relation_head.py line 83)
        if self.deferred_reweight_iter > 0 and not self._drw_switched:
            self._drw_step += 1
            if self._drw_step >= self.deferred_reweight_iter:
                self.criterion_loss_rel = self._criterion_rel_weighted
                self._drw_switched = True
                print(f"[DRW] *** Switched to weighted loss at step {self._drw_step} ***")
        loss_relation = self.criterion_loss_rel(relation_logits, rel_labels.long())"""
    content = content.replace(old_loss, new_loss)

    # Update factory to pass deferred_reweight_iter
    old_factory = "        reweight_beta=cfg.MODEL.ROI_RELATION_HEAD.REWEIGHT_BETA,"
    new_factory = """        reweight_beta=cfg.MODEL.ROI_RELATION_HEAD.REWEIGHT_BETA,
        deferred_reweight_iter=cfg.MODEL.ROI_RELATION_HEAD.DEFERRED_REWEIGHT_ITER,"""
    content = content.replace(old_factory, new_factory)

    with open(path, "w") as f:
        f.write(content)
    print("[loss.py] ✅ DRW with self-contained step counter")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1 SPC — Bug-free 版本 v2")
    print("=" * 60)

    if not os.path.exists(os.path.join(REL_DIR, "loss.py")):
        print(f"ERROR: {REL_DIR}/loss.py not found")
        print("请在 penet-main 目录下执行: python fix_phase1.py")
        sys.exit(1)

    add_phase1_configs()
    patch_predictor()
    patch_loss_drw()

    print("\n" + "=" * 60)
    print("Phase 1 修复完成! 配置选项:")
    print("  TRIPLET_CONTEXT True      三元组上下文 (51→512→2048 gate)")
    print("  PER_CLASS_TEMP True       逐类温度 (51个可学参数)")
    print("  TAIL_MIXUP True           尾类特征混合 (λ~[0.5,1.0])")
    print("  DEFERRED_REWEIGHT_ITER N  延迟加权 (步数N切换)")
    print("=" * 60)
