"""
Context-Aware Prototype Modulation (CAPM) Module

Core innovation of CAPE-SGG: dynamically adjusts predicate prototypes based on
subject-object semantic context using CLIP/OpenCLIP embeddings.

Key insight: The same predicate (e.g., "on") has fundamentally different visual
patterns across subject-object combinations (e.g., "cup on table" vs "person on horse").

Design: Uses per-class basis vectors so each predicate has its OWN modulation
direction, not just different scaling of a shared direction. Multi-head gating
captures diverse aspects of the subject-object semantic relationship.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class CAPM(nn.Module):
    """
    Context-Aware Prototype Modulation (CAPM).
    
    Architecture:
        1. Context: [E_clip[s]; E_clip[o]; E_clip[s]-E_clip[o]] -> W_sem -> context
        2. Multi-head gate: gate_c = sigma(heads_merge(ctx_heads · pred_heads) / tau)
        3. Per-class delta: delta_c = proj(class_basis_c * context_proj) -- EACH class has OWN direction
        4. Modulate: p_c' = p_c + scale * gate_c * delta_c
    """
    
    def __init__(self, clip_embed_dim=768, proto_dim=4096, num_obj_cls=151, 
                 num_rel_cls=51, context_hidden_dim=512, temperature=0.1,
                 num_heads=4, clip_embed_path=None):
        super(CAPM, self).__init__()
        
        self.proto_dim = proto_dim
        self.num_obj_cls = num_obj_cls
        self.num_rel_cls = num_rel_cls
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))  # learnable
        self.num_heads = num_heads
        self.context_hidden_dim = context_hidden_dim
        
        # Load precomputed CLIP embeddings
        if clip_embed_path is not None and os.path.exists(clip_embed_path):
            clip_data = torch.load(clip_embed_path, map_location='cpu')
            self.register_buffer('clip_obj_embeds', clip_data['obj_embeddings'])
            self.register_buffer('clip_pred_embeds', clip_data['pred_embeddings'])
            actual_clip_dim = self.clip_obj_embeds.shape[1]
            if actual_clip_dim != clip_embed_dim:
                clip_embed_dim = actual_clip_dim
        else:
            self.register_buffer('clip_obj_embeds', torch.randn(num_obj_cls, clip_embed_dim))
            self.register_buffer('clip_pred_embeds', torch.randn(num_rel_cls, clip_embed_dim))
            print(f"[CAPM WARNING] No CLIP embeddings at {clip_embed_path}. Using random init.")
        self.clip_embed_dim = clip_embed_dim
        
        # === Context Encoder ===
        self.W_sem = nn.Sequential(
            nn.Linear(3 * clip_embed_dim, context_hidden_dim * 2),
            nn.LayerNorm(context_hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(context_hidden_dim * 2, context_hidden_dim),
        )
        
        # === Multi-head Class-specific Gating ===
        self.pred_proj = nn.Linear(clip_embed_dim, context_hidden_dim)
        self.head_dim = context_hidden_dim // num_heads
        assert context_hidden_dim % num_heads == 0
        self.gate_merge = nn.Linear(num_heads, 1, bias=False)
        
        # === Per-class Modulation Basis (KEY INNOVATION) ===
        # Each predicate has its own direction basis vector
        self.context_to_basis = nn.Linear(context_hidden_dim, context_hidden_dim)
        self.class_basis = nn.Parameter(torch.randn(num_rel_cls, context_hidden_dim))
        self.delta_proj = nn.Sequential(
            nn.Linear(context_hidden_dim, proto_dim),
            nn.Tanh(),
        )
        
        # Learnable modulation scale
        self.mod_scale = nn.Parameter(torch.tensor(0.1))
        
        self._init_weights()
    
    def _init_weights(self):
        # W_sem (context encoder): conservative init — context should start small
        for m in self.W_sem:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        # delta_proj: standard init — output is bounded by Tanh anyway
        for m in self.delta_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # pred_proj: moderate init for gate computation & class_basis seeding
        nn.init.xavier_uniform_(self.pred_proj.weight, gain=0.5)
        nn.init.zeros_(self.pred_proj.bias)
        
        # context_to_basis: standard init so context_proj has meaningful magnitude
        nn.init.xavier_uniform_(self.context_to_basis.weight, gain=1.0)
        nn.init.zeros_(self.context_to_basis.bias)
        
        # Semantic init for class_basis from CLIP pred embeddings
        with torch.no_grad():
            clip_proj = F.linear(self.clip_pred_embeds, self.pred_proj.weight, self.pred_proj.bias)
            self.class_basis.copy_(clip_proj * 0.3)  # 0.3 gives CAPM meaningful initial modulation
        nn.init.ones_(self.gate_merge.weight)
    
    def forward(self, predicate_proto, pair_pred):
        """
        Args:
            predicate_proto: [num_rel_cls, proto_dim]
            pair_pred: [num_pairs, 2]
        Returns:
            modulated_proto: [num_pairs, num_rel_cls, proto_dim]
        """
        num_pairs = pair_pred.shape[0]
        
        sub_classes = pair_pred[:, 0].long().clamp(0, self.num_obj_cls - 1)
        obj_classes = pair_pred[:, 1].long().clamp(0, self.num_obj_cls - 1)
        e_sub = self.clip_obj_embeds[sub_classes]
        e_obj = self.clip_obj_embeds[obj_classes]
        
        # Context encoding
        context_input = torch.cat([e_sub, e_obj, e_sub - e_obj], dim=-1)
        context = self.W_sem(context_input)  # [N, hidden_dim]
        
        # Multi-head gating
        pred_ctx = self.pred_proj(self.clip_pred_embeds)  # [C, hidden_dim]
        ctx_heads = context.view(num_pairs, self.num_heads, self.head_dim)
        pred_heads = pred_ctx.view(self.num_rel_cls, self.num_heads, self.head_dim)
        gate_scores = torch.einsum('nhd,chd->nhc', ctx_heads, pred_heads) / (self.head_dim ** 0.5)
        gate_scores = gate_scores.permute(0, 2, 1)  # [N, C, num_heads]
        gates = torch.sigmoid(self.gate_merge(gate_scores).squeeze(-1) / self.temperature.clamp(min=0.01))  # [N, C]
        
        # Per-class modulation direction
        context_proj = self.context_to_basis(context)  # [N, hidden_dim]
        class_context = context_proj.unsqueeze(1) * self.class_basis.unsqueeze(0)  # [N, C, hidden_dim]
        N, C, H = class_context.shape
        delta_per_class = self.delta_proj(class_context.reshape(N * C, H)).view(N, C, self.proto_dim)
        
        # Apply modulation
        proto_expanded = predicate_proto.unsqueeze(0).expand(num_pairs, -1, -1)
        modulated_proto = proto_expanded + self.mod_scale * gates.unsqueeze(-1) * delta_per_class
        
        return modulated_proto
    
    def get_analysis_data(self, pair_pred):
        """Extract gate activations and modulation info for visualization."""
        with torch.no_grad():
            sub_classes = pair_pred[:, 0].long().clamp(0, self.num_obj_cls - 1)
            obj_classes = pair_pred[:, 1].long().clamp(0, self.num_obj_cls - 1)
            e_sub = self.clip_obj_embeds[sub_classes]
            e_obj = self.clip_obj_embeds[obj_classes]
            context = self.W_sem(torch.cat([e_sub, e_obj, e_sub - e_obj], dim=-1))
            pred_ctx = self.pred_proj(self.clip_pred_embeds)
            ctx_heads = context.view(-1, self.num_heads, self.head_dim)
            pred_heads = pred_ctx.view(self.num_rel_cls, self.num_heads, self.head_dim)
            gate_scores = torch.einsum('nhd,chd->nhc', ctx_heads, pred_heads) / (self.head_dim ** 0.5)
            gate_scores = gate_scores.permute(0, 2, 1)
            gates = torch.sigmoid(self.gate_merge(gate_scores).squeeze(-1) / self.temperature.clamp(min=0.01))
        return {'gates': gates, 'context': context, 'mod_scale': self.mod_scale.item()}


class ContextBiasBaseline(nn.Module):
    """
    Ablation A7: Logit-space context bias baseline.
    Instead of modulating prototypes (CAPM), adds bias to logits.
    This proves that prototype-space modulation > logit-space bias.
    """
    
    def __init__(self, clip_embed_dim=768, num_obj_cls=151, num_rel_cls=51,
                 context_hidden_dim=512, clip_embed_path=None):
        super(ContextBiasBaseline, self).__init__()
        
        if clip_embed_path is not None and os.path.exists(clip_embed_path):
            clip_data = torch.load(clip_embed_path, map_location='cpu')
            self.register_buffer('clip_obj_embeds', clip_data['obj_embeddings'])
            actual_dim = self.clip_obj_embeds.shape[1]
            if actual_dim != clip_embed_dim:
                clip_embed_dim = actual_dim
        else:
            self.register_buffer('clip_obj_embeds', torch.randn(num_obj_cls, clip_embed_dim))
        
        self.num_obj_cls = num_obj_cls
        self.bias_net = nn.Sequential(
            nn.Linear(3 * clip_embed_dim, context_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(context_hidden_dim, num_rel_cls),
        )
        nn.init.zeros_(self.bias_net[-1].weight)
        nn.init.zeros_(self.bias_net[-1].bias)
    
    def forward(self, rel_dists, pair_pred):
        """Add context-dependent bias to logits (not prototypes)."""
        sub_classes = pair_pred[:, 0].long().clamp(0, self.num_obj_cls - 1)
        obj_classes = pair_pred[:, 1].long().clamp(0, self.num_obj_cls - 1)
        e_sub = self.clip_obj_embeds[sub_classes]
        e_obj = self.clip_obj_embeds[obj_classes]
        context_input = torch.cat([e_sub, e_obj, e_sub - e_obj], dim=-1)
        bias = self.bias_net(context_input)  # [N, num_rel_cls]
        return rel_dists + bias


class LearnableGateCAPM(nn.Module):
    """
    Ablation A8: CAPM with purely learnable gate (no CLIP semantic structure).
    Same architecture as full CAPM EXCEPT: gating uses learnable random vectors
    instead of CLIP predicate embeddings. Per-class basis is KEPT identical
    to ensure fair comparison (only isolates CLIP gate contribution).
    """
    
    def __init__(self, clip_embed_dim=768, proto_dim=4096, num_obj_cls=151,
                 num_rel_cls=51, context_hidden_dim=512, temperature=0.1,
                 num_heads=4, clip_embed_path=None):
        super(LearnableGateCAPM, self).__init__()
        
        self.num_obj_cls = num_obj_cls
        self.num_rel_cls = num_rel_cls
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        self.proto_dim = proto_dim
        self.num_heads = num_heads
        self.head_dim = context_hidden_dim // num_heads
        
        if clip_embed_path is not None and os.path.exists(clip_embed_path):
            clip_data = torch.load(clip_embed_path, map_location='cpu')
            self.register_buffer('clip_obj_embeds', clip_data['obj_embeddings'])
            actual_dim = self.clip_obj_embeds.shape[1]
            if actual_dim != clip_embed_dim:
                clip_embed_dim = actual_dim
        else:
            self.register_buffer('clip_obj_embeds', torch.randn(num_obj_cls, clip_embed_dim))
        
        # Same context encoder as full CAPM
        self.W_sem = nn.Sequential(
            nn.Linear(3 * clip_embed_dim, context_hidden_dim * 2),
            nn.LayerNorm(context_hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(context_hidden_dim * 2, context_hidden_dim),
        )
        
        # KEY DIFFERENCE: learnable random embeddings instead of CLIP pred embeddings for gating
        self.learnable_pred_embeds = nn.Parameter(torch.randn(num_rel_cls, context_hidden_dim))
        nn.init.xavier_uniform_(self.learnable_pred_embeds)
        self.gate_merge = nn.Linear(num_heads, 1, bias=False)
        nn.init.ones_(self.gate_merge.weight)
        
        # SAME per-class basis as full CAPM (for fair comparison)
        self.context_to_basis = nn.Linear(context_hidden_dim, context_hidden_dim)
        self.class_basis = nn.Parameter(torch.randn(num_rel_cls, context_hidden_dim))
        nn.init.xavier_uniform_(self.class_basis)
        self.delta_proj = nn.Sequential(
            nn.Linear(context_hidden_dim, proto_dim),
            nn.Tanh(),
        )
        self.mod_scale = nn.Parameter(torch.tensor(0.1))
        
        for m in self.W_sem:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.delta_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.context_to_basis.weight, gain=1.0)
        nn.init.zeros_(self.context_to_basis.bias)
    
    def forward(self, predicate_proto, pair_pred):
        num_pairs = pair_pred.shape[0]
        sub_classes = pair_pred[:, 0].long().clamp(0, self.num_obj_cls - 1)
        obj_classes = pair_pred[:, 1].long().clamp(0, self.num_obj_cls - 1)
        e_sub = self.clip_obj_embeds[sub_classes]
        e_obj = self.clip_obj_embeds[obj_classes]
        
        context = self.W_sem(torch.cat([e_sub, e_obj, e_sub - e_obj], dim=-1))
        
        # Gating with LEARNABLE embeddings (not CLIP) — multi-head for fair comparison
        ctx_heads = context.view(num_pairs, self.num_heads, self.head_dim)
        pred_heads = self.learnable_pred_embeds.view(self.num_rel_cls, self.num_heads, self.head_dim)
        gate_scores = torch.einsum('nhd,chd->nhc', ctx_heads, pred_heads) / (self.head_dim ** 0.5)
        gate_scores = gate_scores.permute(0, 2, 1)
        gates = torch.sigmoid(self.gate_merge(gate_scores).squeeze(-1) / self.temperature.clamp(min=0.01))
        
        # Same per-class modulation as full CAPM
        context_proj = self.context_to_basis(context)
        class_context = context_proj.unsqueeze(1) * self.class_basis.unsqueeze(0)
        N, C, H = class_context.shape
        delta_per_class = self.delta_proj(class_context.reshape(N * C, H)).view(N, C, self.proto_dim)
        
        proto_expanded = predicate_proto.unsqueeze(0).expand(num_pairs, -1, -1)
        modulated_proto = proto_expanded + self.mod_scale * gates.unsqueeze(-1) * delta_per_class
        
        return modulated_proto
