"""
Frequency-Aware Semantic Anchoring (FASA) Module

Prevents tail-class prototype drift by tethering prototypes to CLIP semantic
anchors with frequency-dependent strength.

Enhanced design:
1. Multi-layer anchor projection with bottleneck (preserves CLIP structure)
2. Cosine + L2 hybrid anchoring loss (scale-invariant + position-aware)
3. Continuous sigmoid frequency weighting
4. stop_gradient on anchors
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class FASA(nn.Module):
    """
    Frequency-Aware Semantic Anchoring.
    
    BUG FIX: Complete redesign of anchor computation.
    
    Previous bugs (both present simultaneously):
    1. When anchors were detached: anchor_proj never received gradients, so anchors
       stayed at RANDOM initialization. FASA pulled prototypes toward random noise.
    2. When anchors were NOT detached: cosine loss created mutual attraction between
       anchors and prototypes. L2 loss trained anchor_proj to track prototype positions.
       Net result: anchors collapsed to match prototypes, providing ZERO regularization.
    
    Fix: Use FIXED anchors pre-computed at initialization via orthogonal projection
    of CLIP predicate embeddings. Anchors preserve CLIP's semantic structure (relative
    distances between predicates) and never change during training. Only the cosine
    loss is used, pulling prototypes toward these fixed semantic positions.
    
    Loss: L_anchor = Σ_c α_c · (1 - cos(p_c, a_c_fixed))
    where α_c = sigmoid(k · (log(N_median) - log(N_c)))
    """
    
    def __init__(self, clip_embed_dim=768, proto_dim=4096, num_rel_cls=51,
                 temperature_k=2.0, clip_embed_path=None, predicate_freq=None,
                 cos_weight=0.5, l2_weight=0.5):
        super(FASA, self).__init__()
        
        self.num_rel_cls = num_rel_cls
        self.temperature_k = temperature_k
        
        # Load CLIP predicate embeddings
        if clip_embed_path is not None and os.path.exists(clip_embed_path):
            clip_data = torch.load(clip_embed_path, map_location='cpu')
            clip_pred_embeds = clip_data['pred_embeddings']
            self.register_buffer('clip_pred_embeds', clip_pred_embeds)
            actual_dim = clip_pred_embeds.shape[1]
            if actual_dim != clip_embed_dim:
                clip_embed_dim = actual_dim
        else:
            self.register_buffer('clip_pred_embeds', torch.randn(num_rel_cls, clip_embed_dim))
            print(f"[FASA WARNING] No CLIP embeddings at {clip_embed_path}.")
        
        # BUG FIX: Pre-compute FIXED anchors using orthogonal random projection.
        # This preserves CLIP's semantic structure (relative predicate distances)
        # in the proto_dim space, without any trainable parameters that could
        # cause anchor collapse.
        proj_matrix = torch.empty(clip_embed_dim, proto_dim)
        nn.init.orthogonal_(proj_matrix)
        with torch.no_grad():
            fixed_anchors = self.clip_pred_embeds @ proj_matrix  # [num_rel_cls, proto_dim]
            fixed_anchors = F.normalize(fixed_anchors, dim=-1)  # normalize for cosine
        self.register_buffer('fixed_anchors', fixed_anchors)
        print(f"[FASA] Fixed anchors computed: CLIP {clip_embed_dim}d → proto {proto_dim}d (orthogonal projection)")
        
        # Frequency-aware weights
        if predicate_freq is not None:
            self._compute_freq_weights(predicate_freq)
        else:
            self.register_buffer('freq_weights', torch.ones(num_rel_cls))
    
    def _compute_freq_weights(self, predicate_freq):
        """α_c = sigmoid(k · (log(N_median) - log(N_c)))
        Background class (index 0) is excluded from median computation and gets weight=0.
        """
        freq = torch.FloatTensor(predicate_freq)
        freq = freq.clamp(min=1.0)
        log_freq = torch.log(freq)
        # Compute median over FOREGROUND classes only (exclude index 0)
        log_median = torch.median(log_freq[1:])
        weights = torch.sigmoid(self.temperature_k * (log_median - log_freq))
        weights[0] = 0.0  # background class should NOT be anchored
        self.register_buffer('freq_weights', weights)
    
    def set_predicate_freq(self, predicate_freq):
        self._compute_freq_weights(predicate_freq)
    
    def forward(self, predicate_proto):
        """
        Compute FASA anchoring loss (cosine only, with fixed anchors).
        
        Args:
            predicate_proto: [num_rel_cls, proto_dim]
        Returns:
            anchor_loss: scalar
        """
        # Cosine distance: pull prototypes toward fixed CLIP-based anchors.
        # Anchors are buffers (no grad), so only prototypes receive gradient.
        proto_norm = F.normalize(predicate_proto, dim=-1)
        cos_dist = 1.0 - (proto_norm * self.fixed_anchors).sum(dim=-1)  # [num_rel_cls]
        loss = (self.freq_weights * cos_dist).mean()
        
        return loss
    
    def get_anchor_distances(self, predicate_proto):
        """Per-class analysis."""
        with torch.no_grad():
            cos = F.cosine_similarity(predicate_proto, self.fixed_anchors, dim=-1)
        return {'cosine_sim': cos, 'freq_weights': self.freq_weights}
