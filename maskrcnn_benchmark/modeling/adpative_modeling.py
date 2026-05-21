import torch
import torch.nn as nn
import torch.nn.functional as F

class CoreModule(nn.Module):

    
    def __init__(self, prompt_dim=300, visual_dim=2048, hidden_dim=512, prompt_length=5):
        super().__init__()
        
        # Learnable prompt parameters
        self.detection_prompt = nn.Parameter(torch.randn(prompt_length, prompt_dim))
        self.relation_prompt = nn.Parameter(torch.randn(prompt_length, prompt_dim))
        
        # Visual feature projection layer
        self.visual_proj = nn.Linear(visual_dim, prompt_dim)
        
        # Feature fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(3 * prompt_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prompt_dim)
        )
        
        # BUG FIX: Learnable residual gate — initialized to sigmoid(-3) ≈ 0.047
        # Without this, the fusion_mlp COMPLETELY REPLACES the input GloVe embedding
        # with a random MLP output, destroying PE-NET's semantic initialization.
        # Now: output = (1-gate)*e_static + gate*mlp_output, starting near identity.
        self.residual_gate = nn.Parameter(torch.tensor(-3.0))
        
        # Initialize parameters
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for param in [self.detection_prompt, self.relation_prompt]:
            nn.init.xavier_uniform_(param)
        # BUG FIX: Initialize fusion MLP last layer near zero for safe residual startup
        nn.init.zeros_(self.fusion_mlp[-1].weight)
        nn.init.zeros_(self.fusion_mlp[-1].bias)
    
    def forward(self, e_static, v_visual, role='general'):
        """
        Args:
            e_static: static semantic feature [batch_size, prompt_dim] or [prompt_dim]
            v_visual: visual feature [batch_size, visual_dim] or [visual_dim]
            role: object role ('detection', 'subject', 'object', 'general')
        Returns:
            e_adapted: adaptive feature [batch_size, prompt_dim] or [prompt_dim]
        """
        # Ensure inputs are in batch format
        if e_static.dim() == 1:
            e_static = e_static.unsqueeze(0)
            v_visual = v_visual.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        # Select appropriate prompt
        if role == 'detection':
            prompt = self.detection_prompt
        else:  # 'subject', 'object', or 'general'
            prompt = self.relation_prompt
        
        # Prompt aggregation (mean pooling) - expand to batch dimension
        batch_size = e_static.shape[0]
        p_agg = prompt.mean(dim=0, keepdim=True)  # [1, prompt_dim]
        p_agg = p_agg.expand(batch_size, -1)      # [batch_size, prompt_dim]
        
        # Visual feature projection
        v_proj = self.visual_proj(v_visual)       # [batch_size, prompt_dim]
        
        # Feature fusion
        combined = torch.cat([p_agg, e_static, v_proj], dim=-1)  # [batch_size, 3*prompt_dim]
        delta = self.fusion_mlp(combined)     # [batch_size, prompt_dim]
        
        # BUG FIX: Residual connection — preserves GloVe embeddings
        gate = torch.sigmoid(self.residual_gate)
        e_adapted = (1 - gate) * e_static + gate * delta
        
        if single_input:
            e_adapted = e_adapted.squeeze(0)
            
        return e_adapted


class RelationalContextGating(nn.Module):
    """Relational Context Gating Module"""
    
    def __init__(self, visual_dim=2048, semantic_dim=300, num_basis=16, hidden_dim=256):
        super().__init__()
        
        self.gate_network = nn.Sequential(
            nn.Linear(visual_dim + semantic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_basis),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, v_visual, e_static):
        """
        Args:
            v_visual: visual feature [batch_size, visual_dim] or [visual_dim]
            e_static: static semantic feature [batch_size, semantic_dim] or [semantic_dim]
        Returns:
            gate_weights: gating weights [batch_size, num_basis] or [num_basis]
        """
        # Ensure inputs are in batch format
        if v_visual.dim() == 1:
            v_visual = v_visual.unsqueeze(0)
            e_static = e_static.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        # Concatenate visual and semantic features
        gate_input = torch.cat([v_visual, e_static], dim=-1)
        
        # Generate gating weights
        gate_weights = self.gate_network(gate_input)  # [batch_size, num_basis]
        
        if single_input:
            gate_weights = gate_weights.squeeze(0)
            
        return gate_weights


class BasisPromptSynthesis(nn.Module):
    """Basis Prompt Synthesis"""
    
    def __init__(self, num_basis=16, prompt_length=6, prompt_dim=300):
        super().__init__()
        
        # Basis prompt set [num_basis, prompt_length, prompt_dim]
        self.basis_prompts = nn.Parameter(torch.randn(num_basis, prompt_length, prompt_dim))
        
        # Initialize parameters
        self._init_weights()
    
    def _init_weights(self):
        """Initialize basis prompts"""
        nn.init.xavier_uniform_(self.basis_prompts)
    
    def forward(self, gate_weights):
        """
        Args:
            gate_weights: gating weights [batch_size, num_basis] or [num_basis]
        Returns:
            p_agg: aggregated prompt feature [batch_size, prompt_dim] or [prompt_dim]
        """
        # Ensure inputs are in batch format
        if gate_weights.dim() == 1:
            gate_weights = gate_weights.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        batch_size = gate_weights.shape[0]
        
        # Weighted prompt synthesis [batch_size, prompt_length, prompt_dim]
        synthesized_prompt = torch.einsum('bn,nld->bld', gate_weights, self.basis_prompts)
        
        # Aggregate prompt (mean pooling) [batch_size, prompt_dim]
        p_agg = synthesized_prompt.mean(dim=1)
        
        if single_input:
            p_agg = p_agg.squeeze(0)
            
        return p_agg


class FeatureRefinementFusion(nn.Module):
    """Feature Refinement & Fusion (FRF) Module"""
    
    def __init__(self, prompt_dim=300, visual_dim=2048, hidden_dim=512):
        super().__init__()
        
        # Visual projection layer
        self.visual_proj = nn.Linear(visual_dim, prompt_dim)
        
        # Feature refinement MLP
        self.refinement_mlp = nn.Sequential(
            nn.Linear(3 * prompt_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prompt_dim)
        )
    
    def forward(self, p_agg, e_static, v_visual):
        """
        Args:
            p_agg: aggregated prompt feature [batch_size, prompt_dim] or [prompt_dim]
            e_static: static semantic feature [batch_size, semantic_dim] or [semantic_dim]
            v_visual: visual feature [batch_size, visual_dim] or [visual_dim]
        Returns:
            e_adapted: adaptive feature [batch_size, prompt_dim] or [prompt_dim]
        """
        # Ensure inputs are in batch format
        if p_agg.dim() == 1:
            p_agg = p_agg.unsqueeze(0)
            e_static = e_static.unsqueeze(0)
            v_visual = v_visual.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        # Visual feature projection
        v_proj = self.visual_proj(v_visual)  # [batch_size, prompt_dim]
        
        # Feature fusion
        combined = torch.cat([p_agg, e_static, v_proj], dim=-1)  # [batch_size, 3*prompt_dim]
        e_adapted = self.refinement_mlp(combined)  # [batch_size, prompt_dim]
        
        if single_input:
            e_adapted = e_adapted.squeeze(0)
            
        return e_adapted


class CompositionalGeneralizationPrompter(nn.Module):
    """Integrates RCG, BPS, FRF"""
    
    def __init__(self, num_basis=16, prompt_length=6, prompt_dim=300, 
                 visual_dim=2048, hidden_dim=512):
        super().__init__()
        
        # Three sub-modules
        self.rcg = RelationalContextGating(
            visual_dim=visual_dim, 
            semantic_dim=prompt_dim, 
            num_basis=num_basis
        )
        self.bps = BasisPromptSynthesis(
            num_basis=num_basis,
            prompt_length=prompt_length,
            prompt_dim=prompt_dim
        )
        self.frf = FeatureRefinementFusion(
            prompt_dim=prompt_dim,
            visual_dim=visual_dim,
            hidden_dim=hidden_dim
        )
    
    def forward(self, e_static, v_visual):
        """
        Args:
            e_static: static semantic feature [batch_size, prompt_dim] or [prompt_dim]
            v_visual: visual feature [batch_size, visual_dim] or [visual_dim]
        Returns:
            e_adapted: adaptive feature [batch_size, prompt_dim] or [prompt_dim]
        """

        gate_weights = self.rcg(v_visual, e_static)
        

        p_agg = self.bps(gate_weights)
        

        e_adapted = self.frf(p_agg, e_static, v_visual)
        
        return e_adapted


class UnifiedAPTFramework(nn.Module):
    """Unified APT Framework supporting both standard and open-vocabulary SGG"""
    
    def __init__(self, num_classes, prompt_dim=300, visual_dim=2048, 
                 use_cgp=False, num_basis=16):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_cgp = use_cgp
        
        # Static semantic embedding
        self.static_embedding = nn.Embedding(num_classes, prompt_dim)
        
        # Standard APT module
        self.apt_module = CoreModule(
            prompt_dim=prompt_dim,
            visual_dim=visual_dim
        )
        
        
        if use_cgp:
            self.cgp_module = CompositionalGeneralizationPrompter(
                num_basis=num_basis,
                prompt_dim=prompt_dim,
                visual_dim=visual_dim
            )
    
    def forward(self, visual_features, class_labels, roles=None, is_novel=None):
        """
        Args:
            visual_features: visual features [batch_size, num_objects, visual_dim]
            class_labels: class labels [batch_size, num_objects]
            roles: role labels [batch_size, num_objects] ('detection', 'subject', 'object')
            is_novel: whether it's a novel class [batch_size, num_objects] (for open-vocabulary)
        Returns:
            adapted_features: adaptive features [batch_size, num_objects, prompt_dim]
        """
        batch_size, num_objects = class_labels.shape
        
        # Get static semantic features
        static_features = self.static_embedding(class_labels)  # [batch_size, num_objects, prompt_dim]
        
        # Initialize adaptive features
        adapted_features = []
        
        for i in range(num_objects):
            e_static = static_features[:, i, :]  # [batch_size, prompt_dim]
            v_visual = visual_features[:, i, :]  # [batch_size, visual_dim]
            
            role = roles[:, i] if roles is not None else 'general'
            novel = is_novel[:, i] if is_novel is not None else False
            
            # Select appropriate module
            if self.use_cgp and novel.any():
                # Use CGP for novel classes
                e_adapted = self.cgp_module(e_static, v_visual)
            else:
                # Use standard APT for known classes
                e_adapted = self.apt_module(e_static, v_visual, role)
            
            adapted_features.append(e_adapted)
        
        # Stack features for all objects
        adapted_features = torch.stack(adapted_features, dim=1)  # [batch_size, num_objects, prompt_dim]
        
        return adapted_features


#
