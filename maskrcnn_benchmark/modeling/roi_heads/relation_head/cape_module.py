"""
CAPE-Full: Complete module combining Gate + FiLM + per-class LoRA.
Most expressive variant. Gate→FiLM→LoRA pipeline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CAPEDynamic(nn.Module):
    def __init__(self, proto_dim=4096, num_classes=51, vis_dim=2048, spatial_dim=12):
        super().__init__()
        self.proto_dim = proto_dim
        self.num_classes = num_classes
        self.rank = 4
        self.alpha = 0.5
        self.max_offset_ratio = 0.10
        self._build_ctx_encoder(vis_dim, spatial_dim, 512)
        # Gate
        self.gate_net = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, proto_dim))
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, 5.0)
        # FiLM
        self.gamma_net = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, proto_dim))
        nn.init.zeros_(self.gamma_net[-1].weight)
        nn.init.zeros_(self.gamma_net[-1].bias)
        # LoRA + per-class
        self.down = nn.Linear(512, self.rank, bias=False)
        self.up = nn.Linear(self.rank, proto_dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.up.weight)
        self.class_scale_net = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Linear(128, num_classes))
        nn.init.zeros_(self.class_scale_net[-1].weight)
        nn.init.zeros_(self.class_scale_net[-1].bias)
        print(f"[CAPE-Full] dim={proto_dim}, Gate+FiLM+LoRA(r={self.rank})")

        # Warmup: Defect4 fix
        self.register_buffer('_step', torch.tensor(0, dtype=torch.long))
        self.warmup_steps = 5000

    def _warmup_mix(self, P_static_norm, P_dynamic):
        """Mix static and dynamic prototypes during warmup period."""
        if self.training:
            self._step += 1
        alpha = min(1.0, self._step.item() / self.warmup_steps)
        if alpha < 1.0:
            return F.normalize((1 - alpha) * P_static_norm + alpha * P_dynamic, dim=-1)
        return P_dynamic

    @staticmethod
    def _norm_cap(delta, proto_norms, max_ratio):
        """Cap offset norm to max_ratio * proto_norm. delta: [N,1,D] or [N,C,D]."""
        max_norm = max_ratio * proto_norms
        delta_norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = torch.clamp(max_norm / delta_norm, max=1.0)
        return delta * scale


    def _build_ctx_encoder(self, vis_dim, spatial_dim, out_dim):
        """Build context encoder: sub_vis[2048] + obj_vis[2048] + spatial[12] → ctx[out_dim]."""
        self.spatial_enc = nn.Sequential(
            nn.Linear(spatial_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 128))
        self.ctx_enc = nn.Sequential(
            nn.Linear(vis_dim * 2 + 128, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim))
        # Zero-init last layer
        nn.init.zeros_(self.ctx_enc[-1].weight)
        nn.init.zeros_(self.ctx_enc[-1].bias)

    def _encode_ctx(self, sub_vis, obj_vis, spatial):
        sp = self.spatial_enc(spatial)
        return self.ctx_enc(torch.cat([sub_vis, obj_vis, sp], dim=-1))


    def forward(self, prototypes, sub_vis, obj_vis, spatial):
        B = sub_vis.size(0)
        C, D = prototypes.shape
        ctx = self._encode_ctx(sub_vis, obj_vis, spatial)
        P = prototypes.unsqueeze(0).expand(B, -1, -1)
        P_static_norm = F.normalize(P, dim=-1)
        # Stage 1: Gate
        gate = torch.sigmoid(self.gate_net(ctx)).unsqueeze(1)
        P1 = P * gate
        # Stage 2: FiLM (scale only, no shift for stability)
        gamma = 1.0 + 0.15 * torch.tanh(self.gamma_net(ctx)).unsqueeze(1)
        P2 = gamma * P1
        # Stage 3: Per-class LoRA
        delta = (self.alpha / self.rank) * self.up(F.relu(self.down(ctx)))
        cs = torch.sigmoid(self.class_scale_net(ctx))
        delta_pc = delta.unsqueeze(1) * cs.unsqueeze(-1)
        pn = P2.norm(dim=-1, keepdim=True)
        delta_pc = self._norm_cap(delta_pc, pn, self.max_offset_ratio)
        P_dynamic = F.normalize(P2 + delta_pc, dim=-1)
        return self._warmup_mix(P_static_norm, P_dynamic)
