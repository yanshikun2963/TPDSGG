"""
CAPEv2 Predictor: PE-NET + Context-Aware Prototype Evolution (v2)
All bugs from self-critique fixed:
  - Bug1: top-level import for get_dataset_statistics
  - Bug3: sub_vis/obj_vis passed as separate signals (not fusion_so)
  - Bug4: gradient scaling 0.1x on CAPE params via hooks
  - Bug5: explicit .to(device) for spatial features
  - Bug7: correct make_fc import path
  - Defect1: context = sub_vis + obj_vis + spatial (NOT fusion_so)
  - Defect4: warmup for first 5000 steps
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.modeling.make_layers import make_fc  # Bug7 fixed
from maskrcnn_benchmark.data import get_dataset_statistics  # Bug1 fixed
from .utils_motifs import obj_edge_vectors, rel_vectors, encode_box_info, to_onehot, nms_overlaps


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def fusion_func(x, y):
    return F.relu(x + y) - (x - y) ** 2


def compute_pairwise_spatial(proposals, rel_pair_idxs, device):
    """Compute 12-dim relative spatial features for each (s,o) pair. Bug5: explicit device."""
    spatial_list = []
    for proposal, pair_idx in zip(proposals, rel_pair_idxs):
        boxes = proposal.bbox  # [N_obj, 4] xyxy, on GPU
        img_size = proposal.size
        W, H = float(img_size[0]), float(img_size[1])
        sb = boxes[pair_idx[:, 0]]  # [N_rel, 4]
        ob = boxes[pair_idx[:, 1]]
        sw = (sb[:, 2] - sb[:, 0]).clamp(min=1.0)
        sh = (sb[:, 3] - sb[:, 1]).clamp(min=1.0)
        scx = (sb[:, 0] + sb[:, 2]) / 2
        scy = (sb[:, 1] + sb[:, 3]) / 2
        ow = (ob[:, 2] - ob[:, 0]).clamp(min=1.0)
        oh = (ob[:, 3] - ob[:, 1]).clamp(min=1.0)
        ocx = (ob[:, 0] + ob[:, 2]) / 2
        ocy = (ob[:, 1] + ob[:, 3]) / 2
        dx = (ocx - scx) / sw
        dy = (ocy - scy) / sh
        log_wr = torch.log(ow / sw)
        log_hr = torch.log(oh / sh)
        log_ar = torch.log((ow * oh) / (sw * sh))
        ix1 = torch.max(sb[:, 0], ob[:, 0])
        iy1 = torch.max(sb[:, 1], ob[:, 1])
        ix2 = torch.min(sb[:, 2], ob[:, 2])
        iy2 = torch.min(sb[:, 3], ob[:, 3])
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        union = sw * sh + ow * oh - inter
        iou = inter / (union + 1e-8)
        diag = (W ** 2 + H ** 2) ** 0.5 + 1e-8
        dist = ((ocx - scx) ** 2 + (ocy - scy) ** 2).sqrt() / diag
        angle = torch.atan2(ocy - scy, ocx - scx) / 3.14159265
        spatial = torch.stack([dx, dy, log_wr, log_hr, log_ar, iou,
                               scx / W, scy / H, ocx / W, ocy / H,
                               dist, angle], dim=-1)
        spatial_list.append(spatial)
    return cat(spatial_list, dim=0).to(device)  # Bug5: ensure device


@registry.ROI_RELATION_PREDICTOR.register("CAPEv2PrototypeNetwork")
class CAPEv2PrototypeNetwork(nn.Module):
    def __init__(self, config, in_channels):
        super(CAPEv2PrototypeNetwork, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.cfg = config
        assert in_channels is not None
        self.in_channels = in_channels
        self.obj_dim = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = (
            statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes'])
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.mlp_dim = 2048
        self.post_emb = nn.Linear(self.obj_dim, self.mlp_dim * 2)
        self.embed_dim = 300
        dropout_p = 0.2
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)
        self.W_sub = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.gate_sub = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_obj = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_pred = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.vis2sem = nn.Sequential(*[
            nn.Linear(self.mlp_dim, self.mlp_dim * 2), nn.ReLU(True),
            nn.Dropout(dropout_p), nn.Linear(self.mlp_dim * 2, self.mlp_dim)])
        self.project_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim * 2, 2)
        self.linear_sub = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_obj = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_rel_rep = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.norm_sub = nn.LayerNorm(self.mlp_dim)
        self.norm_obj = nn.LayerNorm(self.mlp_dim)
        self.norm_rel_rep = nn.LayerNorm(self.mlp_dim)
        self.dropout_sub = nn.Dropout(dropout_p)
        self.dropout_obj = nn.Dropout(dropout_p)
        self.dropout_rel_rep = nn.Dropout(dropout_p)
        self.dropout_rel = nn.Dropout(dropout_p)
        self.dropout_pred = nn.Dropout(dropout_p)
        self.down_samp = MLP(self.pooling_dim, self.mlp_dim, self.mlp_dim, 2)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Object label refinement
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True)])
        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)
        self.obj_dim = in_channels
        self.out_obj = make_fc(self.hidden_dim, self.num_obj_classes)
        self.lin_obj_cyx = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim)

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        # ============================================================
        # CAPEv2: Context-Aware Prototype Evolution (dynamic prototypes)
        # ============================================================
        self.proto_dim = self.mlp_dim * 2  # 4096

        from .cape_module import CAPEDynamic
        self.cape = CAPEDynamic(
            proto_dim=self.proto_dim,
            num_classes=self.num_rel_cls,
            vis_dim=self.mlp_dim,      # 2048: per-pair sub/obj visual
            spatial_dim=12,
        )

        # Bug4 fix: gradient scaling for CAPE params (0.1x lr effectively)
        for param in self.cape.parameters():
            param.register_hook(lambda grad: grad * 0.1)

        cape_params = sum(p.numel() for p in self.cape.parameters())
        print(f"[CAPEv2] proto_dim={self.proto_dim}, mode={self.mode}")
        print(f"[CAPEv2] CAPE params: {cape_params:,} (grad_scale=0.1x)")
        print(f"[CAPEv2] Variant: {self.cape.__class__.__doc__.strip().split(chr(10))[0] if self.cape.__class__.__doc__ else 'unknown'}")

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys,
                roi_features, union_features, logger=None):
        add_losses = {}
        add_data = {}

        # Object label refinement (same as PE-NET)
        entity_dists, entity_preds = self.refine_obj_labels(roi_features, proposals)

        # Entity representation
        entity_rep = self.post_emb(roi_features)
        entity_rep = entity_rep.view(entity_rep.size(0), 2, self.mlp_dim)
        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)  # [N_obj, 2048]
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)
        entity_embeds = self.obj_embed(entity_preds)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        sub_reps = sub_rep.split(num_objs, dim=0)
        obj_reps = obj_rep.split(num_objs, dim=0)
        entity_preds_split = entity_preds.split(num_objs, dim=0)
        entity_embeds_split = entity_embeds.split(num_objs, dim=0)

        fusion_so = []
        pair_preds = []
        # Defect1 fix: collect per-pair visual features for CAPE context
        pair_sub_vis = []
        pair_obj_vis = []

        for pair_idx, sub_rep_i, obj_rep_i, entity_pred_i, entity_embed_i, proposal in zip(
            rel_pair_idxs, sub_reps, obj_reps, entity_preds_split,
            entity_embeds_split, proposals
        ):
            s_embed = self.W_sub(entity_embed_i[pair_idx[:, 0]])
            o_embed = self.W_obj(entity_embed_i[pair_idx[:, 1]])
            sem_sub = self.vis2sem(sub_rep_i[pair_idx[:, 0]])
            sem_obj = self.vis2sem(obj_rep_i[pair_idx[:, 1]])
            gate_sem_sub = torch.sigmoid(self.gate_sub(cat((s_embed, sem_sub), dim=-1)))
            gate_sem_obj = torch.sigmoid(self.gate_obj(cat((o_embed, sem_obj), dim=-1)))
            sub = s_embed + sem_sub * gate_sem_sub
            obj = o_embed + sem_obj * gate_sem_obj
            sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
            obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)
            fusion_so.append(fusion_func(sub, obj))
            pair_preds.append(torch.stack((entity_pred_i[pair_idx[:, 0]], entity_pred_i[pair_idx[:, 1]]), dim=1))

            # NEW: collect raw per-pair visual features for CAPE
            pair_sub_vis.append(sub_rep_i[pair_idx[:, 0]])  # [n_pairs, 2048]
            pair_obj_vis.append(obj_rep_i[pair_idx[:, 1]])

        fusion_so = cat(fusion_so, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        pair_sub_vis = cat(pair_sub_vis, dim=0)  # [total_pairs, 2048]
        pair_obj_vis = cat(pair_obj_vis, dim=0)  # [total_pairs, 2048]

        # Relation representation (same as PE-NET)
        sem_pred = self.vis2sem(self.down_samp(union_features))
        gate_sem_pred = torch.sigmoid(self.gate_pred(cat((fusion_so, sem_pred), dim=-1)))
        rel_rep = fusion_so - sem_pred * gate_sem_pred
        predicate_proto = self.W_pred(self.rel_embed.weight)

        # Project to matching space
        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)
        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep)))  # [N, 4096]
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))  # [51, 4096]

        # ============ CAPEv2: Dynamic Prototype Update ============
        # Compute pairwise spatial features
        device = rel_rep.device
        spatial = compute_pairwise_spatial(proposals, rel_pair_idxs, device)  # [N, 12]

        # Dynamic prototypes with chunking for memory safety
        rel_rep_norm = rel_rep / (rel_rep.norm(dim=1, keepdim=True) + 1e-8)
        logit_scale = self.logit_scale.exp()
        N = rel_rep_norm.size(0)
        chunk_size = 512

        if N <= chunk_size:
            dyn_proto = self.cape(predicate_proto, pair_sub_vis, pair_obj_vis, spatial)
            dyn_proto_n = dyn_proto / (dyn_proto.norm(dim=-1, keepdim=True) + 1e-8)
            rel_dists = torch.bmm(
                rel_rep_norm.unsqueeze(1), dyn_proto_n.transpose(1, 2)
            ).squeeze(1) * logit_scale
        else:
            chunks = []
            for s in range(0, N, chunk_size):
                e = min(s + chunk_size, N)
                dp = self.cape(predicate_proto, pair_sub_vis[s:e], pair_obj_vis[s:e], spatial[s:e])
                dp_n = dp / (dp.norm(dim=-1, keepdim=True) + 1e-8)
                cd = torch.bmm(
                    rel_rep_norm[s:e].unsqueeze(1), dp_n.transpose(1, 2)
                ).squeeze(1) * logit_scale
                chunks.append(cd)
                del dp, dp_n
            rel_dists = cat(chunks, dim=0)

        entity_dists = entity_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # Training losses (on STATIC prototypes, same as PE-NET)
        if self.training:
            predicate_proto_norm = predicate_proto / (predicate_proto.norm(dim=1, keepdim=True) + 1e-8)
            target_pn = predicate_proto_norm.clone().detach()
            simil_mat = predicate_proto_norm @ target_pn.t()
            C = self.num_rel_cls
            l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (C * C)
            add_losses["l21_loss"] = l21

            gamma2 = 7.0
            pa = predicate_proto.unsqueeze(1).expand(-1, C, -1)
            pb = predicate_proto.detach().unsqueeze(0).expand(C, -1, -1)
            pdm = (pa - pb).norm(dim=2) ** 2
            spdm, _ = torch.sort(pdm, dim=1)
            topK = spdm[:, :2].sum(dim=1) / 1
            add_losses["dist_loss2"] = torch.max(
                torch.zeros(C, device=device), -topK + gamma2).mean()

            rl = cat(rel_labels, dim=0)
            gamma1 = 1.0
            re = rel_rep.unsqueeze(1).expand(-1, C, -1)
            pe = predicate_proto.unsqueeze(0).expand(rl.size(0), -1, -1)
            ds = (re - pe).norm(dim=2) ** 2
            mn = torch.ones(rl.size(0), C, device=device)
            mn[torch.arange(rl.size(0)), rl] = 0
            dsn = ds * mn
            dsp = ds[torch.arange(rl.size(0)), rl]
            sdsn, _ = torch.sort(dsn, dim=1)
            topKn = sdsn[:, :11].sum(dim=1) / 10
            add_losses["loss_dis"] = torch.max(
                torch.zeros(rl.size(0), device=device), dsp - topKn + gamma1).mean()

        return entity_dists, rel_dists, add_losses, add_data

    def refine_obj_labels(self, roi_features, proposals):
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([p.get_field("labels") for p in proposals], dim=0) if use_gt_label else None
        pos_embed = self.pos_embed(encode_box_info(proposals))
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([p.get_field("predict_logits") for p in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight
        assert proposals[0].mode == 'xyxy'
        pos_embed = self.pos_embed(encode_box_info(proposals))
        num_objs = [len(p) for p in proposals]
        obj_pre = self.lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1))
        if self.mode == 'predcls':
            obj_preds = obj_labels.long()
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        else:
            obj_dists = self.out_obj(obj_pre)
            if self.mode == 'sgdet' and not self.training:
                boxes_per_cls = [p.get_field('boxes_per_cls') for p in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs).long()
            else:
                obj_preds = (obj_dists[:, 1:].max(1)[1] + 1).long()
        return obj_dists, obj_preds

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh
            ods = F.softmax(obj_dists[i], -1).cpu().numpy()
            ods[:, 0] = -1
            out_label = obj_dists[i].new(num_objs[i]).fill_(0)
            for j in range(num_objs[i]):
                bi, ci = np.unravel_index(ods.argmax(), ods.shape)
                out_label[int(bi)] = int(ci)
                ods[is_overlap[bi, :, ci], ci] = 0.0
                ods[bi] = -1.0
            obj_preds.append(out_label.long())
        return torch.cat(obj_preds, dim=0)
