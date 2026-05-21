# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import numpy.random as npr

from maskrcnn_benchmark.layers import smooth_l1_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat


class RelationLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN

    CB-Loss is OPTIONAL, controlled by cb_loss_beta:
      - cb_loss_beta = None  -> vanilla CrossEntropyLoss (PE-Net original behavior)
      - cb_loss_beta = 0.9999 -> Class-Balanced reweighting (Cui et al., CVPR 2019)

    The n_c (per-class effective number) is derived from REL_PROP (static frequencies
    hardcoded in defaults.py). This is invariant to dataset relabeling like IETrans,
    so CB-Loss and IETrans relabeling are mathematically orthogonal.
    """

    def __init__(
        self,
        attri_on,
        num_attri_cat,
        max_num_attri,
        attribute_sampling,
        attribute_bgfg_ratio,
        use_label_smoothing,
        predicate_proportion,
        cb_loss_beta=None,
    ):
        self.attri_on = attri_on
        self.num_attri_cat = num_attri_cat
        self.max_num_attri = max_num_attri
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio
        self.use_label_smoothing = use_label_smoothing
        self.pred_weight = (1.0 / torch.FloatTensor([0.5, ] + predicate_proportion)).cuda()

        if self.use_label_smoothing:
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
        else:
            if cb_loss_beta is not None:
                beta = cb_loss_beta
                # n_c comes from static REL_PROP, invariant to relabeling
                total_fg_samples = 287000  # approximate VG150 fg count
                fg_counts = np.array(predicate_proportion) * total_fg_samples
                fg_counts = np.maximum(fg_counts, 1.0)
                bg_count = total_fg_samples * 3.0  # bg:fg ~ 3:1 (POSITIVE_FRACTION=0.25)
                samples_per_cls = np.concatenate([[bg_count], fg_counts])

                num_classes = len(samples_per_cls)
                effective_num = 1.0 - np.power(beta, samples_per_cls)
                cb_weights = (1.0 - beta) / np.array(effective_num)
                cb_weights = cb_weights / np.sum(cb_weights) * num_classes

                self.criterion_loss = nn.CrossEntropyLoss(
                    weight=torch.FloatTensor(cb_weights).cuda()
                )
                print("=" * 60)
                print(f"[CB-Loss] ENABLED, beta = {beta}")
                print(f"[CB-Loss] Weight range: [{cb_weights.min():.6f}, {cb_weights.max():.6f}]")
                print(f"[CB-Loss] Weight ratio (max/min): {cb_weights.max()/cb_weights.min():.2f}")
                print(f"[CB-Loss] Background (cls 0) weight: {cb_weights[0]:.6f}")
                print(f"[CB-Loss] Applied ONLY to relation classification loss")
                print("=" * 60)
            else:
                # PE-Net original behavior
                self.criterion_loss = nn.CrossEntropyLoss()
                print("[CB-Loss] DISABLED (vanilla CrossEntropyLoss for relation classification)")

        # Object classification loss: ALWAYS unweighted
        if self.use_label_smoothing:
            self.criterion_loss_obj = Label_Smoothing_Regression(e=0.01)
        else:
            self.criterion_loss_obj = nn.CrossEntropyLoss()

    def __call__(self, proposals, rel_labels, relation_logits, refine_logits):
        if self.attri_on:
            if isinstance(refine_logits[0], (list, tuple)):
                refine_obj_logits, refine_att_logits = refine_logits
            else:
                self.attri_on = False
                refine_obj_logits = refine_logits
        else:
            refine_obj_logits = refine_logits

        relation_logits = cat(relation_logits, dim=0)
        refine_obj_logits = cat(refine_obj_logits, dim=0)

        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        rel_labels = cat(rel_labels, dim=0)

        loss_relation = self.criterion_loss(relation_logits, rel_labels.long())
        loss_refine_obj = self.criterion_loss_obj(refine_obj_logits, fg_labels.long())

        if self.attri_on:
            refine_att_logits = cat(refine_att_logits, dim=0)
            fg_attributes = cat([proposal.get_field("attributes") for proposal in proposals], dim=0)

            attribute_targets, fg_attri_idx = self.generate_attributes_target(fg_attributes)
            if float(fg_attri_idx.sum()) > 0:
                refine_att_logits = refine_att_logits[fg_attri_idx > 0]
                attribute_targets = attribute_targets[fg_attri_idx > 0]
            else:
                refine_att_logits = refine_att_logits[0].view(1, -1)
                attribute_targets = attribute_targets[0].view(1, -1)

            loss_refine_att = self.attribute_loss(
                refine_att_logits, attribute_targets,
                fg_bg_sample=self.attribute_sampling,
                bg_fg_ratio=self.attribute_bgfg_ratio,
            )
            return loss_relation, (loss_refine_obj, loss_refine_att)
        else:
            return loss_relation, loss_refine_obj

    def generate_attributes_target(self, attributes):
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):
        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            num_bg = max(int(num_fg * bg_fg_ratio), 1)
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1)
        logpt = F.log_softmax(input)
        logpt = logpt.index_select(-1, target).diag()
        logpt = logpt.view(-1)
        pt = logpt.exp()
        logpt = logpt * self.alpha * (target > 0).float() + logpt * (1 - self.alpha) * (target <= 0).float()
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def make_roi_relation_loss_evaluator(cfg):
    # Read CB-Loss settings from config (default: disabled, PE-Net original)
    if cfg.MODEL.ROI_RELATION_HEAD.USE_CB_LOSS:
        cb_loss_beta = cfg.MODEL.ROI_RELATION_HEAD.CB_LOSS_BETA
    else:
        cb_loss_beta = None

    loss_evaluator = RelationLossComputation(
        cfg.MODEL.ATTRIBUTE_ON,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
        cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
        cb_loss_beta=cb_loss_beta,
    )

    return loss_evaluator
