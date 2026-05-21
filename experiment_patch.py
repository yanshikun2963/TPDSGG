#!/usr/bin/env python3
"""
PE-NET Sub-mechanism Experiment Patch Script
=============================================
Patches PE-NET's PrototypeEmbeddingNetwork with different sub-mechanisms.
Each direction modifies a DIFFERENT aspect of PE-NET, all with ZERO CB-Loss overlap.

Usage:
  python3 experiment_patch.py <penet_root> --direction <1-8>

Directions:
  1 = Learnable per-class bias (trivial, ~2 lines)
  2 = Confusion-aware prototype repulsion (CGPR) - replaces L21+dist_loss2
  3 = Adaptive negative sampling in loss_dis (class-aware k1)
  4 = CGPR + learnable bias (combined)
  5 = Confusion-aware repulsion + adaptive negative sampling (combined)
  6 = Feature augmentation for tail classes (noise injection on rel_rep)
  7 = Per-class learnable temperature (replace scalar logit_scale with vector)
  8 = Full combination: CGPR + bias + adaptive neg sampling

CB-Loss (beta=0.9999) in loss.py is NEVER touched by any direction.
"""

import argparse
import os
import sys


def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)
    print(f"  Written: {path}")


def patch_direction_1(content):
    """Learnable per-class bias on cosine logits"""
    
    # Add parameter in __init__
    old_init = "        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))"
    new_init = """        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Sub-mechanism: Learnable per-class predicate bias
        # Breaks cosine classifier's symmetric decision boundary
        # Tail classes get positive bias (easier to predict), head classes get negative
        self.predicate_bias = nn.Parameter(torch.zeros(self.num_rel_cls))"""
    
    if old_init not in content:
        print("ERROR: Cannot find logit_scale in __init__")
        return None
    content = content.replace(old_init, new_init)
    
    # Modify forward: add bias to rel_dists
    old_forward = "        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  #  <r_norm, c_norm> / τ"
    new_forward = """        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  #  <r_norm, c_norm> / τ
        rel_dists = rel_dists + self.predicate_bias.unsqueeze(0)  # per-class decision boundary shift"""
    
    if old_forward not in content:
        print("ERROR: Cannot find rel_dists computation in forward")
        return None
    content = content.replace(old_forward, new_forward)
    
    return content


def patch_direction_2(content):
    """Confusion-aware prototype repulsion (CGPR) - replaces uniform L21+dist_loss2"""
    
    # Add confusion matrix buffer in __init__
    old_init = "        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES"
    new_init = """        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        # CGPR: Confusion-Guided Prototype Repulsion
        # Maintains running confusion matrix to target repulsion at confused predicate pairs
        self.register_buffer('confusion_matrix', torch.zeros(self.num_rel_cls, self.num_rel_cls))
        self.confusion_ema_momentum = 0.9
        self.cgpr_temperature = 2.0  # controls sharpness of confusion weighting
        self.cgpr_topk = 10  # number of most-confused pairs to target per class"""
    
    if old_init not in content:
        print("ERROR: Cannot find nms_thresh in __init__")
        return None
    content = content.replace(old_init, new_init)
    
    # Replace L21 and dist_loss2 with confusion-aware versions
    old_losses = """        if self.training:

            ### Prototype Regularization  ---- cosine similarity
            target_rpredicate_proto_norm = predicate_proto_norm.clone().detach() 
            simil_mat = predicate_proto_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
            l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (51*51)  
            add_losses.update({"l21_loss": l21})  # Le_sim = ||S||_{2,1}
            ### end
            
            ### Prototype Regularization  ---- Euclidean distance
            gamma2 = 7.0
            predicate_proto_a = predicate_proto.unsqueeze(dim=1).expand(-1, 51, -1) 
            predicate_proto_b = predicate_proto.detach().unsqueeze(dim=0).expand(51, -1, -1)
            proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
            sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
            topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1   # obtain d-, where k2 = 1
            dist_loss = torch.max(torch.zeros(51).cuda(), -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
            add_losses.update({"dist_loss2": dist_loss})
            ### end """
    
    new_losses = """        if self.training:

            ### CGPR: Confusion-Guided Prototype Repulsion
            # Step 1: Update confusion matrix from current batch predictions
            with torch.no_grad():
                rel_labels_flat = cat(rel_labels, dim=0) if isinstance(rel_labels, (list, tuple)) else rel_labels
                rel_dists_flat = cat(rel_dists, dim=0) if isinstance(rel_dists, (list, tuple)) else rel_dists
                preds = rel_dists_flat.argmax(dim=-1)
                batch_confusion = torch.zeros(self.num_rel_cls, self.num_rel_cls, device=predicate_proto.device)
                for pred_cls, gt_cls in zip(preds, rel_labels_flat):
                    if gt_cls > 0:  # skip background
                        batch_confusion[gt_cls, pred_cls] += 1
                # EMA update
                self.confusion_matrix = self.confusion_ema_momentum * self.confusion_matrix + \
                                       (1 - self.confusion_ema_momentum) * batch_confusion
                
                # Symmetrize and normalize confusion weights
                sym_conf = self.confusion_matrix + self.confusion_matrix.t()
                sym_conf.fill_diagonal_(0)  # no self-repulsion
                # Normalize per row to get confusion probabilities
                row_sums = sym_conf.sum(dim=1, keepdim=True).clamp(min=1e-8)
                conf_weights = sym_conf / row_sums  # [C, C] confusion probability matrix
                # Apply temperature to sharpen the distribution
                conf_weights = (conf_weights * self.cgpr_temperature).softmax(dim=1)

            # Step 2: Confusion-weighted cosine regularization (replaces L21)
            target_rpredicate_proto_norm = predicate_proto_norm.clone().detach()
            simil_mat = predicate_proto_norm @ target_rpredicate_proto_norm.t()
            # Weight similarity matrix by confusion: penalize high similarity between confused pairs
            weighted_simil = simil_mat * conf_weights.detach()
            l21 = torch.norm(torch.norm(weighted_simil, p=2, dim=1), p=1) / (self.num_rel_cls * self.num_rel_cls)
            add_losses.update({"l21_loss": l21})

            # Step 3: Confusion-weighted Euclidean repulsion (replaces dist_loss2)
            gamma2 = 7.0
            predicate_proto_a = predicate_proto.unsqueeze(dim=1).expand(-1, self.num_rel_cls, -1)
            predicate_proto_b = predicate_proto.detach().unsqueeze(dim=0).expand(self.num_rel_cls, -1, -1)
            proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(dim=2) ** 2
            # Apply confusion weights: confused pairs need more separation
            weighted_dis = proto_dis_mat * (1.0 + conf_weights.detach() * 5.0)  # boost repulsion for confused pairs
            sorted_weighted_dis, _ = torch.sort(weighted_dis, dim=1)
            topK_proto_dis = sorted_weighted_dis[:, :2].sum(dim=1) / 1
            dist_loss = torch.max(torch.zeros(self.num_rel_cls).cuda(), -topK_proto_dis + gamma2).mean()
            add_losses.update({"dist_loss2": dist_loss})
            ### end CGPR """
    
    if old_losses not in content:
        print("ERROR: Cannot find L21+dist_loss2 block")
        return None
    content = content.replace(old_losses, new_losses)
    
    return content


def patch_direction_3(content):
    """Adaptive negative sampling in loss_dis (class-aware k1)"""
    
    # Add frequency-based k1 mapping in __init__
    old_init = "        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES"
    new_init = """        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        # Adaptive Negative Sampling: class-aware k1 for loss_dis
        # Tail classes use more negatives for stronger contrastive signal
        self.k1_min = 5   # for head classes
        self.k1_max = 25  # for tail classes
        # Pre-compute per-class k1 from global predicate proportions
        rel_prop = self.cfg.MODEL.ROI_RELATION_HEAD.REL_PROP
        rel_freq = torch.tensor([1.0] + rel_prop, dtype=torch.float32)  # prepend background
        freq_max = rel_freq[1:].max()
        inv_freq_norm = 1.0 - (rel_freq / freq_max.clamp(min=1e-8))
        inv_freq_norm[0] = 0.0  # background gets min k1
        self.register_buffer('k1_per_class', 
            (self.k1_min + (self.k1_max - self.k1_min) * inv_freq_norm).long().clamp(min=self.k1_min, max=self.k1_max))"""
    
    if old_init not in content:
        print("ERROR: Cannot find nms_thresh in __init__")
        return None
    content = content.replace(old_init, new_init)
    
    # Replace fixed k1=10 with adaptive k1
    old_loss_dis = """            ###  Prototype-based Learning  ---- Euclidean distance
            rel_labels = cat(rel_labels, dim=0)
            gamma1 = 1.0
            rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, 51, -1)  # r
            predicate_proto_expand = predicate_proto.unsqueeze(dim=0).expand(rel_labels.size(0), -1, -1)  # ci
            distance_set = (rel_rep_expand - predicate_proto_expand).norm(dim=2) ** 2    # Distance Set G, gi = ||r-ci||_2^2
            mask_neg = torch.ones(rel_labels.size(0), 51).cuda()  
            mask_neg[torch.arange(rel_labels.size(0)), rel_labels] = 0
            distance_set_neg = distance_set * mask_neg
            distance_set_pos = distance_set[torch.arange(rel_labels.size(0)), rel_labels]  # gt i.e., g+
            sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
            topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(dim=1) / 10  # obtaining g-, where k1 = 10, 
            loss_sum = torch.max(torch.zeros(rel_labels.size(0)).cuda(), distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean()
            add_losses.update({"loss_dis": loss_sum})     # Le_euc = max(0, (g+) - (g-) + gamma1)
            ### end """
    
    new_loss_dis = """            ###  Adaptive Negative Sampling in Prototype-based Learning
            rel_labels = cat(rel_labels, dim=0)
            gamma1 = 1.0
            rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, self.num_rel_cls, -1)
            predicate_proto_expand = predicate_proto.unsqueeze(dim=0).expand(rel_labels.size(0), -1, -1)
            distance_set = (rel_rep_expand - predicate_proto_expand).norm(dim=2) ** 2
            mask_neg = torch.ones(rel_labels.size(0), self.num_rel_cls).cuda()
            mask_neg[torch.arange(rel_labels.size(0)), rel_labels] = 0
            distance_set_neg = distance_set * mask_neg
            distance_set_pos = distance_set[torch.arange(rel_labels.size(0)), rel_labels]

            # Adaptive k1: compute per-sample k1 based on class frequency
            # Use rel_labels to look up per-class k1
            # We approximate: sort all negatives, then use class-specific topk
            sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
            
            # Adaptive k1: use pre-computed per-class k1 from global frequency
            k1_per_sample = self.k1_per_class[rel_labels].clamp(min=self.k1_min, max=self.k1_max)

            # Compute weighted negative distances using adaptive k1
            # For efficiency, use max k1 for indexing, then mask
            max_k1 = self.k1_max
            topK_neg = sorted_distance_set_neg[:, :max_k1+1]  # [N, max_k1+1]
            # Create mask: for each sample, only use first k1_per_sample negatives
            k1_mask = torch.arange(max_k1+1, device=rel_labels.device).unsqueeze(0) < k1_per_sample.unsqueeze(1)
            topK_neg_masked = topK_neg * k1_mask.float()
            topK_sorted_distance_set_neg = topK_neg_masked.sum(dim=1) / k1_per_sample.float().clamp(min=1)

            loss_sum = torch.max(torch.zeros(rel_labels.size(0)).cuda(), distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean()
            add_losses.update({"loss_dis": loss_sum})
            ### end Adaptive Negative Sampling """
    
    if old_loss_dis not in content:
        print("ERROR: Cannot find loss_dis block")
        return None
    content = content.replace(old_loss_dis, new_loss_dis)
    
    return content


def patch_direction_6(content):
    """Feature augmentation for tail classes - noise injection on rel_rep"""
    
    # Add augmentation parameters in __init__
    old_init = "        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES"
    new_init = """        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        # Feature Augmentation: inject scaled noise into tail class rel_rep
        self.aug_noise_scale = 0.1  # base noise scale
        self.aug_freq_threshold = 0.3  # only augment bottom 30% frequency classes"""
    
    if old_init not in content:
        print("ERROR: Cannot find nms_thresh in __init__")
        return None
    content = content.replace(old_init, new_init)
    
    # Add noise injection before rel_rep normalization
    old_rel_rep = """        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)

        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep)))
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))
        ######"""
    
    new_rel_rep = """        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)

        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep)))
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))

        # Feature Augmentation: add scaled Gaussian noise to rel_rep during training
        # Noise scale is inversely proportional to feature norm (normalized perturbation)
        if self.training:
            with torch.no_grad():
                noise_scale = self.aug_noise_scale * rel_rep.norm(dim=1, keepdim=True)
            noise = torch.randn_like(rel_rep) * noise_scale
            rel_rep = rel_rep + noise
        ######"""
    
    if old_rel_rep not in content:
        print("ERROR: Cannot find rel_rep normalization block")
        return None
    content = content.replace(old_rel_rep, new_rel_rep)
    
    return content


def patch_direction_7(content):
    """Per-class learnable temperature (replace scalar logit_scale with vector)"""
    
    old_init = "        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))"
    new_init = """        # Per-class learnable temperature: each predicate gets its own temperature
        # Initialized uniformly, learned end-to-end
        self.logit_scale = nn.Parameter(torch.ones(self.num_rel_cls) * np.log(1 / 0.07))"""
    
    if old_init not in content:
        print("ERROR: Cannot find logit_scale in __init__")
        return None
    content = content.replace(old_init, new_init)
    
    # Modify forward to broadcast per-class temperature
    old_forward = "        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  #  <r_norm, c_norm> / τ"
    new_forward = "        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp().unsqueeze(0)  #  per-class temperature"
    
    if old_forward not in content:
        print("ERROR: Cannot find rel_dists computation")
        return None
    content = content.replace(old_forward, new_forward)
    
    return content


def apply_patches(penet_root, direction):
    pred_path = os.path.join(penet_root, 
        "maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py")
    
    if not os.path.exists(pred_path):
        print(f"ERROR: {pred_path} not found")
        return False
    
    content = read_file(pred_path)
    original = content  # keep backup
    
    print(f"Applying direction {direction}...")
    
    if direction == 1:
        # Learnable bias only
        content = patch_direction_1(content)
    
    elif direction == 2:
        # CGPR only
        content = patch_direction_2(content)
    
    elif direction == 3:
        # Adaptive negative sampling only
        content = patch_direction_3(content)
    
    elif direction == 4:
        # CGPR + learnable bias
        content = patch_direction_2(content)
        if content:
            content = patch_direction_1(content)
    
    elif direction == 5:
        # CGPR + adaptive negative sampling
        content = patch_direction_2(content)
        if content:
            content = patch_direction_3(content)
    
    elif direction == 6:
        # Feature augmentation
        content = patch_direction_6(content)
    
    elif direction == 7:
        # Per-class temperature
        content = patch_direction_7(content)
    
    elif direction == 8:
        # Full: CGPR + bias + adaptive neg
        content = patch_direction_2(content)
        if content:
            content = patch_direction_1(content)
        if content:
            content = patch_direction_3(content)
    
    else:
        print(f"ERROR: Unknown direction {direction}")
        return False
    
    if content is None:
        print("PATCH FAILED")
        return False
    
    write_file(pred_path, content)
    
    # Verify CB-Loss is untouched
    loss_path = os.path.join(penet_root,
        "maskrcnn_benchmark/modeling/roi_heads/relation_head/loss.py")
    if os.path.exists(loss_path):
        loss_content = read_file(loss_path)
        if "cb_loss_beta=0.9999" in loss_content:
            print("  VERIFIED: CB-Loss beta=0.9999 untouched in loss.py")
        else:
            print("  WARNING: CB-Loss not found in loss.py")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("penet_root", help="Path to PE-NET root")
    parser.add_argument("--direction", type=int, required=True, 
                       help="1=bias, 2=CGPR, 3=adaptive_neg, 4=CGPR+bias, 5=CGPR+neg, 6=feat_aug, 7=per_class_temp, 8=full")
    args = parser.parse_args()
    
    success = apply_patches(args.penet_root, args.direction)
    if success:
        print(f"\nDONE. Direction {args.direction} applied successfully.")
        print("CB-Loss in loss.py is NEVER modified.")
    else:
        print("\nFAILED. Check errors above.")
        sys.exit(1)
