#!/usr/bin/env python3
import argparse, json, os, sys, torch, torch.nn as nn, torch.nn.functional as F, numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--glove-dir", required=True)
    args = parser.parse_args()
    dict_path = os.path.join(args.glove_dir, 'VG-SGG-dicts-with-attri.json')
    with open(dict_path, 'r') as f:
        vg_dict = json.load(f)
    idx_to_pred = vg_dict['idx_to_predicate']
    pred_names = ['__background__'] + [idx_to_pred[str(i)] for i in range(1, 51)]
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    prefix = "roi_heads.relation.predictor."
    rel_embed_weight = state_dict[prefix + "rel_embed.weight"]
    num_classes, embed_dim = rel_embed_weight.shape
    print(f"rel_embed: [{num_classes}, {embed_dim}]")
    post_emb_weight = state_dict[prefix + "post_emb.weight"]
    mlp_dim = post_emb_weight.shape[0] // 2
    print(f"mlp_dim: {mlp_dim}")
    ph_layer0_weight = state_dict[prefix + "project_head.layers.0.weight"]
    ph_layer1_weight = state_dict[prefix + "project_head.layers.1.weight"]
    ph_hidden = ph_layer0_weight.shape[0]
    ph_output = ph_layer1_weight.shape[0]
    print(f"project_head: [{mlp_dim}] -> [{ph_hidden}] -> [{ph_output}]")
    W_pred = MLP(embed_dim, mlp_dim // 2, mlp_dim, 2)
    W_pred.layers[0].weight.data.copy_(state_dict[prefix + "W_pred.layers.0.weight"])
    W_pred.layers[0].bias.data.copy_(state_dict[prefix + "W_pred.layers.0.bias"])
    W_pred.layers[1].weight.data.copy_(state_dict[prefix + "W_pred.layers.1.weight"])
    W_pred.layers[1].bias.data.copy_(state_dict[prefix + "W_pred.layers.1.bias"])
    project_head = MLP(mlp_dim, ph_hidden, ph_output, 2)
    project_head.layers[0].weight.data.copy_(ph_layer0_weight)
    project_head.layers[0].bias.data.copy_(state_dict[prefix + "project_head.layers.0.bias"])
    project_head.layers[1].weight.data.copy_(ph_layer1_weight)
    project_head.layers[1].bias.data.copy_(state_dict[prefix + "project_head.layers.1.bias"])
    with torch.no_grad():
        predicate_proto = W_pred(rel_embed_weight)
        predicate_proto = project_head(F.relu(predicate_proto))
        proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)
        sim_matrix = proto_norm @ proto_norm.t()
    sim_np = sim_matrix.numpy()
    np.fill_diagonal(sim_np, -1)
    pairs = []
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            pairs.append((sim_np[i, j], i, j))
    pairs.sort(reverse=True)
    print("\n" + "=" * 70)
    print("TOP-20 MOST SIMILAR PROTOTYPE PAIRS")
    print("=" * 70)
    print(f"{'Rank':<6} {'Pred_i':<18} {'Pred_j':<18} {'Cosine Sim':<12}")
    print("-" * 70)
    for rank, (sim, i, j) in enumerate(pairs[:20], 1):
        print(f"{rank:<6} {pred_names[i]:<18} {pred_names[j]:<18} {sim:<12.4f}")
    top20_sims = [p[0] for p in pairs[:20]]
    median_sim = np.median(top20_sims)
    suggested_margin = round(median_sim - 0.1, 2)
    suggested_margin = max(suggested_margin, 0.1)
    print("\n" + "=" * 70)
    print("RECOMMENDED CAPR MARGIN")
    print("=" * 70)
    print(f"Top-20 median similarity: {median_sim:.4f}")
    print(f"Suggested margin: {suggested_margin}")
    if median_sim < 0.3:
        print("!! WARNING: median < 0.3, prototype confusion may NOT be the bottleneck.")
    elif median_sim < 0.5:
        print("Moderate confusion. CAPR may help.")
    else:
        print("Strong confusion. CAPR should be effective.")
    mask = 1.0 - np.eye(num_classes)
    sim_fixed = sim_np.copy()
    np.fill_diagonal(sim_fixed, 0)
    repulsion = np.maximum(sim_fixed * mask - suggested_margin, 0)
    num_active = int(np.sum(repulsion > 0))
    capr_loss_init = repulsion.sum() / max(num_active, 1)
    print(f"\nInitial CAPR loss (m={suggested_margin}): {capr_loss_init:.6f}")
    print(f"Active pairs above margin: {num_active}")
    if capr_loss_init > 0:
        print(f"Suggested lambda range: [{0.1/capr_loss_init:.1f}, {1.0/capr_loss_init:.1f}]")
    stuck = {"above","attached to","part of","wears","belonging to","near","with"}
    print("\n" + "=" * 70)
    print("STUCK PREDICATES IN TOP PAIRS")
    print("=" * 70)
    for sv, i, j in pairs[:40]:
        if pred_names[i] in stuck or pred_names[j] in stuck:
            print(f"  {pred_names[i]:<18} -- {pred_names[j]:<18}  sim={sv:.4f}")

if __name__ == "__main__":
    main()
