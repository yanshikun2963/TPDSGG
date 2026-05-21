import os, math, torch, numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.data import make_data_loader

CFG_FILE = "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"

# TPD-Net final checkpoint
CKPT_PATH = "./checkpoints/M_cb_9995/model_final.pth"

OUTPUT_DIR = "./drift_analysis_tpdnet"
BASELINE_CSV = "./drift_analysis/drift.csv"

N_BATCHES = 5000
NUM_PREDICATES = 50
DEVICE = "cuda"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cfg.merge_from_file(CFG_FILE)
cfg.merge_from_list([
    "MODEL.ROI_RELATION_HEAD.USE_GT_BOX", True,
    "MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL", True,
    "MODEL.ROI_RELATION_HEAD.PREDICTOR", "PrototypeEmbeddingNetwork",
    "GLOVE_DIR", "./datasets/vg/",
    "MODEL.PRETRAINED_DETECTOR_CKPT", "./checkpoints/pretrained_faster_rcnn/model_final.pth",
    "TEST.IMS_PER_BATCH", 1,
    "DTYPE", "float32",
])
cfg.freeze()

device = torch.device(DEVICE)

model = build_detection_model(cfg).to(device)

# Important: train mode is required so that rel_labels are passed into predictor.
model.train()

predictor = model.roi_heads.relation.predictor

# Important: keep dropout deterministic while still using train-mode forward path.
for m in predictor.modules():
    if isinstance(m, torch.nn.Dropout):
        m.eval()

checkpointer = DetectronCheckpointer(cfg, model, save_dir=os.path.dirname(CKPT_PATH))
_ = checkpointer.load(CKPT_PATH)

print("[OK] TPD-Net model loaded (train mode, dropout off)")
print("Checkpoint:", CKPT_PATH)

try:
    loaders = make_data_loader(cfg, mode='train', is_distributed=False)
except TypeError:
    loaders = make_data_loader(cfg, is_train=True, is_distributed=False)

train_loader = loaders[0] if isinstance(loaders, list) else loaders
print("[OK] Train loader built")

# ============================================================
# Prototype extraction
# Must match the successful PE-Net baseline script:
#   prototype = project_head(ReLU(W_pred(rel_embed)))
# ============================================================
with torch.no_grad():
    raw_proto = predictor.W_pred(predictor.rel_embed.weight)
    projected_proto = predictor.project_head(torch.relu(raw_proto))

prototypes = projected_proto[1:].cpu()
print(f"[OK] Projected prototypes: {tuple(prototypes.shape)}")

cap = {"rel_rep_proj": None, "rel_labels": None}

def pre_hook(module, args):
    # PrototypeEmbeddingNetwork.forward:
    # (proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None)
    rl = args[2]
    cap["rel_labels"] = [x.detach() for x in rl] if rl is not None else None

def hook_norm(module, inp, out):
    # Must match prototype space:
    #   rel_rep = project_head(ReLU(norm_rel_rep_output))
    with torch.no_grad():
        projected = predictor.project_head(torch.relu(out))
    cap["rel_rep_proj"] = projected.detach()

predictor.register_forward_pre_hook(pre_hook)
predictor.norm_rel_rep.register_forward_hook(hook_norm)

feat_dim = prototypes.shape[1]
sum_feat = torch.zeros(NUM_PREDICATES, feat_dim, dtype=torch.float64)
count = torch.zeros(NUM_PREDICATES, dtype=torch.long)

loader_iter = iter(train_loader)

with torch.no_grad():
    pbar = tqdm(range(N_BATCHES), desc="r_so collection (VG / TPD-Net)")

    for i in pbar:
        try:
            batch = next(loader_iter)
        except StopIteration:
            break

        cap["rel_rep_proj"] = None
        cap["rel_labels"] = None

        try:
            images, targets, _ = batch
            images = images.to(device)
            targets = [t.to(device) for t in targets]
            _ = model(images, targets)
        except Exception:
            # Same strategy as the successful PE-Net baseline script:
            # some training forward branches may throw after hooks;
            # the hooked outputs are still usable if captured.
            pass

        if cap["rel_rep_proj"] is None or cap["rel_labels"] is None:
            continue

        rep = cap["rel_rep_proj"].cpu()
        labels = torch.cat(cap["rel_labels"]).cpu()

        if rep.shape[0] != labels.shape[0]:
            continue

        for c in range(1, NUM_PREDICATES + 1):
            mask = (labels == c)
            n_c = int(mask.sum().item())
            if n_c > 0:
                sum_feat[c-1] += rep[mask].double().sum(dim=0)
                count[c-1] += n_c

        if (i + 1) % 500 == 0:
            seen = (count > 0).sum().item()
            pbar.set_postfix(
                min_cnt=int(count.min()),
                max_cnt=int(count.max()),
                seen=f"{seen}/50"
            )

print(f"\n[OK] Done. counts: min={count.min().item()}, max={count.max().item()}, sum={count.sum().item()}")

PRED_NAMES = [
    'above','across','against','along','and','at','attached to',
    'behind','belonging to','between','carrying','covered in','covering',
    'eating','flying in','for','from','growing on','hanging from','has',
    'holding','in','in front of','laying on','looking at','lying on',
    'made of','mounted on','near','of','on','on back of','over',
    'painted on','parked on','part of','playing','riding','says',
    'sitting on','standing on','to','under','using','walking in',
    'walking on','watching','wearing','wears','with',
]

mean_feat = (sum_feat / count.unsqueeze(1).clamp(min=1).double()).float()

proto_n = F.normalize(prototypes.float(), dim=1)
center_n = F.normalize(mean_feat, dim=1)

cos_sim = (proto_n * center_n).sum(dim=1).clamp(-1 + 1e-7, 1 - 1e-7)
angle_deg = (torch.acos(cos_sim) * 180.0 / math.pi).numpy()
counts_np = count.numpy()

csv_path = os.path.join(OUTPUT_DIR, "drift.csv")
order = sorted(range(NUM_PREDICATES), key=lambda c: -counts_np[c])

with open(csv_path, "w") as f:
    f.write("rank,class_id,predicate,train_count,angle_deg,cos_sim\n")
    for rk, c in enumerate(order):
        f.write(
            f"{rk+1},{c+1},{PRED_NAMES[c]},{counts_np[c]},"
            f"{angle_deg[c]:.4f},{cos_sim[c].item():.4f}\n"
        )

print(f"[OK] CSV: {csv_path}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sorted_idx = np.argsort(-counts_np)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(range(1, 51), angle_deg[sorted_idx], "o-", color="steelblue")
    axes[0].axvline(x=25, color="gray", ls="--", alpha=0.5, label="median")
    axes[0].set_xlabel("Predicate rank (head->tail)")
    axes[0].set_ylabel("Angle (deg)")
    axes[0].set_title("VG150: TPD-Net after correction")
    axes[0].legend()

    axes[1].scatter(
        np.log10(counts_np[sorted_idx] + 1),
        angle_deg[sorted_idx],
        color="darkorange",
        alpha=0.7
    )
    axes[1].set_xlabel("log10(freq+1)")
    axes[1].set_ylabel("Angle (deg)")
    axes[1].set_title("angle vs log freq")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "drift_plot.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, "drift_plot.png"), dpi=150)
    print("[OK] Plot saved")

except Exception as e:
    print(f"[WARN] plot: {e}")

med = np.median(counts_np)
head_mask = counts_np > med
tail_mask = counts_np <= med

head_angle = angle_deg[head_mask].mean()
tail_angle = angle_deg[tail_mask].mean()
gap = tail_angle - head_angle

print(f"\n{'='*60}\nSUMMARY (VG150 / TPD-Net)\n{'='*60}")
print(f"Median train count: {med:.0f}")
print(f"HEAD ({head_mask.sum()} classes): mean angle = {head_angle:.2f} deg")
print(f"TAIL ({tail_mask.sum()} classes): mean angle = {tail_angle:.2f} deg")
print(f"Δ (TAIL - HEAD)                 = {gap:+.2f} deg")

# ============================================================
# Compare with PE-Net baseline CSV if available
# ============================================================
if os.path.exists(BASELINE_CSV):
    import csv

    rows = []
    with open(BASELINE_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    rows = sorted(rows, key=lambda r: int(r["class_id"]))
    base_counts = np.array([int(float(r["train_count"])) for r in rows])
    base_angles = np.array([float(r["angle_deg"]) for r in rows])

    base_med = np.median(base_counts)
    base_head = base_counts > base_med
    base_tail = base_counts <= base_med

    base_head_angle = base_angles[base_head].mean()
    base_tail_angle = base_angles[base_tail].mean()
    base_gap = base_tail_angle - base_head_angle

    print(f"\n{'='*60}\nDRIFT ALLEVIATION\n{'='*60}")
    print(f"PE-Net baseline HEAD angle = {base_head_angle:.2f} deg")
    print(f"PE-Net baseline TAIL angle = {base_tail_angle:.2f} deg")
    print(f"PE-Net baseline GAP        = {base_gap:+.2f} deg")
    print()
    print(f"TPD-Net HEAD angle         = {head_angle:.2f} deg")
    print(f"TPD-Net TAIL angle         = {tail_angle:.2f} deg")
    print(f"TPD-Net GAP                = {gap:+.2f} deg")
    print()
    print(f"Gap reduction              = {base_gap - gap:+.2f} deg")
    print(f"Tail angle change          = {tail_angle - base_tail_angle:+.2f} deg")
else:
    print(f"\n[WARN] Baseline CSV not found: {BASELINE_CSV}")

