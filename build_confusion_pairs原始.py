"""
build_confusion_pairs.py: Compute (tail, head) confusion pairs via (s, o)
                          distribution overlap (Bhattacharyya coefficient).

Reads:  datasets/vg/soo_freq.npy        # from build_freq_table.py
Writes: datasets/vg/cptr_pairs.npy      # shape (P, 3): [tail_id, head_id, weight]

Each tail predicate gets TOP_K head predicates whose (s, o) distribution
overlaps most heavily with it. These are the most likely confusion targets.

Run from project root:
    cd /root/autodl-tmp/penet-main
    python3 build_confusion_pairs.py
"""
import os
import sys
import numpy as np

PROJECT_DIR = "/root/autodl-tmp/penet-main"
FREQ_TABLE = os.path.join(PROJECT_DIR, "datasets/vg/soo_freq.npy")
OUTPUT = os.path.join(PROJECT_DIR, "datasets/vg/cptr_pairs.npy")
TOP_K = 5


def main():
    if not os.path.exists(FREQ_TABLE):
        print(f"[ERROR] {FREQ_TABLE} not found.")
        print("        Run: python3 build_freq_table.py first.")
        sys.exit(1)

    freq = np.load(FREQ_TABLE)  # (151, 151, 51)
    print(f"[load] freq_table shape={freq.shape}, total={freq.sum():.0f}")

    # Per-predicate totals
    pred_freq = freq.sum(axis=(0, 1))      # (51,)
    pred_freq_nobg = pred_freq[1:]         # (50,) skip background
    median = np.median(pred_freq_nobg)
    print(f"[info] predicate freq range: {pred_freq_nobg.min():.0f} - {pred_freq_nobg.max():.0f}")
    print(f"[info] median freq: {median:.0f}")

    # 50 predicate ids (1..50), classified into tail / head by median
    tail_ids = np.where(pred_freq_nobg < median)[0] + 1
    head_ids = np.where(pred_freq_nobg >= median)[0] + 1
    print(f"[info] tail predicates: {len(tail_ids)}, head predicates: {len(head_ids)}")
    print(f"[info] tail ids: {tail_ids.tolist()}")

    # For each tail, compute Bhattacharyya coefficient with each head
    pairs = []
    eps = 1e-12
    for t in tail_ids:
        soo_t = freq[:, :, t]
        total_t = soo_t.sum()
        if total_t == 0:
            print(f"  [warn] tail {t}: no samples, skipping")
            continue
        pdf_t = soo_t / total_t

        overlaps = []
        for h in head_ids:
            soo_h = freq[:, :, h]
            total_h = soo_h.sum()
            if total_h == 0:
                continue
            pdf_h = soo_h / total_h
            # Bhattacharyya coefficient: sum(sqrt(pdf_t * pdf_h)) in [0, 1]
            bc = float(np.sum(np.sqrt(pdf_t * pdf_h)))
            overlaps.append((int(h), bc))

        # Take top-K most overlapping head predicates
        overlaps.sort(key=lambda x: -x[1])
        topk = overlaps[:TOP_K]
        wsum = sum(w for _, w in topk) + eps
        for h, w in topk:
            # Normalize weights within tail's topk so each tail contributes equally
            pairs.append([float(t), float(h), w / wsum])

    pairs_arr = np.array(pairs, dtype=np.float32)
    print(f"\n[stats] total confused pairs: {len(pairs_arr)} (= {len(tail_ids)} tail × {TOP_K} top-k)")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    np.save(OUTPUT, pairs_arr)
    print(f"[done] saved to {OUTPUT}")

    # Sanity print
    print("\nSample confused pairs (first 15):")
    print(f"  {'tail':>4}  {'head':>4}  {'weight':>7}")
    for i in range(min(15, len(pairs_arr))):
        print(f"  {int(pairs_arr[i, 0]):>4}  {int(pairs_arr[i, 1]):>4}  {pairs_arr[i, 2]:>7.3f}")


if __name__ == "__main__":
    main()
