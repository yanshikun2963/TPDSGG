"""
build_confusion_pairs.py: Compute (tail, head) confusion pairs via (s, o)
                          distribution overlap (Bhattacharyya coefficient).

Produces TWO pair files so the four CPTR machines can run a 2-D sweep:

  datasets/vg/cptr_pairs.npy           # UNFILTERED  -> used by machine A
       each tail -> top-K heads among ALL head predicates

  datasets/vg/cptr_pairs_filtered.npy  # FILTERED    -> used by machines B/C/D
       each tail -> top-K heads, but the few SUPER-HIGH-FREQUENCY head
       predicates (proportion > EXCLUDE_FREQ_THRESH, i.e. on/has/wearing/
       of/in/near) are removed from the candidate pool. Those predicates
       carry ~78% of all samples, so R@50 is dominated by them; not making
       tail prototypes repel them reduces the indirect (shared-MLP) damage
       to R. Whether this filtering actually helps is exactly what the
       A-vs-B comparison tests.

Each file: shape (P, 3) = [tail_id, head_id, weight].

Reads:  datasets/vg/soo_freq.npy        # from build_freq_table.py
Run from project root:
    cd /root/autodl-tmp/penet-main
    python3 build_confusion_pairs.py
"""
import os
import sys
import numpy as np

PROJECT_DIR = "/root/autodl-tmp/penet-main"
FREQ_TABLE = os.path.join(PROJECT_DIR, "datasets/vg/soo_freq.npy")
OUT_UNFILTERED = os.path.join(PROJECT_DIR, "datasets/vg/cptr_pairs.npy")
OUT_FILTERED = os.path.join(PROJECT_DIR, "datasets/vg/cptr_pairs_filtered.npy")

TOP_K = 5
EXCLUDE_FREQ_THRESH = 0.04   # head predicates with sample-proportion above
                             # this are excluded from the FILTERED pool


def build_pairs(freq, tail_ids, head_candidate_ids, top_k):
    """For each tail, pick the top-K most (s,o)-overlapping heads from the
    given candidate set. Returns (P, 3) array [tail, head, weight]."""
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
        for h in head_candidate_ids:
            soo_h = freq[:, :, h]
            total_h = soo_h.sum()
            if total_h == 0:
                continue
            pdf_h = soo_h / total_h
            # Bhattacharyya coefficient in [0, 1]
            bc = float(np.sum(np.sqrt(pdf_t * pdf_h)))
            overlaps.append((int(h), bc))

        overlaps.sort(key=lambda x: -x[1])
        topk = overlaps[:top_k]
        wsum = sum(w for _, w in topk) + eps
        for h, w in topk:
            # normalize within each tail's top-k so every tail contributes equally
            pairs.append([float(t), float(h), w / wsum])

    return np.array(pairs, dtype=np.float32)


def main():
    if not os.path.exists(FREQ_TABLE):
        print(f"[ERROR] {FREQ_TABLE} not found.")
        print("        Run: python3 build_freq_table.py first.")
        sys.exit(1)

    freq = np.load(FREQ_TABLE)  # (151, 151, 51)
    print(f"[load] freq_table shape={freq.shape}, total={freq.sum():.0f}")

    # Per-predicate totals and tail/head split (median over 50 real predicates)
    pred_freq = freq.sum(axis=(0, 1))      # (51,)
    pred_freq_nobg = pred_freq[1:]         # (50,) skip background
    median = np.median(pred_freq_nobg)
    total_nobg = pred_freq_nobg.sum() + 1e-12
    print(f"[info] predicate freq range: {pred_freq_nobg.min():.0f} - {pred_freq_nobg.max():.0f}")
    print(f"[info] median freq: {median:.0f}")

    tail_ids = np.where(pred_freq_nobg < median)[0] + 1     # class ids 1..50
    head_ids = np.where(pred_freq_nobg >= median)[0] + 1
    print(f"[info] tail predicates: {len(tail_ids)}, head predicates: {len(head_ids)}")

    # Super-high-frequency heads: proportion of all samples > threshold
    pred_prop = pred_freq / total_nobg
    excluded = [int(h) for h in head_ids if pred_prop[h] > EXCLUDE_FREQ_THRESH]
    head_ids_filtered = np.array([h for h in head_ids if int(h) not in excluded])
    print(f"[info] EXCLUDE_FREQ_THRESH={EXCLUDE_FREQ_THRESH}")
    print(f"[info] excluded super-freq head ids (proportion shown): "
          f"{[(h, round(float(pred_prop[h]), 4)) for h in excluded]}")
    print(f"[info] filtered head pool size: {len(head_ids_filtered)} "
          f"(was {len(head_ids)})")

    # --- UNFILTERED pairs (machine A) ---
    print("\n[build] UNFILTERED pairs (all heads as candidates)...")
    pairs_unf = build_pairs(freq, tail_ids, head_ids, TOP_K)
    os.makedirs(os.path.dirname(OUT_UNFILTERED), exist_ok=True)
    np.save(OUT_UNFILTERED, pairs_unf)
    print(f"[done] {len(pairs_unf)} pairs -> {OUT_UNFILTERED}")

    # --- FILTERED pairs (machines B/C/D) ---
    print("\n[build] FILTERED pairs (super-freq heads removed from candidates)...")
    pairs_filt = build_pairs(freq, tail_ids, head_ids_filtered, TOP_K)
    np.save(OUT_FILTERED, pairs_filt)
    print(f"[done] {len(pairs_filt)} pairs -> {OUT_FILTERED}")

    # Sanity print
    for name, arr in [("UNFILTERED", pairs_unf), ("FILTERED", pairs_filt)]:
        print(f"\nSample {name} pairs (first 10):")
        print(f"  {'tail':>4}  {'head':>4}  {'weight':>7}")
        for i in range(min(10, len(arr))):
            print(f"  {int(arr[i, 0]):>4}  {int(arr[i, 1]):>4}  {arr[i, 2]:>7.3f}")


if __name__ == "__main__":
    main()
