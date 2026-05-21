"""Construct confusion-pair tables for CPTR via (subject, object) distribution overlap.

For each tail predicate ``t``, the Bhattacharyya coefficient

    BC(t, h) = sum_{s, o} sqrt( P^t(s, o) * P^h(s, o) )

is computed against every head predicate ``h`` and the top-K most overlapping
heads are retained. Two pair tables are produced:

  datasets/vg/cptr_pairs.npy
      Each tail is paired with its top-K heads chosen from the full set of head
      predicates.

  datasets/vg/cptr_pairs_filtered.npy
      Same as above, but a small set of dominant head predicates whose sample
      proportion exceeds ``EXCLUDE_FREQ_THRESH`` (e.g., ``on``, ``has``,
      ``wearing``, ``of``, ``in``, ``near``) is excluded from the candidate
      pool. These predicates account for a large fraction of all relations and
      tend to dominate Recall, so excluding them from the repulsion target
      reduces interference with the head decision boundary while preserving
      the targeted tail-head correction.

Each pair file has shape ``(P, 3)`` with columns ``[tail_id, head_id, weight]``.
``weight`` is the Bhattacharyya coefficient normalised within each tail's top-K
candidates so that every tail contributes equally to the CPTR loss.

Reads:  datasets/vg/soo_freq.npy  (produced by datasets/prepare_freq_table.py)

Run from the project root:

    python datasets/prepare_confusion_pairs.py
"""
import os
import sys

import numpy as np


FREQ_TABLE = "datasets/vg/soo_freq.npy"
OUT_UNFILTERED = "datasets/vg/cptr_pairs.npy"
OUT_FILTERED = "datasets/vg/cptr_pairs_filtered.npy"

TOP_K = 5
EXCLUDE_FREQ_THRESH = 0.04  # head predicates above this sample proportion are
                            # excluded from the filtered candidate pool


def build_pairs(freq, tail_ids, head_candidate_ids, top_k):
    """For each tail, select the top-K most (s, o)-overlapping head predicates
    from ``head_candidate_ids``. Returns an array of ``[tail, head, weight]``."""
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
            bc = float(np.sum(np.sqrt(pdf_t * pdf_h)))
            overlaps.append((int(h), bc))

        overlaps.sort(key=lambda x: -x[1])
        topk = overlaps[:top_k]
        wsum = sum(w for _, w in topk) + eps
        for h, w in topk:
            pairs.append([float(t), float(h), w / wsum])

    return np.array(pairs, dtype=np.float32)


def main():
    if not os.path.exists(FREQ_TABLE):
        print(f"[error] {FREQ_TABLE} not found.")
        print("        Run datasets/prepare_freq_table.py first.")
        sys.exit(1)

    freq = np.load(FREQ_TABLE)
    print(f"[load] freq_table shape={freq.shape}, total={freq.sum():.0f}")

    pred_freq = freq.sum(axis=(0, 1))
    pred_freq_nobg = pred_freq[1:]
    median = np.median(pred_freq_nobg)
    total_nobg = pred_freq_nobg.sum() + 1e-12
    print(
        f"[info] predicate freq range: {pred_freq_nobg.min():.0f} - {pred_freq_nobg.max():.0f}"
    )
    print(f"[info] median freq: {median:.0f}")

    tail_ids = np.where(pred_freq_nobg < median)[0] + 1
    head_ids = np.where(pred_freq_nobg >= median)[0] + 1
    print(f"[info] tail predicates: {len(tail_ids)}, head predicates: {len(head_ids)}")

    pred_prop = pred_freq / total_nobg
    excluded = [int(h) for h in head_ids if pred_prop[h] > EXCLUDE_FREQ_THRESH]
    head_ids_filtered = np.array([h for h in head_ids if int(h) not in excluded])
    print(f"[info] EXCLUDE_FREQ_THRESH = {EXCLUDE_FREQ_THRESH}")
    print(
        f"[info] excluded dominant head predicates (id, proportion): "
        f"{[(h, round(float(pred_prop[h]), 4)) for h in excluded]}"
    )
    print(f"[info] filtered head pool size: {len(head_ids_filtered)} (was {len(head_ids)})")

    print("\n[build] unfiltered pairs (all heads as candidates) ...")
    pairs_unf = build_pairs(freq, tail_ids, head_ids, TOP_K)
    os.makedirs(os.path.dirname(OUT_UNFILTERED), exist_ok=True)
    np.save(OUT_UNFILTERED, pairs_unf)
    print(f"[done] {len(pairs_unf)} pairs -> {OUT_UNFILTERED}")

    print("\n[build] filtered pairs (dominant heads removed) ...")
    pairs_filt = build_pairs(freq, tail_ids, head_ids_filtered, TOP_K)
    np.save(OUT_FILTERED, pairs_filt)
    print(f"[done] {len(pairs_filt)} pairs -> {OUT_FILTERED}")

    for name, arr in [("unfiltered", pairs_unf), ("filtered", pairs_filt)]:
        print(f"\nSample {name} pairs (first 10):")
        print(f"  {'tail':>4}  {'head':>4}  {'weight':>7}")
        for i in range(min(10, len(arr))):
            print(f"  {int(arr[i, 0]):>4}  {int(arr[i, 1]):>4}  {arr[i, 2]:>7.3f}")


if __name__ == "__main__":
    main()
