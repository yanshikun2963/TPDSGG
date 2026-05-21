"""
build_freq_table.py: Precompute (subject, object, predicate) joint frequency table
                     from VG-150 training split. Used by ECG component.

Reads:  datasets/vg/VG-SGG-with-attri.h5
Writes: datasets/vg/soo_freq.npy        # shape (151, 151, 51) float32

Run from project root:
    cd /root/autodl-tmp/penet-main
    python3 build_freq_table.py
"""
import os
import sys
import h5py
import numpy as np

PROJECT_DIR = "/root/autodl-tmp/penet-main"
VG_DATA = os.path.join(PROJECT_DIR, "datasets/vg/VG-SGG-with-attri.h5")
OUTPUT = os.path.join(PROJECT_DIR, "datasets/vg/soo_freq.npy")


def main():
    if not os.path.exists(VG_DATA):
        print(f"[ERROR] {VG_DATA} not found.")
        sys.exit(1)

    print(f"[load] {VG_DATA}")
    f = h5py.File(VG_DATA, 'r')

    labels = f['labels'][:].flatten()             # (total_boxes,)
    predicates = f['predicates'][:].flatten()      # (total_rels,)
    relationships = f['relationships'][:]          # (total_rels, 2) local indices
    img_to_first_box = f['img_to_first_box'][:]
    img_to_last_box = f['img_to_last_box'][:]
    img_to_first_rel = f['img_to_first_rel'][:]
    img_to_last_rel = f['img_to_last_rel'][:]
    split = f['split'][:]
    f.close()

    # Standard VG-SGG split convention: 0=train, 2=test
    train_imgs = np.where(split == 0)[0]
    if len(train_imgs) == 0:
        # Fallback: not-test as train
        train_imgs = np.where(split != 2)[0]
    print(f"[info] training images: {len(train_imgs)}")
    print(f"[info] total relationships in file: {len(predicates)}")

    freq = np.zeros((151, 151, 51), dtype=np.float32)
    n_added = 0
    n_skipped = 0

    for img_idx in train_imgs:
        rfirst = img_to_first_rel[img_idx]
        rlast = img_to_last_rel[img_idx]
        if rfirst < 0:
            continue
        box_start = img_to_first_box[img_idx]
        for rel_idx in range(rfirst, rlast + 1):
            
            s_global, o_global = relationships[rel_idx]; s = int(labels[s_global])
            o = int(labels[o_global])
            p = int(predicates[rel_idx])
            if 0 < s < 151 and 0 < o < 151 and 0 < p < 51:
                freq[s, o, p] += 1
                n_added += 1
            else:
                n_skipped += 1

    print(f"[stats] added {n_added}, skipped {n_skipped}")
    print(f"[stats] non-zero cells: {(freq > 0).sum()} / {freq.size} "
          f"({(freq > 0).mean() * 100:.3f}%)")
    print(f"[stats] max freq: {freq.max():.0f}, total: {freq.sum():.0f}")

    # Per-predicate sanity check
    pred_totals = freq.sum(axis=(0, 1))[1:]  # skip bg
    print(f"[stats] predicate freq range: {pred_totals.min():.0f} - {pred_totals.max():.0f}")
    print(f"[stats] predicate freq median: {np.median(pred_totals):.0f}")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    np.save(OUTPUT, freq)
    print(f"[done] saved to {OUTPUT}")
    print(f"[done] file size: {os.path.getsize(OUTPUT) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
