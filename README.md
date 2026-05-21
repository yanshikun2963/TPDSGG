# Tail Prototype Drift Correction for Long-Tailed Scene Graph Generation

Official implementation of **"Tail Prototype Drift: Empirical Calibration and Confusion-Guided Repulsion for Long-Tailed Scene Graph Generation"** (ICONIP 2026).

## Overview

Prototype-based scene graph generation (SGG) methods represent each predicate class with a learned semantic prototype. Under long-tailed predicate distributions, we observe that tail-class prototypes drift away from the empirical centers of their relation features — a phenomenon we term *Tail Prototype Drift* (TPD). This geometric misalignment degrades tail-class recognition.

We propose two post-training correction losses that operate on a frozen PE-Net checkpoint:

- **TEPA** (Tail-aware Empirical Prototype Alignment): aligns tail prototypes with their EMA-estimated empirical relation centers.
- **CPTR** (Confusion-Pair Targeted Repulsion): pushes tail prototypes away from their most confusable head counterparts, identified via subject-object distribution overlap (Bhattacharyya coefficient).

Combined with class-balanced relation loss reweighting, the resulting **TPD-Net** substantially improves mean Recall while maintaining competitive Recall.

## Main Results

### Visual Genome (VG150)

| Model | Task | R@50 | mR@50 | F@50 |
|-------|------|------|-------|------|
| PE-Net (baseline) | PredCls | 64.90 | 31.50 | 42.41 |
| **TPD-Net** | **PredCls** | **58.84** | **40.94** | **48.28** |
| **TPD-Net** | **SGCls** | **36.64** | **21.86** | **27.38** |
| **TPD-Net** | **SGDet** | **27.75** | **15.68** | **20.04** |

### GQA-200

| Model | Task | R@50 | mR@50 | F@50 |
|-------|------|------|-------|------|
| PE-Net (baseline) | PredCls | 54.30 | 26.20 | 35.40 |
| **TPD-Net** | **PredCls** | **52.46** | **37.49** | **43.73** |

## Installation

See [INSTALL.md](INSTALL.md) for detailed instructions.

**Quick start:**

```bash
git clone https://github.com/yanshikun2963/TEPA-CPTR.git
cd TEPA-CPTR

# Create conda environment
conda create -n tpdnet python=3.8 -y
conda activate tpdnet

# Install dependencies
pip install -r requirements.txt
python setup.py build develop
```

## Dataset Preparation

See [DATASET.md](DATASET.md) for VG-150 and GQA-200 dataset setup instructions.

## Usage

### Step 1: Obtain PE-Net Baseline Checkpoint

Train PE-Net from scratch or download a pretrained checkpoint:

```bash
bash scripts/train.sh
```

Place the trained checkpoint at `checkpoints/PENET_baseline/model_final.pth`.

### Step 2: Build Confusion Pair Tables

```bash
python datasets/prepare_freq_table.py
python datasets/prepare_confusion_pairs.py
```

This produces `datasets/vg/soo_freq.npy` and `datasets/vg/cptr_pairs_filtered.npy`.

### Step 3: Train TPD-Net

```bash
bash scripts/train_tpdnet.sh
```

### Step 4: Evaluate

```bash
bash scripts/test_tpdnet.sh
```

## Pretrained Models

| Model | Task | Checkpoint | Training Log |
|-------|------|------------|--------------|
| PE-Net baseline | PredCls | [Google Drive](https://drive.google.com/file/d/1rjsLs3N33iiOB5xYO7zetNhR7ebi385W/view?usp=share_link) | [Log](https://drive.google.com/file/d/1YK0dLWVkmfWQjpreBdeWi4H0XyV61XMl/view?usp=share_link) |
| TPD-Net | PredCls | Coming soon | Coming soon |

## Method Summary

Given a trained prototype-based SGG model, TPD-Net resumes training with two correction losses:

**TEPA** maintains an exponential moving average of per-class relation feature centers. For each tail predicate (frequency below median), it minimises the cosine distance between the learned prototype and its empirical center. The alignment is restricted to tail classes to preserve head-class decision boundaries.

**CPTR** identifies confusable (tail, head) predicate pairs by measuring subject-object distribution overlap via the Bhattacharyya coefficient. For the top-K most overlapping head predicates per tail class, a smoothed-hinge repulsion loss pushes the tail prototype away from the (stop-gradient) head prototype.

Both losses are activated after a warmup period and introduce no additional inference cost.

## Citation

```bibtex
@inproceedings{yan2026tpdnet,
  title={Tail Prototype Drift: Empirical Calibration and Confusion-Guided Repulsion for Long-Tailed Scene Graph Generation},
  author={Yan, Shikun and others},
  booktitle={International Conference on Neural Information Processing (ICONIP)},
  year={2026}
}
```

## Acknowledgement

This codebase is built upon [PE-Net](https://github.com/VL-Group/PENET) (Zheng et al., CVPR 2023) and [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) (Tang et al.).

## License

This project is released under the [MIT License](LICENSE).
