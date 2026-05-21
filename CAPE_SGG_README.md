# CAPE-SGG 部署指南

## 环境信息

| 项目 | 值 |
|------|-----|
| 显卡 | RTX PRO 6000 Blackwell 96GB |
| 系统CUDA | 12.8 |
| Python | 3.12 |
| PyTorch | 2.7.0 或 2.8.0（二选一，推荐2.7.0已验证） |
| 虚拟环境名 | penet |
| 代码路径 | `/root/autodl-tmp/penet-main` |

---

## 第一步：创建虚拟环境并安装PyTorch

```bash
conda create -n penet python=3.12 -y
conda activate penet

# PyTorch 2.7.0 + CUDA 12.8
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# 如果想用2.8.0：
# pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

验证：
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); x=torch.randn(2,device='cuda'); print('OK', x.device)"
```

---

## 第二步：安装依赖

```bash
conda activate penet
cd /root/autodl-tmp/penet-main

pip install ninja yacs cython matplotlib tqdm opencv-python overrides scipy h5py pycocotools
pip install open-clip-torch
```

---

## 第三步：打补丁 + 编译

maskrcnn_benchmark的CUDA源码用了PyTorch 1.11就移除的旧头文件，必须先打补丁：

```bash
cd /root/autodl-tmp/penet-main

# 打补丁（自动替换THC头文件、THCudaCheck等）
bash scripts/patch_cuda_compat.sh

# 清理旧编译 + 重新编译
rm -rf build/
python setup.py build develop
```

如果编译报 `atomicAdd` + `c10::Half` 的错，手动在 `maskrcnn_benchmark/csrc/cuda/deform_pool_kernel_cuda.cu` 第一行加 `#include <cuda_fp16.h>`，再重新编译。

验证：
```bash
python -c "
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.roi_heads.relation_head.capm_module import CAPM
print('编译成功')
"
```

---

## 第四步：准备数据集

需要下载4类文件。

### 4.1 VG图片（约14GB）

```bash
mkdir -p /root/autodl-tmp/penet-main/datasets/vg/VG_100K
cd /root/autodl-tmp/penet-main/datasets/vg/

wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

unzip images.zip -d VG_100K/
unzip images2.zip -d VG_100K/
mv VG_100K/VG_100K_2/* VG_100K/ 2>/dev/null
rmdir VG_100K/VG_100K_2 2>/dev/null
```

### 4.2 标注文件（3个文件）

下载链接：https://1drv.ms/u/s!AmRLLNf6bzcir8xf9oC3eNWlVMTRDw?e=63t7Ed

下载后放到 `datasets/vg/` 下：
- `VG-SGG-with-attri.h5`（约1.8GB）
- `VG-SGG-dicts-with-attri.json`
- `image_data.json`

### 4.3 GloVe词向量

```bash
cd /root/autodl-tmp/penet-main/datasets/vg/
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
rm -f glove.6B.50d.txt glove.6B.100d.txt glove.6B.zip
```

### 4.4 预训练Faster R-CNN检测器

下载链接：https://1drv.ms/u/s!AmRLLNf6bzcir8xemVHbqPBrvjjtQg?e=hAhYCw

```bash
mkdir -p /root/autodl-tmp/penet-main/checkpoints/pretrained_faster_rcnn/
# 下载后放到：checkpoints/pretrained_faster_rcnn/model_final.pth
```

### 4.5 验证

```bash
cd /root/autodl-tmp/penet-main
python -c "
import os, json
b = 'datasets/vg'
for name, path in [
    ('VG图片',   f'{b}/VG_100K'),
    ('标注H5',   f'{b}/VG-SGG-with-attri.h5'),
    ('类名字典', f'{b}/VG-SGG-dicts-with-attri.json'),
    ('图片信息', f'{b}/image_data.json'),
    ('GloVe200', f'{b}/glove.6B.200d.txt'),
    ('GloVe300', f'{b}/glove.6B.300d.txt'),
    ('检测器',   'checkpoints/pretrained_faster_rcnn/model_final.pth'),
]:
    ok = os.path.isdir(path) if 'VG_100K' in path else os.path.isfile(path)
    print(f\"  {'✓' if ok else '✗'} {name}\")
"
```

---

## 第五步：生成CLIP嵌入

```bash
cd /root/autodl-tmp/penet-main
conda activate penet

python tools/clip_precompute.py \
    --output_dir ./datasets/vg/ \
    --dict_file ./datasets/vg/VG-SGG-dicts-with-attri.json
```

首次运行会下载OpenCLIP ViT-L/14模型（约1.5GB），之后缓存。

---

## 第六步：训练

```bash
cd /root/autodl-tmp/penet-main
conda activate penet

# PredCls（单卡，约12-15小时）
bash scripts/train_cape_sgg.sh predcls 0

# 多卡并行
bash scripts/train_cape_sgg.sh predcls 0 &
bash scripts/train_cape_sgg.sh sgcls 1 &
bash scripts/train_cape_sgg.sh sgdet 2 &
```

正常启动日志：
```
[CAPE-SGG] CLIP semantic bridge: 768d → 300d, blended with GloVe
[CAPE-SGG] APT module loaded successfully
[CAPE-SGG] CAPM module initialized
[CAPE-SGG] FASA module initialized
[CAPE-SGG] CAPEPrototypeEmbeddingNetwork initialized (mode=predcls)
```

---

## 常见问题

| 报错 | 解决 |
|------|------|
| `THC/THC.h: No such file` | 第三步的补丁没打，重新 `bash scripts/patch_cuda_compat.sh` 后编译 |
| `No module named 'maskrcnn_benchmark'` | 没编译，跑 `python setup.py build develop` |
| `atomicAdd` + `c10::Half` 编译错 | 在 `deform_pool_kernel_cuda.cu` 开头加 `#include <cuda_fp16.h>` |
| `CUDA out of memory` | 脚本里 `IMS_PER_BATCH` 从12改成8 |
| `CAPM WARNING: CLIP not found` | 第五步没跑 |
| 训练中断 | 重跑同样命令，自动恢复 |

---

## 目录结构总览

```
/root/autodl-tmp/penet-main/
├── datasets/vg/
│   ├── VG_100K/                        ← 图片
│   ├── VG-SGG-with-attri.h5           ← 标注
│   ├── VG-SGG-dicts-with-attri.json   ← 类名
│   ├── image_data.json                ← 图片信息
│   ├── glove.6B.{200,300}d.txt        ← 词向量
│   └── clip_embeddings.pt             ← 第五步生成
├── checkpoints/pretrained_faster_rcnn/
│   └── model_final.pth                ← 预训练检测器
├── scripts/
│   ├── patch_cuda_compat.sh           ← CUDA补丁脚本
│   └── train_cape_sgg.sh             ← 训练脚本
├── configs/cape_sgg_predcls.yaml     ← 训练配置
└── tools/clip_precompute.py          ← CLIP预计算
```
