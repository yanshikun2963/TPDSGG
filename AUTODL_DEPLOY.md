# CAPE-SGG AutoDL部署指南 (RTX PRO 6000 + CUDA 12.8)

你的显卡: RTX PRO 6000 (96GB)
代码位置: `/root/autodl-tmp/penet-main`
虚拟环境名: `penet`

---

## 第〇步：确认你的GPU型号

先SSH进AutoDL，运行一条命令确认GPU版本：

```bash
nvidia-smi
```

看输出第一行的GPU名字：
- 如果显示 **RTX PRO 6000 Ada** → 你是Ada架构 (sm_89)，PyTorch 2.1+ cu121 即可
- 如果显示 **RTX PRO 6000 Blackwell** → 你是Blackwell架构 (sm_100/120)，必须 PyTorch 2.6+ cu128

同时记下CUDA Version那一栏（如12.8、12.6等）。

---

## 第一步：创建conda虚拟环境

```bash
# 创建Python 3.8环境（PE-NET框架要求）
conda create -n penet python=3.8 -y
conda activate penet

# 确认
python --version   # 应该是3.8.x
which python       # 应该在conda环境里
```

---

## 第二步：安装PyTorch（根据你的GPU型号选一个）

### 方案A：如果是 Ada架构 (sm_89) + CUDA 12.x

```bash
conda activate penet

# PyTorch 2.1.2 + CUDA 12.1（稳定且兼容Ada sm_89）
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 验证
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

### 方案B：如果是 Blackwell架构 (sm_100/120) + CUDA 12.8

```bash
conda activate penet

# 必须用CUDA 12.8的PyTorch（2.6+）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 验证
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

> 如果上面报错 `sm_1xx is not compatible`，说明需要更新到最新nightly：
> ```bash
> pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
> ```

---

## 第三步：安装项目依赖

```bash
conda activate penet
cd /root/autodl-tmp/penet-main

# 基础依赖
pip install ninja yacs cython matplotlib tqdm opencv-python overrides scipy h5py pycocotools

# CAPE-SGG额外依赖
pip install open-clip-torch
```

---

## 第四步：编译项目C++扩展（最关键的一步）

PE-NET的maskrcnn_benchmark有CUDA C++扩展，需要针对你的GPU架构编译。
这一步最容易出错，根据报错选对应方案。

### 4.1 先尝试直接编译

```bash
conda activate penet
cd /root/autodl-tmp/penet-main

# 清理旧编译
rm -rf build/ *.egg-info maskrcnn_benchmark/*.so

# 编译
python setup.py build develop
```

### 4.2 如果报错：`THC/THC.h: No such file or directory`

这是PyTorch 1.11+移除了THC头文件导致的。需要打补丁：

```bash
cd /root/autodl-tmp/penet-main

# 替换所有CUDA源文件中的THC头文件引用
find maskrcnn_benchmark/csrc -name "*.cu" -o -name "*.cuh" | xargs sed -i 's|#include <THC/THC.h>|#include <ATen/ATen.h>\n#include <ATen/cuda/CUDAContext.h>|g'

# 替换THCudaCheck为AT_CUDA_CHECK
find maskrcnn_benchmark/csrc -name "*.cu" -o -name "*.cuh" | xargs sed -i 's/THCudaCheck/AT_CUDA_CHECK/g'

# 替换THCState相关
find maskrcnn_benchmark/csrc -name "*.cu" -o -name "*.cuh" | xargs sed -i 's/THCState \*state = at::globalContext().lazyInitCUDA();//g'
find maskrcnn_benchmark/csrc -name "*.cu" -o -name "*.cuh" | xargs sed -i 's/state,//g'

# 重新编译
rm -rf build/
python setup.py build develop
```

### 4.3 如果报错：`atomicAdd` 的 `c10::Half` 重载不存在

这是高版本CUDA的半精度兼容问题。修复方法：

```bash
cd /root/autodl-tmp/penet-main

# 找到所有用到atomicAdd的CUDA文件
grep -rn "atomicAdd" maskrcnn_benchmark/csrc/cuda/

# 在有问题的文件（通常是deform_pool_kernel_cuda.cu和deform_conv_kernel_cuda.cu）
# 的头部添加以下代码：

cat > /tmp/atomic_fix.h << 'EOF'
#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
// atomicAdd for half is only natively supported on sm_70+
__device__ __forceinline__ void atomicAdd(c10::Half* address, c10::Half val) {
    unsigned int* address_as_ui = (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    do {
        assumed = old;
        unsigned short hsum = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
        hsum += __half_as_ushort(__float2half(__half2float(__ushort_as_half(hsum)) + static_cast<float>(val)));
        old = (size_t)address & 2 ? (old & 0xffff) | (hsum << 16) : (old & 0xffff0000) | hsum;
        old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
}
#endif
EOF

# 在有问题的.cu文件开头插入
for f in maskrcnn_benchmark/csrc/cuda/deform_pool_kernel_cuda.cu maskrcnn_benchmark/csrc/cuda/deform_conv_kernel_cuda.cu; do
    if [ -f "$f" ]; then
        cat /tmp/atomic_fix.h <(cat "$f") > "${f}.tmp" && mv "${f}.tmp" "$f"
        echo "Patched: $f"
    fi
done

# 重新编译
rm -rf build/
python setup.py build develop
```

### 4.4 如果报错：`sm_xxx is not supported`

在setup.py中指定你的GPU架构：

```bash
cd /root/autodl-tmp/penet-main

# 查看你的GPU计算能力
python -c "import torch; print(torch.cuda.get_device_capability())"
# 输出例如 (8, 9) 代表 sm_89

# 设置环境变量后编译（根据你的GPU改数字）
# Ada RTX PRO 6000:
TORCH_CUDA_ARCH_LIST="8.9" python setup.py build develop

# Blackwell RTX PRO 6000:
TORCH_CUDA_ARCH_LIST="10.0" python setup.py build develop

# 如果不确定，用带PTX的通用版本（稍慢但兼容）：
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9+PTX" python setup.py build develop
```

### 4.5 编译成功验证

```bash
python -c "
from maskrcnn_benchmark.layers import nms
print('C++ extensions compiled: OK')
from maskrcnn_benchmark.modeling.roi_heads.relation_head.capm_module import CAPM
from maskrcnn_benchmark.modeling.roi_heads.relation_head.fasa_module import FASA
print('CAPM: OK')
print('FASA: OK')
print('All modules ready!')
"
```

---

## 第五步：下载数据集

在AutoDL上用 `autodl-tmp` 目录存大文件（不占系统盘空间）。

### 5.1 创建目录结构

```bash
mkdir -p /root/autodl-tmp/penet-main/datasets/vg/VG_100K
mkdir -p /root/autodl-tmp/penet-main/checkpoints/pretrained_faster_rcnn
```

### 5.2 下载VG图片（约14GB）

```bash
cd /root/autodl-tmp/penet-main/datasets/vg/

# 开AutoDL学术加速（如果可用）
# source /etc/network_turbo

wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip -O images_part1.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip -O images_part2.zip

unzip images_part1.zip -d VG_100K/
unzip images_part2.zip -d VG_100K/
mv VG_100K/VG_100K_2/* VG_100K/ 2>/dev/null
rmdir VG_100K/VG_100K_2 2>/dev/null

# 验证
ls VG_100K/*.jpg | wc -l
# 应该有约108,077张
```

> **如果wget太慢**：在本地电脑下载后用 `scp` 传到AutoDL：
> ```bash
> scp images_part1.zip root@<你的AutoDL地址>:/root/autodl-tmp/penet-main/datasets/vg/
> ```

### 5.3 下载标注文件

标注文件有三个，需要从OneDrive或镜像站下载：

```bash
cd /root/autodl-tmp/penet-main/datasets/vg/

# 方式1：OneDrive原始链接（可能需要手动下载再上传）
# https://1drv.ms/u/s!AmRLLNf6bzcir8xf9oC3eNWlVMTRDw?e=63t7Ed
# 下载后应得到：VG-SGG-with-attri.h5, VG-SGG-dicts-with-attri.json, image_data.json

# 方式2：如果有国内镜像或百度网盘的，直接用那个

# 验证
ls -lh VG-SGG-with-attri.h5           # 约1.8GB
ls -lh VG-SGG-dicts-with-attri.json   # 约1MB
ls -lh image_data.json                # 约11MB
```

### 5.4 下载GloVe词向量

```bash
cd /root/autodl-tmp/penet-main/datasets/vg/

wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
# 保留 glove.6B.200d.txt 和 glove.6B.300d.txt，其他可以删

# 验证
ls -lh glove.6B.200d.txt  # 约252MB
ls -lh glove.6B.300d.txt  # 约376MB
```

### 5.5 下载预训练Faster R-CNN检测器

```bash
cd /root/autodl-tmp/penet-main/checkpoints/pretrained_faster_rcnn/

# 来自Scene-Graph-Benchmark提供的预训练权重
# OneDrive: https://1drv.ms/u/s!AmRLLNf6bzcir8xemVHbqPBrvjjtQg?e=hAhYCw
# 下载后放到此目录，命名为 model_final.pth

ls -lh model_final.pth  # 约485MB
```

### 5.6 一键验证所有数据

```bash
conda activate penet
cd /root/autodl-tmp/penet-main

python -c "
import os, h5py, json
base = 'datasets/vg'
checks = [
    ('VG图片目录',   os.path.isdir(f'{base}/VG_100K')),
    ('标注H5文件',   os.path.isfile(f'{base}/VG-SGG-with-attri.h5')),
    ('类名字典',     os.path.isfile(f'{base}/VG-SGG-dicts-with-attri.json')),
    ('图片元信息',   os.path.isfile(f'{base}/image_data.json')),
    ('GloVe 200d',  os.path.isfile(f'{base}/glove.6B.200d.txt')),
    ('GloVe 300d',  os.path.isfile(f'{base}/glove.6B.300d.txt')),
    ('检测器权重',   os.path.isfile('checkpoints/pretrained_faster_rcnn/model_final.pth')),
]
all_ok = True
for name, ok in checks:
    print(f\"  {'✓' if ok else '✗ MISSING'} {name}\")
    if not ok: all_ok = False

if all_ok:
    d = json.load(open(f'{base}/VG-SGG-dicts-with-attri.json'))
    print(f'  Object classes: {len(d[\"idx_to_label\"])}')
    print(f'  Predicate classes: {len(d[\"idx_to_predicate\"])}')
    h = h5py.File(f'{base}/VG-SGG-with-attri.h5', 'r')
    print(f'  Total images in h5: {h[\"split\"].shape[0]}')
    h.close()
    print('\n  === ALL CHECKS PASSED ===')
else:
    print('\n  === SOME FILES MISSING ===')
"
```

---

## 第六步：预计算CLIP嵌入

```bash
conda activate penet
cd /root/autodl-tmp/penet-main

python tools/clip_precompute.py \
    --output_dir ./datasets/vg/ \
    --dict_file ./datasets/vg/VG-SGG-dicts-with-attri.json

# 验证
python -c "
import torch
d = torch.load('datasets/vg/clip_embeddings.pt')
print(f'Model: {d[\"model_name\"]}')
print(f'Embed dim: {d[\"embed_dim\"]}')
print(f'Object: {d[\"obj_embeddings\"].shape}')
print(f'Predicate: {d[\"pred_embeddings\"].shape}')
print('OK!')
"
```

---

## 第七步：开始训练

```bash
conda activate penet
cd /root/autodl-tmp/penet-main
export PYTHONPATH=$(pwd):$PYTHONPATH

# PredCls任务（最快验证方法是否正确）
bash scripts/train_cape_sgg.sh predcls 0

# 查看日志
tail -f output/cape_sgg_predcls/log.txt
```

正常启动应看到：
```
[CAPE-SGG] CLIP semantic bridge: 768d → 300d, blended with GloVe
[CAPE-SGG] APT module loaded successfully
[CAPE-SGG] CAPM module initialized
[CAPE-SGG] Predicate freq from fg_matrix: min=..., max=..., median=...
[CAPE-SGG] FASA module initialized
[CAPE-SGG] CAPEPrototypeEmbeddingNetwork initialized (mode=predcls)
```

### 多GPU并行（如果有多张卡）

```bash
bash scripts/train_cape_sgg.sh predcls 0 &
bash scripts/train_cape_sgg.sh sgcls 1 &
bash scripts/train_cape_sgg.sh sgdet 2 &
```

---

## 常见问题

### Q: `No module named 'maskrcnn_benchmark'`
```bash
cd /root/autodl-tmp/penet-main && python setup.py build develop
```

### Q: `RuntimeError: CUDA out of memory`
编辑 `scripts/train_cape_sgg.sh`，把 `SOLVER.IMS_PER_BATCH 12` 改成 `8` 或 `6`。

### Q: 编译报一堆看不懂的CUDA错误
重点看第一个错误。通常是上面4.2~4.4三种情况之一。先确定是THC头文件问题、atomicAdd问题，还是架构不支持问题，对号入座修复。

### Q: 训练中途断了
直接重新跑同样的命令，框架会自动从最近的checkpoint恢复。

### Q: `open_clip` 下载模型很慢
可以手动下载模型权重放到 `~/.cache/huggingface/hub/` 下，或者设置HF镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
