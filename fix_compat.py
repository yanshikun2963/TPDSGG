#!/usr/bin/env python3
"""
PE-NET 环境兼容性修复脚本
修复问题：
  1. torch._six 在 PyTorch 2.0+ 中已移除
  2. apex.amp 在不安装 apex 时导致 ImportError
  
用法: cd /root/autodl-tmp/penet-main && python fix_compat.py
"""
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))

def fix_file(relpath, old, new, desc):
    path = os.path.join(BASE, relpath)
    if not os.path.exists(path):
        print(f"  [跳过] {relpath} 不存在")
        return
    with open(path, "r") as f:
        content = f.read()
    if old not in content:
        print(f"  [跳过] {relpath} 已修复或不匹配")
        return
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print(f"  [修复] {relpath}: {desc}")


if __name__ == "__main__":
    print("=" * 50)
    print("PE-NET 环境兼容性修复 (PyTorch 2.7 + 无apex)")
    print("=" * 50)
    
    # ===== 1. torch._six =====
    print("\n[1/5] 修复 torch._six ...")
    fix_file(
        "maskrcnn_benchmark/utils/imports.py",
        "if True:",
        "if True:  # torch._six removed in PyTorch 2.0+",
        "移除 torch._six 引用"
    )
    
    # ===== 2. nms.py =====
    print("\n[2/5] 修复 layers/nms.py ...")
    fix_file(
        "maskrcnn_benchmark/layers/nms.py",
        """try:
    from apex import amp
except ImportError:
    pass

# Only valid with fp32 inputs - give AMP the hint
nms = _C.nms""",
        """# apex removed - not needed for PE-NET training
nms = _C.nms""",
        "移除 apex.amp.float_function"
    )
    
    # ===== 3. roi_align.py =====
    print("\n[3/5] 修复 layers/roi_align.py ...")
    fix_file(
        "maskrcnn_benchmark/layers/roi_align.py",
        "try:
    from apex import amp
except ImportError:
    pass",
        "# apex removed - not needed for PE-NET training",
        "移除 apex import"
    )
    fix_file(
        "maskrcnn_benchmark/layers/roi_align.py",
        "    @amp.float_function\n    def forward(self, input, rois):",
        "    def forward(self, input, rois):",
        "移除 @amp.float_function 装饰器"
    )
    
    # ===== 4. roi_pool.py =====
    print("\n[4/5] 修复 layers/roi_pool.py ...")
    fix_file(
        "maskrcnn_benchmark/layers/roi_pool.py",
        "try:
    from apex import amp
except ImportError:
    pass",
        "# apex removed - not needed for PE-NET training",
        "移除 apex import"
    )
    fix_file(
        "maskrcnn_benchmark/layers/roi_pool.py",
        "    @amp.float_function\n    def forward(self, input, rois):",
        "    def forward(self, input, rois):",
        "移除 @amp.float_function 装饰器"
    )
    
    # ===== 5. trainer.py =====
    print("\n[5/5] 修复 engine/trainer.py ...")
    fix_file(
        "maskrcnn_benchmark/engine/trainer.py",
        "try:
    from apex import amp
except ImportError:
    pass",
        "# apex removed - using PyTorch native AMP if needed",
        "移除 apex import"
    )
    
    print("\n" + "=" * 50)
    print("全部修复完成！")
    print("验证: python -c \"from maskrcnn_benchmark.layers import nms; print('OK')\"")
    print("=" * 50)
