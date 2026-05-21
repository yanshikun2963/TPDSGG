#!/bin/bash
# ============================================================
# patch_cuda_compat.sh
# 
# Patches ALL maskrcnn_benchmark source files for PyTorch 2.7+
# Covers: .cu (CUDA kernels), .h (headers), .cpp (CPU), .py (Python)
#
# Usage: cd /root/autodl-tmp/penet-main && bash scripts/patch_cuda_compat.sh
# ============================================================

set -e
echo "=== Patching maskrcnn_benchmark for PyTorch 2.7.0 + CUDA 12.8 ==="

# ============================================================
# 1. Fix .cu files: replace THC headers
# ============================================================
echo "[1/5] Patching .cu files (THC headers)..."
for f in maskrcnn_benchmark/csrc/cuda/*.cu; do
    sed -i 's|#include <THC/THC.h>|#include <ATen/ATen.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAGuard.h>|g' "$f"
    sed -i 's|#include <THC/THCAtomics.cuh>|#include <ATen/cuda/Atomic.cuh>|g' "$f"
    sed -i 's|#include <THC/THCDeviceUtils.cuh>||g' "$f"
    sed -i 's|THCudaCheck(|C10_CUDA_CHECK(|g' "$f"
    echo "  ✓ $f"
done

# ============================================================
# 2. Fix .h files: replace .type().is_cuda() with .is_cuda()
# ============================================================
echo "[2/5] Patching .h files (.type().is_cuda deprecation)..."
for f in maskrcnn_benchmark/csrc/*.h; do
    # input.type().is_cuda() → input.is_cuda()
    sed -i 's|\.type()\.is_cuda()|.is_cuda()|g' "$f"
    echo "  ✓ $f"
done

# ============================================================
# 3. Fix .cpp files: AT_CHECK → TORCH_CHECK (if any)
# ============================================================
echo "[3/5] Patching .cpp files..."
for f in maskrcnn_benchmark/csrc/cpu/*.cpp; do
    if grep -q "AT_CHECK" "$f" 2>/dev/null; then
        sed -i 's|AT_CHECK|TORCH_CHECK|g' "$f"
        echo "  ✓ $f (AT_CHECK→TORCH_CHECK)"
    fi
done

# ============================================================
# 4. Fix deform_pool_kernel_cuda.cu: add cuda_fp16.h for atomicAdd half
# ============================================================
echo "[4/5] Patching deform_pool_kernel_cuda.cu (atomicAdd half)..."
DEFORM_POOL="maskrcnn_benchmark/csrc/cuda/deform_pool_kernel_cuda.cu"
if [ -f "$DEFORM_POOL" ]; then
    if ! grep -q "cuda_fp16.h" "$DEFORM_POOL"; then
        sed -i '1i #include <cuda_fp16.h>' "$DEFORM_POOL"
        echo "  ✓ Added cuda_fp16.h"
    else
        echo "  - Already patched"
    fi
fi

# ============================================================
# 5. Fix Python: torch._six removed in PyTorch 1.9+
# ============================================================
echo "[5/5] Patching Python files (torch._six removal)..."
IMPORTS_FILE="maskrcnn_benchmark/utils/imports.py"
if grep -q "torch._six" "$IMPORTS_FILE" 2>/dev/null; then
    cat > "$IMPORTS_FILE" << 'PYEOF'
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Patched for PyTorch 2.7+ compatibility (torch._six removed)
import importlib
import importlib.util
import sys


def import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module
PYEOF
    echo "  ✓ $IMPORTS_FILE"
else
    echo "  - Already patched"
fi

echo ""
echo "=== All patches applied ==="
echo ""
echo "Now run:"
echo "  rm -rf build/"
echo "  python setup.py build develop"
