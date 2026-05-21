#!/bin/bash
# Quick check script for all running experiments
# Usage: bash scripts/check_all.sh

echo "============================================"
echo "  CAPE-SGG Experiment Status $(date +%H:%M)"
echo "============================================"
echo ""

echo "=== Checkpoints ==="
for dir in output/*/; do
    [ -d "$dir" ] || continue
    name=$(basename "$dir")
    last_ckpt=$(ls -t "$dir"/model_*.pth 2>/dev/null | head -1)
    if [ -n "$last_ckpt" ]; then
        iter=$(echo "$last_ckpt" | grep -oP '\d+(?=\.pth)')
        pct=$((iter * 100 / 50000))
        echo "  ✅ $name: iter $iter / 50000 (${pct}%)"
    else
        echo "  ⏳ $name: no checkpoint yet"
    fi
done

echo ""
echo "=== Latest Metrics (from validation) ==="
for log in logs/*.log; do
    [ -f "$log" ] || continue
    name=$(basename "$log" .log)
    # Get last validation R@100 and mR@100
    r100=$(grep "R @ 100:" "$log" 2>/dev/null | grep "type=Recall(Main)" | tail -1 | grep -oP 'R @ 100: \K[0-9.]+')
    mr100=$(grep "mR @ 100:" "$log" 2>/dev/null | grep "type=Mean Recall" | tail -1 | grep -oP 'mR @ 100: \K[0-9.]+')
    r50=$(grep "R @ 50:" "$log" 2>/dev/null | grep "type=Recall(Main)" | tail -1 | grep -oP 'R @ 50: \K[0-9.]+')
    mr50=$(grep "mR @ 50:" "$log" 2>/dev/null | grep "type=Mean Recall" | tail -1 | grep -oP 'mR @ 50: \K[0-9.]+')
    if [ -n "$r100" ]; then
        echo "  $name: R@50=$r50 mR@50=$mr50 R@100=$r100 mR@100=$mr100"
    fi
done

echo ""
echo "=== Latest Loss ==="
for log in logs/*.log; do
    [ -f "$log" ] || continue
    name=$(basename "$log" .log)
    last=$(grep "iter:" "$log" 2>/dev/null | tail -1)
    if [ -n "$last" ]; then
        iter=$(echo "$last" | grep -oP 'iter: \K\d+')
        lr=$(echo "$last" | grep -oP 'loss_rel: \K[0-9.]+')
        ld=$(echo "$last" | grep -oP 'loss_dis: \K[0-9.]+')
        echo "  $name: iter=$iter loss_rel=$lr loss_dis=$ld"
    fi
done

echo ""
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"
