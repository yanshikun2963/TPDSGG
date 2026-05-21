#!/usr/bin/env python
"""
Collect results from all experiments and format for paper tables.
Usage: python scripts/collect_results.py
"""
import os
import re
import glob

def parse_log(log_path):
    """Extract best validation metrics from a training log."""
    if not os.path.exists(log_path):
        return None
    
    results = {}
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # Find the LAST validation block (best model is usually last or near-last)
    for i, line in enumerate(lines):
        if 'SGG eval:' in line and 'R @ 20' in line and 'type=Recall(Main)' in line:
            m = re.findall(r'R @ (\d+): ([0-9.]+)', line)
            for k, v in m:
                results[f'R@{k}'] = float(v)
        if 'SGG eval:' in line and 'mR @ 20' in line and 'type=Mean Recall.' in line:
            m = re.findall(r'mR @ (\d+): ([0-9.]+)', line)
            for k, v in m:
                results[f'mR@{k}'] = float(v)
        if 'SGG eval:' in line and 'zR @ 20' in line:
            m = re.findall(r'zR @ (\d+): ([0-9.]+)', line)
            for k, v in m:
                results[f'zR@{k}'] = float(v)
    
    # Compute F@K
    for k in ['50', '100']:
        rk = results.get(f'R@{k}', 0)
        mk = results.get(f'mR@{k}', 0)
        if rk > 0 and mk > 0:
            results[f'F@{k}'] = 2 * rk * mk / (rk + mk)
    
    return results if results else None

def main():
    experiments = {
        # Main experiments
        'CAPE-Full PredCls': 'logs/cape_predcls_full.log',
        'CAPE-Full SGCls': 'logs/cape_sgcls_full.log',
        'CAPE-Full SGDet': 'logs/cape_sgdet_full.log',
        'PE-NET Baseline': 'logs/penet_baseline.log',
        # Ablations
        'A1: +APT only': 'logs/ablation_A1.log',
        'A2: +CLIP Bridge': 'logs/ablation_A2.log',
        'A3: +CAPM (no FASA)': 'logs/ablation_A3.log',
        'A4: +FASA (no CAPM)': 'logs/ablation_A4.log',
        'A7: ContextBias': 'logs/ablation_A7.log',
        'A8: LearnableGate': 'logs/ablation_A8.log',
        # Hyperparameters
        'FASA λ=0.05': 'logs/ablation_fasa005.log',
        'FASA λ=0.2': 'logs/ablation_fasa02.log',
        'FASA λ=0.5': 'logs/ablation_fasa05.log',
        'Heads=2': 'logs/ablation_heads2.log',
        'Heads=8': 'logs/ablation_heads8.log',
    }
    
    # Also try to find logs with glob
    for log_path in sorted(glob.glob('logs/*.log')):
        name = os.path.basename(log_path).replace('.log', '')
        if name not in [os.path.basename(v).replace('.log', '') for v in experiments.values()]:
            experiments[name] = log_path
    
    print("=" * 100)
    print("CAPE-SGG Results Collection")
    print("=" * 100)
    
    # Table 1: Main comparison (all metrics)
    print("\n### Table 1: Comparison with State-of-the-Art (PredCls)\n")
    print(f"{'Method':<25} {'mR@20':>7} {'mR@50':>7} {'mR@100':>7} {'R@20':>7} {'R@50':>7} {'R@100':>7} {'F@50':>7} {'F@100':>7}")
    print("-" * 100)
    
    for name, log_path in experiments.items():
        res = parse_log(log_path)
        if res:
            print(f"{name:<25} {res.get('mR@20',0):>7.4f} {res.get('mR@50',0):>7.4f} {res.get('mR@100',0):>7.4f} "
                  f"{res.get('R@20',0):>7.4f} {res.get('R@50',0):>7.4f} {res.get('R@100',0):>7.4f} "
                  f"{res.get('F@50',0):>7.4f} {res.get('F@100',0):>7.4f}")
        else:
            print(f"{name:<25} {'(no results)':>50}")
    
    # Table 2: Ablation (simplified)
    print("\n\n### Table 2: Ablation Study (PredCls)\n")
    print(f"{'ID':<25} {'mR@50':>7} {'R@50':>7} {'F@50':>7} {'Δ mR@50':>8}")
    print("-" * 60)
    
    base_mr50 = None
    for name, log_path in experiments.items():
        res = parse_log(log_path)
        if res and res.get('mR@50', 0) > 0:
            mr50 = res['mR@50']
            r50 = res.get('R@50', 0)
            f50 = res.get('F@50', 0)
            if base_mr50 is None:
                base_mr50 = mr50
            delta = mr50 - (base_mr50 if base_mr50 else mr50)
            print(f"{name:<25} {mr50:>7.4f} {r50:>7.4f} {f50:>7.4f} {delta:>+8.4f}")
    
    print("\n" + "=" * 100)

if __name__ == '__main__':
    main()
