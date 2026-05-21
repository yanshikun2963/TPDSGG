import pandas as pd
import numpy as np

CSV = './drift_analysis_tpdnet/drift.csv'
df = pd.read_csv(CSV)

print(f"Loaded: {CSV}")
print(f"Total classes: {len(df)}")
print(f"Min/Max train_count: {df['train_count'].min()}, {df['train_count'].max()}")
print()

print(f"{'N_min':>6} | {'classes':>7} | {'HEAD#':>5} | {'TAIL#':>5} | {'HEAD ang':>9} | {'TAIL ang':>9} | {'gap':>7}")
print("-"*70)
for th in [1, 5, 10, 20, 30, 50, 100, 200, 500, 1000]:
    d = df[df['train_count'] >= th]
    if len(d) < 4:
        continue
    med = d['train_count'].median()
    head = d[d['train_count'] > med]
    tail = d[d['train_count'] <= med]
    if len(head) == 0 or len(tail) == 0:
        continue
    h_ang = head['angle_deg'].mean()
    t_ang = tail['angle_deg'].mean()
    print(f"{th:>6} | {len(d):>7} | {len(head):>5} | {len(tail):>5} | "
          f"{h_ang:>9.2f} | {t_ang:>9.2f} | {t_ang-h_ang:>+7.2f}")

df['log_freq'] = np.log10(df['train_count'] + 1)
pearson = df[['log_freq', 'angle_deg']].corr(method='pearson').iloc[0,1]
spearman = df[['log_freq', 'angle_deg']].corr(method='spearman').iloc[0,1]
print(f"\nPearson  corr(log_freq, angle) = {pearson:+.4f}")
print(f"Spearman corr(log_freq, angle) = {spearman:+.4f}")

print("\nTop 10 largest angles (most drifted in TPD-Net):")
print(df.sort_values('angle_deg', ascending=False)[['class_id','predicate','train_count','angle_deg']].head(10).to_string(index=False))

print("\nBottom 10 smallest angles (most aligned in TPD-Net):")
print(df.sort_values('angle_deg', ascending=True)[['class_id','predicate','train_count','angle_deg']].head(10).to_string(index=False))
