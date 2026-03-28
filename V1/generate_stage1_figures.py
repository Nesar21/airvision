import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'

print("="*80)
print("GENERATING STAGE 1 PHYSICS FIGURES")
print("="*80)

fold1 = pd.read_csv('stage2_results/fold1_predictions.csv')
fold2 = pd.read_csv('stage2_results/fold2_predictions.csv')
fold3 = pd.read_csv('stage2_results/fold3_predictions.csv')

print(f"\n[1/3] Loaded {len(fold1)} + {len(fold2)} + {len(fold3)} samples")

all_preds = pd.concat([fold1, fold2, fold3], ignore_index=True)
all_preds['error'] = all_preds['predicted_aqi'] - all_preds['true_aqi']
all_preds['abs_error'] = np.abs(all_preds['error'])

mae = all_preds['abs_error'].mean()
rmse = np.sqrt((all_preds['error']**2).mean())
r2 = 1 - ((all_preds['error']**2).sum() / ((all_preds['true_aqi'] - all_preds['true_aqi'].mean())**2).sum())
bias = all_preds['error'].mean()

print(f"[2/3] MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}, Bias={bias:+.2f}")

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(all_preds['true_aqi'], all_preds['predicted_aqi'], alpha=0.6, s=30, c='darkblue', edgecolors='navy', linewidths=0.5)
max_val = max(all_preds['true_aqi'].max(), all_preds['predicted_aqi'].max())
ax.plot([0, max_val], [0, max_val], 'r--', lw=2.5, label='Perfect Prediction', alpha=0.8)
ax.set_xlabel('True AQI', fontsize=14, fontweight='bold')
ax.set_ylabel('Predicted AQI', fontsize=14, fontweight='bold')
ax.set_title('Stage 1 (Physics): Cross-Validation Results', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3, linestyle='--')
textstr = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.4f}\nBias: {bias:+.2f}\nn = {len(all_preds)}'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.7, edgecolor='black', linewidth=1.5)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=props, family='monospace')
plt.tight_layout()
plt.savefig('results/fig0_stage1_physics_scatter.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(all_preds['error'], bins=50, color='darkblue', alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax.axvline(bias, color='green', linestyle='--', linewidth=2, label=f'Mean Bias: {bias:+.2f}')
ax.set_xlabel('Prediction Error (Predicted - True)', fontsize=13, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
ax.set_title('Stage 1 (Physics): Residual Distribution', fontsize=15, fontweight='bold', pad=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('results/fig0b_stage1_physics_residuals.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("[3/3] Figures saved to results/")
print("="*80)
print("✅ DONE: fig0_stage1_physics_scatter.png")
print("✅ DONE: fig0b_stage1_physics_residuals.png")
print("="*80)
