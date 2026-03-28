"""
================================================================================
VICTORY VISUALIZATION (LOCAL MAC VERSION)
Objective: Generate the 3 critical plots using your local CSV.
================================================================================
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Target the file specifically on your Mac
RESULTS_PATH = "/Users/nesar/VS/V1/stage3_results/stage3_test_results.csv"
OUTPUT_DIR = "/Users/nesar/VS/V1/stage3_results"

if not os.path.exists(RESULTS_PATH):
    raise FileNotFoundError(f"❌ File not found at: {RESULTS_PATH}")

print(f"✅ Loading results from: {RESULTS_PATH}")
df = pd.read_csv(RESULTS_PATH)

# Set Style
sns.set_style("whitegrid")

# ==========================================
# PLOT 1: SCATTER (Truth vs Pred)
# ==========================================
plt.figure(figsize=(8, 8))
sns.scatterplot(x='True_AQI', y='Pred_AQI', data=df, alpha=0.5, color='#2ecc71', edgecolor='w')
min_val = min(df['True_AQI'].min(), df['Pred_AQI'].min())
max_val = max(df['True_AQI'].max(), df['Pred_AQI'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
plt.title('Stage 3 Fusion: True vs Predicted AQI', fontsize=14, fontweight='bold')
plt.xlabel('Ground Truth AQI', fontsize=12)
plt.ylabel('Predicted AQI (Fusion)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'final_plot_scatter.png'), dpi=300)
print("🖼️ Saved Scatter Plot")

# ==========================================
# PLOT 2: GATE BEHAVIOR
# ==========================================
plt.figure(figsize=(10, 5))
sns.histplot(df['Gate'], bins=50, kde=True, color='#3498db', edgecolor='black', alpha=0.7)
plt.title('Gate Arbitration (0=Vision, 1=Physics)', fontsize=14, fontweight='bold')
plt.xlabel('Gate Value', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.axvline(x=0.5, color='red', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'final_plot_gate.png'), dpi=300)
print("🖼️ Saved Gate Plot")

# ==========================================
# PLOT 3: ERROR DENSITY
# ==========================================
plt.figure(figsize=(10, 5))
df['Fusion_Error'] = df['Pred_AQI'] - df['True_AQI']
if 'Vision_Input' in df.columns:
    df['Vision_Error'] = df['Vision_Input'] - df['True_AQI']
    sns.kdeplot(df['Vision_Error'], fill=True, color='red', alpha=0.2, label='Stage 2 (Vision Only)')
sns.kdeplot(df['Fusion_Error'], fill=True, color='blue', alpha=0.4, label='Stage 3 (Fusion)')
plt.title('Error Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Error', fontsize=12)
plt.xlim(-150, 150)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'final_plot_error.png'), dpi=300)
print("🖼️ Saved Error Plot")