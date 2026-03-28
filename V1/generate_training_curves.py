import matplotlib.pyplot as plt
import numpy as np

epochs = list(range(1, 11))
train_mae = [34.26, 30.78, 29.03, 27.27, 26.16, 25.62, 24.58, 23.59, 22.96, 23.02]
val_mae = [29.26, 26.41, 24.09, 23.03, 21.85, 20.96, 20.43, 20.22, 19.27, 18.63]
train_r2 = [0.7602, 0.8072, 0.8353, 0.8531, 0.8645, 0.8752, 0.8826, 0.8946, 0.9010, 0.9015]
val_r2 = [0.8237, 0.8569, 0.8797, 0.8934, 0.9035, 0.9126, 0.9156, 0.9206, 0.9266, 0.9309]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs, train_mae, 'o-', linewidth=2.5, markersize=8, color='#2E86AB', label='Training MAE')
ax1.plot(epochs, val_mae, 's-', linewidth=2.5, markersize=8, color='#A23B72', label='Validation MAE')
ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=13, fontweight='bold')
ax1.set_title('Stage 2 Training Progress: MAE', fontsize=15, fontweight='bold', pad=15)
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xticks(epochs)

ax2.plot(epochs, train_r2, 'o-', linewidth=2.5, markersize=8, color='#2E86AB', label='Training R²')
ax2.plot(epochs, val_r2, 's-', linewidth=2.5, markersize=8, color='#A23B72', label='Validation R²')
ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax2.set_ylabel('R² Score', fontsize=13, fontweight='bold')
ax2.set_title('Stage 2 Training Progress: R²', fontsize=15, fontweight='bold', pad=15)
ax2.legend(fontsize=11, loc='lower right')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xticks(epochs)
ax2.set_ylim([0.75, 0.95])

plt.tight_layout()
plt.savefig('results/fig9_stage2_training_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("="*80)
print("✅ Training curves saved: results/fig9_stage2_training_curves.png")
print("="*80)
