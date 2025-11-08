#!/usr/bin/env python3
"""
Plot training curves for the best model
"""
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python plot_training_curves.py <path_to_training_log.csv>")
    print("\nExample:")
    print("  python plot_training_curves.py results_homework/YOUR_BEST_RUN/training_log.csv")
    sys.exit(1)

log_path = sys.argv[1]

if not os.path.exists(log_path):
    print(f"Error: File not found: {log_path}")
    sys.exit(1)

# Load training log
df = pd.read_csv(log_path)
print(f"Loaded {len(df)} epochs from {log_path}")

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Loss curves
ax1.plot(df['epoch'], df['loss'], label='Training Loss', linewidth=2, color='#1f77b4')
ax1.plot(df['epoch'], df['val_loss'], label='Validation Loss', linewidth=2, color='#ff7f0e')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss (MAE)', fontsize=12)
ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: MAE curves
ax2.plot(df['epoch'], df['raw_mae'], label='Training MAE', linewidth=2, color='#1f77b4', linestyle='--')
ax2.plot(df['epoch'], df['val_raw_mae'], label='Validation MAE', linewidth=2, color='#ff7f0e')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Mean Absolute Error', fontsize=12)
ax2.set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, loc='upper right')
ax2.grid(True, alpha=0.3)

# Find best epoch
best_epoch = df['val_raw_mae'].idxmin()
best_val_mae = df.loc[best_epoch, 'val_raw_mae']

# Mark best epoch on both plots
ax1.axvline(x=best_epoch, color='green', linestyle=':', alpha=0.7, linewidth=1.5)
ax2.axvline(x=best_epoch, color='green', linestyle=':', alpha=0.7, linewidth=1.5, 
            label=f'Best: Epoch {best_epoch} (Val MAE={best_val_mae:.3f})')
ax2.legend(fontsize=10, loc='upper right')

plt.tight_layout()

# Save
output_path = os.path.join(os.path.dirname(log_path), 'training_curves.pdf')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved to: {output_path}")

# Also save PNG for preview
output_png = output_path.replace('.pdf', '.png')
plt.savefig(output_png, dpi=150, bbox_inches='tight')
print(f"✅ PNG preview: {output_png}")

# Show statistics
print("\n" + "="*60)
print("TRAINING STATISTICS")
print("="*60)
print(f"Total epochs trained: {len(df)}")
print(f"Best validation epoch: {best_epoch}")
print(f"Best validation MAE: {best_val_mae:.4f}")
print(f"Final training MAE: {df['raw_mae'].iloc[-1]:.4f}")
print(f"Final validation MAE: {df['val_raw_mae'].iloc[-1]:.4f}")
print(f"Improvement: {df['val_raw_mae'].iloc[0]:.4f} → {best_val_mae:.4f} ({df['val_raw_mae'].iloc[0] - best_val_mae:.4f})")
print("="*60)

plt.show()