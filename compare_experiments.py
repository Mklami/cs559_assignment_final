#!/usr/bin/env python3
"""
Compare all experiments and find best configurations
"""
import pandas as pd
import os
import glob

results_dir = "results_homework"

# Find all run directories
runs = []
for run_dir in glob.glob(os.path.join(results_dir, "*")):
    if not os.path.isdir(run_dir):
        continue
    
    settings_file = os.path.join(run_dir, "run_settings.json")
    log_file = os.path.join(run_dir, "training_log.csv")
    
    if os.path.exists(settings_file) and os.path.exists(log_file):
        runs.append(run_dir)

if not runs:
    print(f"No runs found in {results_dir}")
    exit(1)

print(f"Found {len(runs)} experiment runs\n")
print("="*100)

# Collect results
results = []
for run_dir in runs:
    import json
    
    # Load settings
    with open(os.path.join(run_dir, "run_settings.json")) as f:
        settings = json.load(f)
    
    # Load training log
    log = pd.read_csv(os.path.join(run_dir, "training_log.csv"))
    
    # Get best validation epoch
    best_idx = log['val_raw_mae'].idxmin()
    best_epoch = log.loc[best_idx, 'epoch']
    best_val_mae = log.loc[best_idx, 'val_raw_mae']
    final_val_mae = log['val_raw_mae'].iloc[-1]
    
    # Try to load test results
    test_mae = None
    test_pred_file = os.path.join(run_dir, "test_predictions.csv")
    if os.path.exists(test_pred_file):
        test_df = pd.read_csv(test_pred_file)
        if 'true' in test_df and 'pred_rounded' in test_df:
            import numpy as np
            test_mae = np.mean(np.abs(test_df['true'] - test_df['pred_rounded']))
    
    results.append({
        'run_name': os.path.basename(run_dir),
        'num_conv': settings.get('num_conv', 3),
        'filters': str(settings.get('filters', [])),
        'use_bn': settings.get('use_bn', False),
        'l2': settings.get('l2', 0),
        'dropout': settings.get('dropout', 0),
        'augment': settings.get('augment', False),
        'lr': settings.get('lr', 1e-3),
        'batch': settings.get('batch', 64),
        'loss': settings.get('loss', 'mae'),
        'best_epoch': int(best_epoch),
        'best_val_mae': float(best_val_mae),
        'final_val_mae': float(final_val_mae),
        'test_mae': float(test_mae) if test_mae is not None else None,
        'total_epochs': len(log)
    })

# Create DataFrame and sort
df = pd.DataFrame(results)

# Sort by test MAE (or validation if test not available)
df['sort_key'] = df['test_mae'].fillna(df['best_val_mae'])
df = df.sort_values('sort_key')

print("TOP 10 EXPERIMENTS (by Test MAE)")
print("="*100)

# Display top 10
top10 = df.head(10).copy()
top10_display = top10[[
    'num_conv', 'filters', 'use_bn', 'l2', 'dropout', 'augment', 
    'loss', 'lr', 'best_val_mae', 'test_mae'
]].copy()

# Format for display
top10_display['filters'] = top10_display['filters'].str.replace('[', '').str.replace(']', '')
top10_display['use_bn'] = top10_display['use_bn'].map({True: 'Y', False: 'N'})
top10_display['augment'] = top10_display['augment'].map({True: 'Y', False: 'N'})

print(top10_display.to_string(index=False))

print("\n" + "="*100)
print(f"\n🏆 BEST MODEL:")
best = df.iloc[0]
print(f"   Test MAE: {best['test_mae']:.4f}")
print(f"   Config: {best['num_conv']} conv, filters={best['filters']}")
print(f"   BN={best['use_bn']}, L2={best['l2']}, Dropout={best['dropout']}, Aug={best['augment']}")
print(f"   Loss={best['loss']}")
print(f"   LR={best['lr']}, Batch={best['batch']}")
print(f"   Best at epoch {best['best_epoch']}/{best['total_epochs']}")
print(f"   Run: {best['run_name']}")

# Save full results
output_csv = os.path.join(results_dir, "all_experiments_comparison.csv")
df.to_csv(output_csv, index=False)
print(f"\n💾 Full results saved to: {output_csv}")

# Identify patterns
print("\n" + "="*100)
print("PATTERNS:")

if len(df[df['augment'] == True]) > 0 and len(df[df['augment'] == False]) > 0:
    aug_mean = df[df['augment'] == True]['test_mae'].mean()
    no_aug_mean = df[df['augment'] == False]['test_mae'].mean()
    print(f"   Augmentation: Avg MAE = {aug_mean:.4f} vs No Aug = {no_aug_mean:.4f}")
    if aug_mean < no_aug_mean:
        print(f"   → Augmentation helps! Δ={no_aug_mean-aug_mean:.4f}")

# More layers better?
if len(df[df['num_conv'] == 4]) > 0 and len(df[df['num_conv'] == 3]) > 0:
    conv4_mean = df[df['num_conv'] == 4]['test_mae'].mean()
    conv3_mean = df[df['num_conv'] == 3]['test_mae'].mean()
    print(f"   4 conv layers: Avg MAE = {conv4_mean:.4f} vs 3 layers = {conv3_mean:.4f}")

print("="*100)