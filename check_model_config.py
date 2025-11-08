#!/usr/bin/env python3
"""
Check model configuration from a saved run
Helps you know what parameters to use for fine-tuning
"""
import json
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python check_model_config.py <path_to_run_directory>")
    print("\nExample:")
    print("  python check_model_config.py results_homework/20251108-112840_conv4_bn_glorot_l2-1e-05_drop-0.2_lr-0.001")
    sys.exit(1)

run_dir = sys.argv[1]
settings_file = os.path.join(run_dir, "run_settings.json")

if not os.path.exists(settings_file):
    print(f"❌ Error: {settings_file} not found")
    print("\nMake sure you're pointing to a results directory that has run_settings.json")
    sys.exit(1)

# Load settings
with open(settings_file) as f:
    settings = json.load(f)

print("="*60)
print("MODEL CONFIGURATION")
print("="*60)
print(f"Run: {os.path.basename(run_dir)}")
print("")

# Architecture
print("ARCHITECTURE:")
print(f"  num_conv:      {settings.get('num_conv', 'N/A')}")
print(f"  filters:       {settings.get('filters', 'N/A')}")
print(f"  dense_units:   {settings.get('dense_units', 'N/A')}")
print(f"  conv_per_block: {settings.get('conv_per_block', 1)} {'(VGG-style)' if settings.get('conv_per_block', 1) > 1 else ''}")
print(f"  use_strided:   {settings.get('use_strided_conv', False)}")
print()

# Regularization
print("REGULARIZATION:")
print(f"  use_bn:        {settings.get('use_bn', False)}")
print(f"  l2:            {settings.get('l2', 0)}")
print(f"  dropout:       {settings.get('dropout', 0)}")
print(f"  augment:       {settings.get('augment', False)}")
print()

# Training
print("TRAINING:")
print(f"  init:          {settings.get('init', 'glorot')}")
print(f"  loss:          {settings.get('loss', 'mae')}")
print(f"  lr:            {settings.get('lr', 1e-3)}")
print(f"  batch:         {settings.get('batch', 64)}")
print(f"  epochs:        {settings.get('epochs', 100)}")
print(f"  patience:      {settings.get('patience', 15)}")

print()
print("="*60)
print("FINE-TUNING COMMAND")
print("="*60)

# Generate fine-tuning command
model_path = os.path.join(run_dir, "best_model.keras")

# Determine which script to use
if settings.get('conv_per_block', 1) > 1 or settings.get('use_strided_conv', False):
    script = "finetune_simple.py"
else:
    script = "finetune_simple.py"  # Use simple for both

cmd = f"python {script} \\\n"
cmd += f"  --checkpoint {model_path} \\\n"
cmd += f"  --num_conv {settings.get('num_conv', 4)} \\\n"

# Filters
filters = settings.get('filters', [32, 64, 128, 256])
if isinstance(filters, list):
    cmd += f"  --filters {' '.join(map(str, filters))} \\\n"

# Architecture features
if settings.get('use_bn', False):
    cmd += f"  --use_bn \\\n"
if settings.get('conv_per_block', 1) > 1:
    cmd += f"  --conv_per_block {settings.get('conv_per_block')} \\\n"
if settings.get('use_strided_conv', False):
    cmd += f"  --use_strided_conv \\\n"

# Regularization
cmd += f"  --l2 {settings.get('l2', 1e-4)} \\\n"
cmd += f"  --dropout {settings.get('dropout', 0.2)} \\\n"
if settings.get('augment', False):
    cmd += f"  --augment \\\n"

# Fine-tuning params (10x lower LR)
original_lr = settings.get('lr', 1e-3)
finetune_lr = original_lr / 10
cmd += f"  --lr {finetune_lr} \\\n"
cmd += f"  --epochs 50 \\\n"
cmd += f"  --patience 20"

print(cmd)
print()
print("="*60)
print("\n💡 Copy and paste the command above to fine-tune this model")
print()