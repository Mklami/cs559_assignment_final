"""
Script to find best (success) and worst (failure) prediction examples
from your best performing model for the homework report.

Use this with your best model run: 
- run_name: results_homework/20251108-164527_finetune_20251108-112840_conv4_bn_glorot_l2-1e-05_drop-0.2_lr-0.001_lr-0.0001
- Test MAE: 0.78
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import os

# Step 1: Load your best model
# (Adjust path to your saved model)
MODEL_PATH = 'results_homework/20251108-164527_finetune_20251108-112840_conv4_bn_glorot_l2-1e-05_drop-0.2_lr-0.001_lr-0.0001/finetuned_model.keras'
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Step 2: Load test set images and labels
# This assumes you have a way to load your test data
# Adjust based on how you structured your data loading

def load_test_data(test_dir):
    """
    Load test images and extract labels from filenames
    Format: <attractiveness_level>_<acquisition_id>.jpg
    """
    images = []
    labels = []
    filenames = []
    
    for filename in sorted(os.listdir(test_dir)):
        if filename.endswith('.jpg'):
            # Extract label from filename
            label = float(filename.split('_')[0])
            
            # Load and preprocess image
            img_path = os.path.join(test_dir, filename)
            img = Image.open(img_path)
            img = np.array(img) / 255.0  # Normalize to [0, 1]
            
            images.append(img)
            labels.append(label)
            filenames.append(filename)
    
    return np.array(images), np.array(labels), filenames

# Load test data
test_images, test_labels, test_filenames = load_test_data('test')

# Step 3: Make predictions
predictions = model.predict(test_images)
predictions = predictions.flatten()  # Convert to 1D array

# Step 4: Calculate absolute errors
errors = np.abs(test_labels - predictions)

# Step 5: Find best predictions (smallest errors)
best_indices = np.argsort(errors)[:10]  # Top 10 best predictions
print("Best Predictions (Success Examples):")
print("-" * 60)
for idx in best_indices[:5]:  # Show top 5
    print(f"File: {test_filenames[idx]}")
    print(f"  True Label: {test_labels[idx]:.2f}")
    print(f"  Predicted: {predictions[idx]:.2f}")
    print(f"  Error: {errors[idx]:.4f}")
    print()

# Step 6: Find worst predictions (largest errors)
worst_indices = np.argsort(errors)[-10:][::-1]  # Top 10 worst predictions
print("\nWorst Predictions (Failure Examples):")
print("-" * 60)
for idx in worst_indices[:5]:  # Show top 5
    print(f"File: {test_filenames[idx]}")
    print(f"  True Label: {test_labels[idx]:.2f}")
    print(f"  Predicted: {predictions[idx]:.2f}")
    print(f"  Error: {errors[idx]:.4f}")
    print()

# Step 7: Visualize and save examples for report
def save_examples_for_report(indices, images, labels, predictions, errors, 
                             prefix='example', output_dir='report_figures'):
    """Save example images with predictions for the report"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, idx in enumerate(indices):
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        
        # Display image
        ax.imshow(images[idx])
        ax.axis('off')
        
        # Add title with prediction info
        title = f'True: {labels[idx]:.1f}, Pred: {predictions[idx]:.2f}\nError: {errors[idx]:.3f}'
        ax.set_title(title, fontsize=10)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prefix}_{i+1}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()

# Save success examples (best 3)
save_examples_for_report(best_indices[:3], test_images, test_labels, 
                        predictions, errors, prefix='success')

# Save failure examples (worst 3)
save_examples_for_report(worst_indices[:3], test_images, test_labels, 
                        predictions, errors, prefix='failure')

print(f"\nExample images saved to 'report_figures/' directory")

# Step 8: Additional analysis - what makes predictions good/bad?
print("\n" + "="*60)
print("ANALYSIS:")
print("="*60)

# Analyze error distribution
print(f"\nOverall Test Statistics:")
print(f"  Mean Error: {np.mean(errors):.4f}")
print(f"  Median Error: {np.median(errors):.4f}")
print(f"  Std Dev of Error: {np.std(errors):.4f}")
print(f"  Max Error: {np.max(errors):.4f}")
print(f"  Min Error: {np.min(errors):.4f}")

# Analyze if certain attractiveness levels are harder
print(f"\nError by True Attractiveness Level:")
for level in sorted(set(test_labels)):
    mask = test_labels == level
    if np.sum(mask) > 0:
        level_errors = errors[mask]
        print(f"  Level {level:.0f}: Mean Error = {np.mean(level_errors):.4f} " +
              f"(n={np.sum(mask)})")