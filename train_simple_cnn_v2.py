"""
Simple CNN for Facial Attractiveness Estimation
Following CS559 Homework Specifications
"""
import os, csv, argparse, datetime, shutil, math, json
import tensorflow as tf
from keras import layers as L, models, regularizers, initializers
from keras.optimizers import Adam
from keras.optimizers.schedules import CosineDecay
import pandas as pd
from prepare_dataset import build_ds, IMG_SIZE
from evaluator import evaluate_predictions_csv

# ===========================
# Simple CNN Architecture (Assignment Compliant)
# ===========================

def build_simple_cnn(num_conv_layers=3, filters=[32, 64, 128], 
                     dense_units=128, init="glorot", l2=0.0, dropout=0.0,
                     use_bn=False, use_strided_conv=False, conv_per_block=1):
    """
    Build a simple CNN with specified number of layers.
    
    Architecture follows assignment requirements:
    - Series of convolutional layers with ReLU
    - Pooling layers (or strided convolutions)
    - Fully connected layers
    - Regression output
    
    Args:
        num_conv_layers: Number of convolutional blocks
        filters: List of filter sizes for each block
        dense_units: Number of units in FC layer
        init: Weight initialization ('glorot' or 'randn')
        l2: L2 regularization weight
        dropout: Dropout rate
        use_bn: Whether to use batch normalization
        use_strided_conv: Use strided conv instead of max pooling
        conv_per_block: Number of conv layers per block (VGG-style)
    """
    kernel_init = (initializers.GlorotUniform() if init == "glorot" 
                   else initializers.RandomNormal(stddev=0.05))
    
    inp = L.Input((*IMG_SIZE, 3))
    x = inp
    
    # Convolutional blocks
    for i in range(num_conv_layers):
        block_filters = filters[i] if i < len(filters) else filters[-1]
        
        # Multiple conv layers per block (VGG-style)
        for j in range(conv_per_block):
            # Determine stride: only last conv in block downsamples (if strided)
            is_last_in_block = (j == conv_per_block - 1)
            stride = 2 if (use_strided_conv and is_last_in_block) else 1
            
            x = L.Conv2D(
                block_filters,
                kernel_size=3,
                strides=stride,
                padding="same",
                use_bias=(not use_bn),
                kernel_initializer=kernel_init,
                kernel_regularizer=regularizers.l2(l2) if l2 > 0 else None,
                name=f"conv{i+1}_{j+1}"
            )(x)
            
            # Batch Normalization (optional)
            if use_bn:
                x = L.BatchNormalization(name=f"bn{i+1}_{j+1}")(x)
            
            # ReLU activation
            x = L.ReLU(name=f"relu{i+1}_{j+1}")(x)
        
        # Pooling (only if NOT using strided convolutions)
        if not use_strided_conv:
            x = L.MaxPooling2D(pool_size=2, strides=2, name=f"pool{i+1}")(x)
        
        # Dropout after each block (optional)
        if dropout > 0:
            x = L.Dropout(dropout, name=f"dropout_block{i+1}")(x)
    
    # Global Average Pooling (reduces spatial dimensions)
    x = L.GlobalAveragePooling2D(name="gap")(x)
    
    # Fully connected layer
    x = L.Dense(
        dense_units,
        activation="relu",
        kernel_initializer=kernel_init,
        kernel_regularizer=regularizers.l2(l2) if l2 > 0 else None,
        name="fc1"
    )(x)
    
    # Dropout before output
    if dropout > 0:
        x = L.Dropout(dropout, name="dropout_fc")(x)
    
    # Regression output (linear activation)
    out = L.Dense(1, activation="linear", name="output")(x)
    
    return models.Model(inp, out, name="SimpleCNN")


# ===========================
# Metrics
# ===========================

@tf.function
def rounded_mae(y_true, y_pred):
    """MAE after rounding predictions (as required by assignment)"""
    y_pred_clamped = tf.clip_by_value(y_pred, 1.0, 8.0)
    return tf.reduce_mean(tf.abs(y_true - tf.round(y_pred_clamped)))

def raw_mae(y_true, y_pred):
    """MAE without rounding"""
    return tf.reduce_mean(tf.abs(y_true - y_pred))


# ===========================
# Epoch Printer Callback
# ===========================

class EpochPrinter(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        keys = [k for k in ["loss", "raw_mae", "rounded_mae", "val_loss", 
                           "val_raw_mae", "val_rounded_mae"] if k in logs]
        tail = " | ".join(f"{k}={logs[k]:.4f}" for k in keys)
        print(f"Epoch {epoch+1}: {tail}", flush=True)


# ===========================
# Main Training Function
# ===========================

def main():
    ap = argparse.ArgumentParser(description="Simple CNN for Attractiveness Estimation")
    
    # Architecture parameters
    ap.add_argument("--num_conv", type=int, default=3, 
                    help="Number of convolutional blocks")
    ap.add_argument("--filters", type=int, nargs="+", default=[32, 64, 128],
                    help="Filter sizes for conv blocks")
    ap.add_argument("--dense_units", type=int, default=128,
                    help="Number of units in FC layer")
    ap.add_argument("--conv_per_block", type=int, default=1,
                    help="Number of conv layers per block (VGG-style: 2-3)")
    ap.add_argument("--use_strided_conv", action="store_true",
                    help="Use strided convolutions instead of max pooling")
    
    # Training parameters
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    
    # Techniques to explore (as required by assignment)
    ap.add_argument("--loss", choices=["mae", "huber"], default="mae",
                    help="Loss function (importance: 10%)")
    ap.add_argument("--init", choices=["glorot", "randn"], default="glorot",
                    help="Weight initialization (importance: 10%)")
    ap.add_argument("--use_bn", action="store_true",
                    help="Use batch normalization (importance: 10%)")
    ap.add_argument("--l2", type=float, default=0.0,
                    help="L2 regularization weight (importance: 15%)")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="Dropout rate (importance: 15%)")
    ap.add_argument("--augment", action="store_true",
                    help="Enable data augmentation")
    
    # Other parameters
    ap.add_argument("--patience", type=int, default=15,
                    help="Early stopping patience (importance: 25%)")
    ap.add_argument("--results_dir", default="results_homework")
    
    args = ap.parse_args()
    
    # Create run directory
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Build descriptive run name including architecture details
    arch_desc = []
    arch_desc.append(f"conv{args.num_conv}")
    
    # Add conv_per_block if > 1 (VGG-style)
    if args.conv_per_block > 1:
        arch_desc.append(f"x{args.conv_per_block}")  # e.g., "conv4x2" = 4 blocks, 2 conv each
    
    # Add strided conv indicator
    if args.use_strided_conv:
        arch_desc.append("strided")
    
    # Add filter configuration (abbreviated)
    filter_str = "-".join(map(str, args.filters))
    arch_desc.append(f"f{filter_str}")
    
    # Add BN indicator
    arch_desc.append('bn' if args.use_bn else 'nobn')
    
    # Add augmentation indicator
    if args.augment:
        arch_desc.append('aug')
    
    # Combine architecture description
    arch_name = "_".join(arch_desc)
    
    # Add hyperparameters
    run_name = (f"{ts}_{arch_name}_"
                f"{args.init}_l2-{args.l2}_drop-{args.dropout}_"
                f"lr-{args.lr}")
    
    os.makedirs(args.results_dir, exist_ok=True)
    run_dir = os.path.join(args.results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    train_ds = build_ds("training/*.jpg", batch=args.batch, 
                       is_train=True, augment=args.augment)
    val_ds = build_ds("validation/*.jpg", batch=args.batch, 
                     is_train=False, augment=False)
    test_ds = build_ds("test/*.jpg", batch=args.batch, 
                      is_train=False, augment=False)
    
    train_paths = sorted(tf.io.gfile.glob("training/*.jpg"))
    steps_per_epoch = math.ceil(len(train_paths) / args.batch)
    
    # Build model
    print(f"\nBuilding model...")
    print(f"  Conv blocks: {args.num_conv}")
    print(f"  Conv per block: {args.conv_per_block}")
    print(f"  Filters: {args.filters}")
    print(f"  Strided conv: {args.use_strided_conv}")
    print(f"  Batch Norm: {args.use_bn}")
    print(f"  Initialization: {args.init}")
    print(f"  L2: {args.l2}")
    print(f"  Dropout: {args.dropout}")
    
    model = build_simple_cnn(
        num_conv_layers=args.num_conv,
        filters=args.filters,
        dense_units=args.dense_units,
        init=args.init,
        l2=args.l2,
        dropout=args.dropout,
        use_bn=args.use_bn,
        use_strided_conv=args.use_strided_conv,
        conv_per_block=args.conv_per_block
    )
    
    model.summary()
    
    # Compile model
    loss_fn = (tf.keras.losses.MeanAbsoluteError() if args.loss == "mae" 
               else tf.keras.losses.Huber())
    
    optimizer = Adam(learning_rate=args.lr)
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[raw_mae, rounded_mae]
    )
    
    # Callbacks
    ckpt_path = os.path.join(run_dir, "best_model.keras")
    csv_log_path = os.path.join(run_dir, "training_log.csv")
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_raw_mae",
            mode="min",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_raw_mae",
            mode="min",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1
        ),
        EpochPrinter(),
        tf.keras.callbacks.CSVLogger(csv_log_path, separator=",", append=False)
    ]
    
    # Train
    print("\nTraining...")
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save history
    hist_json_path = os.path.join(run_dir, "training_history.json")
    with open(hist_json_path, "w") as f:
        json.dump(history.history, f)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = model.evaluate(test_ds, return_dict=True)
    print("TEST METRICS:", test_metrics)
    
    # Save predictions
    print("Saving predictions...")
    preds = model.predict(test_ds).reshape(-1)
    test_paths = sorted(tf.io.gfile.glob("test/*.jpg"))
    true_labels = [float(os.path.basename(p).split("_")[0]) for p in test_paths]
    
    preds_csv = os.path.join(run_dir, "test_predictions.csv")
    with open(preds_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "pred_rounded"])
        for p, t, pr in zip(test_paths, true_labels, preds):
            w.writerow([p, t, pr, round(np.clip(pr, 1, 8))])
    
    # Evaluate with confusion matrix
    metrics = evaluate_predictions_csv(preds_csv, label_set=None,
                                      save_confusion_to=os.path.join(run_dir, "confusion.csv"))
    
    # Save run settings
    settings_path = os.path.join(run_dir, "run_settings.json")
    with open(settings_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"Results saved to: {run_dir}")
    print(f"Test MAE (rounded): {metrics['mae_rounded']:.4f}")
    print(f"Test MAE (raw): {metrics['mae_raw']:.4f}")


if __name__ == "__main__":
    import numpy as np
    main()