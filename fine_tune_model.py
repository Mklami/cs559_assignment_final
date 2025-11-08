"""
Simple fine-tuning by loading weights and continuing training
This avoids custom metric serialization issues
"""
import os, sys, argparse

# Import your training script
import train_simple_cnn as train_module

def main():
    ap = argparse.ArgumentParser(description="Fine-tune by continuing training")
    
    # Required: checkpoint to load
    ap.add_argument("--checkpoint", required=True,
                    help="Path to best_model.keras to fine-tune")
    
    # Fine-tuning parameters (lower defaults)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="Fine-tuning learning rate (default: 1e-4)")
    ap.add_argument("--epochs", type=int, default=50,
                    help="Additional epochs")
    ap.add_argument("--patience", type=int, default=20,
                    help="Early stopping patience")
    
    # Architecture params (must match original model!)
    ap.add_argument("--num_conv", type=int, default=4)
    ap.add_argument("--filters", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--dense_units", type=int, default=128)
    ap.add_argument("--use_bn", action="store_true", help="Use batch norm (match original)")
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--init", choices=["glorot", "randn"], default="glorot")
    ap.add_argument("--loss", choices=["mae", "huber"], default="mae")
    
    # Optional architecture features (if using train_simple_cnn_v2.py)
    ap.add_argument("--conv_per_block", type=int, default=1)
    ap.add_argument("--use_strided_conv", action="store_true")
    
    ap.add_argument("--results_dir", default="results_homework")
    
    args = ap.parse_args()
    
    print("="*60)
    print("SIMPLE FINE-TUNING")
    print("="*60)
    print(f"\nLoading weights from: {args.checkpoint}")
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    # Build the model architecture (must match checkpoint!)
    print("\nRebuilding model architecture...")
    import tensorflow as tf
    from keras.optimizers import Adam
    
    try:
        from train_simple_cnn_v2 import build_simple_cnn
        print("Using train_simple_cnn_v2 architecture")
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
    except ImportError:
        from train_simple_cnn import build_simple_cnn
        print("Using train_simple_cnn architecture")
        model = build_simple_cnn(
            num_conv_layers=args.num_conv,
            filters=args.filters,
            dense_units=args.dense_units,
            init=args.init,
            l2=args.l2,
            dropout=args.dropout,
            use_bn=args.use_bn
        )
    
    # Load weights
    print("Loading weights...")
    try:
        model.load_weights(args.checkpoint)
        print("✅ Weights loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        print("\nMake sure architecture parameters match the original model!")
        print("Check the original run's run_settings.json for correct parameters.")
        return
    
    # Now continue training with lower LR
    print(f"\nFine-tuning with LR={args.lr} for {args.epochs} epochs...")
    
    # Import training components
    from prepare_dataset import build_ds
    import math
    import datetime
    import json
    import csv
    import numpy as np
    from evaluator import evaluate_predictions_csv
    
    @tf.function
    def rounded_mae(y_true, y_pred):
        y_pred_clamped = tf.clip_by_value(y_pred, 1.0, 8.0)
        return tf.reduce_mean(tf.abs(y_true - tf.round(y_pred_clamped)))
    
    def raw_mae(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # Compile with fine-tuning LR
    loss_fn = (tf.keras.losses.MeanAbsoluteError() if args.loss == "mae" 
               else tf.keras.losses.Huber())
    optimizer = Adam(learning_rate=args.lr)
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[raw_mae, rounded_mae]
    )
    
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
    
    # Create run directory
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = os.path.basename(os.path.dirname(args.checkpoint))
    run_name = f"{ts}_finetune_{base_name}_lr-{args.lr}"
    
    os.makedirs(args.results_dir, exist_ok=True)
    run_dir = os.path.join(args.results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Results will be saved to: {run_dir}")
    
    # Callbacks
    ckpt_path = os.path.join(run_dir, "finetuned_model.keras")
    csv_log_path = os.path.join(run_dir, "finetuning_log.csv")
    
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
        tf.keras.callbacks.CSVLogger(csv_log_path, separator=",", append=False)
    ]
    
    # Fine-tune
    print("\n" + "="*60)
    print(f"FINE-TUNING FOR {args.epochs} EPOCHS")
    print("="*60)
    
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating fine-tuned model...")
    test_metrics = model.evaluate(test_ds, return_dict=True)
    print("\nTEST METRICS:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Raw MAE: {test_metrics['raw_mae']:.4f}")
    print(f"  Rounded MAE: {test_metrics['rounded_mae']:.4f}")
    
    # Save predictions
    preds = model.predict(test_ds).reshape(-1)
    test_paths = sorted(tf.io.gfile.glob("test/*.jpg"))
    true_labels = [float(os.path.basename(p).split("_")[0]) for p in test_paths]
    
    preds_csv = os.path.join(run_dir, "test_predictions.csv")
    with open(preds_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "pred_rounded"])
        for p, t, pr in zip(test_paths, true_labels, preds):
            w.writerow([p, t, pr, round(np.clip(pr, 1, 8))])
    
    metrics = evaluate_predictions_csv(preds_csv, label_set=None,
                                      save_confusion_to=os.path.join(run_dir, "confusion.csv"))
    
    # Save settings
    settings = vars(args)
    settings["original_checkpoint"] = args.checkpoint
    
    settings_path = os.path.join(run_dir, "finetune_settings.json")
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ FINE-TUNING COMPLETE!")
    print("="*60)
    print(f"Test MAE (rounded): {metrics['mae_rounded']:.4f}")
    print(f"Test MAE (raw): {metrics['mae_raw']:.4f}")
    print(f"Results saved to: {run_dir}")
    print("="*60)

if __name__ == "__main__":
    main()