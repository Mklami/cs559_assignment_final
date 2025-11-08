"""
Fine-tune a pre-trained model with lower learning rate
"""
import os, csv, argparse, datetime, json
import tensorflow as tf
from keras import models
from keras.optimizers import Adam
import pandas as pd
from prepare_dataset import build_ds
from evaluator import evaluate_predictions_csv
import numpy as np
import math

def main():
    ap = argparse.ArgumentParser(description="Fine-tune a pre-trained model")
    
    # Required: path to model
    ap.add_argument("--model_path", required=True,
                    help="Path to best_model.keras to fine-tune")
    
    # Fine-tuning parameters
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="Fine-tuning learning rate (default: 1e-4, 10x lower)")
    ap.add_argument("--epochs", type=int, default=50,
                    help="Additional epochs for fine-tuning")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--augment", action="store_true",
                    help="Use augmentation during fine-tuning")
    ap.add_argument("--reduce_dropout", type=float, default=None,
                    help="Optionally reduce dropout (e.g., from 0.2 to 0.1)")
    ap.add_argument("--patience", type=int, default=20,
                    help="Early stopping patience")
    ap.add_argument("--results_dir", default="results_homework")
    
    args = ap.parse_args()
    
    # Load pre-trained model
    print("="*60)
    print("FINE-TUNING PRE-TRAINED MODEL")
    print("="*60)
    print(f"\nLoading model from: {args.model_path}")
    
    try:
        model = models.load_model(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.summary()
    
    # Optionally reduce dropout
    if args.reduce_dropout is not None:
        print(f"\nReducing dropout to {args.reduce_dropout}...")
        from keras import layers as L
        
        # Create new model with reduced dropout
        # This is tricky - for simplicity, we'll just note it in logs
        print("Note: Dropout reduction requires model reconstruction")
        print("Continuing with original dropout for now")
    
    # Create run directory
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = os.path.basename(os.path.dirname(args.model_path))
    run_name = f"{ts}_finetune_{base_name}_lr-{args.lr}"
    
    os.makedirs(args.results_dir, exist_ok=True)
    run_dir = os.path.join(args.results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\nFine-tuning results will be saved to: {run_dir}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_ds = build_ds("training/*.jpg", batch=args.batch, 
                       is_train=True, augment=args.augment)
    val_ds = build_ds("validation/*.jpg", batch=args.batch, 
                     is_train=False, augment=False)
    test_ds = build_ds("test/*.jpg", batch=args.batch, 
                      is_train=False, augment=False)
    
    train_paths = sorted(tf.io.gfile.glob("training/*.jpg"))
    steps_per_epoch = math.ceil(len(train_paths) / args.batch)
    
    # Compile with lower learning rate
    print(f"\nCompiling with fine-tuning LR: {args.lr}")
    optimizer = Adam(learning_rate=args.lr)
    
    model.compile(
        optimizer=optimizer,
        loss=model.loss,
        metrics=model.metrics
    )
    
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
    
    # Evaluate on test set
    print("\nEvaluating fine-tuned model on test set...")
    test_metrics = model.evaluate(test_ds, return_dict=True)
    print("\nTEST METRICS (after fine-tuning):")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Raw MAE: {test_metrics['raw_mae']:.4f}")
    print(f"  Rounded MAE: {test_metrics['rounded_mae']:.4f}")
    
    # Save predictions
    print("\nSaving predictions...")
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
    
    # Save settings
    settings = {
        "original_model": args.model_path,
        "finetune_lr": args.lr,
        "finetune_epochs": args.epochs,
        "augment": args.augment,
        "batch": args.batch,
        "patience": args.patience
    }
    
    settings_path = os.path.join(run_dir, "finetune_settings.json")
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)
    
    print("\n" + "="*60)
    print("FINE-TUNING COMPLETE!")
    print("="*60)
    print(f"Test MAE (rounded): {metrics['mae_rounded']:.4f}")
    print(f"Test MAE (raw): {metrics['mae_raw']:.4f}")
    print(f"Results saved to: {run_dir}")
    print("="*60)

if __name__ == "__main__":
    main()