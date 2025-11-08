import os, csv, argparse, datetime, shutil, math, json
import tensorflow as tf
from keras import layers as L, models, regularizers, initializers
from keras.optimizers import Adam
from keras.saving import register_keras_serializable
import pandas as pd

from prepare_dataset import build_ds, IMG_SIZE, parse_example
from evaluator import evaluate_predictions_csv


# ===========================
# Callbacks
# ===========================
class EpochPrinter(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        keys = [k for k in ["loss","raw_mae","rounded_mae",
                            "val_loss","val_raw_mae","val_rounded_mae"]
                if k in logs]
        tail = " | ".join(f"{k}={logs[k]:.4f}" for k in keys)
        print(f"Epoch {epoch+1}: {tail}", flush=True)


# ===========================
# Architecture (no skip connections)
# ===========================
def conv_block(x, f, k=3, s=1, bn=True, l2=0.0, dropout=0.0):
    x = L.Conv2D(
        f, k, strides=s, padding="same",
        kernel_regularizer=regularizers.l2(l2) if l2>0 else None
    )(x)
    if bn:
        x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    if dropout > 0:
        x = L.Dropout(dropout)(x)
    return x

def build_model(init="glorot", bn=True, l2=1e-5, dropout=0.2):
    kernel_init = initializers.GlorotUniform() if init=="glorot" \
                  else initializers.RandomNormal(stddev=0.05)
    inp = L.Input((*IMG_SIZE, 3))

    # Stem
    x = L.Conv2D(32, 3, padding="same", kernel_initializer=kernel_init,
                 kernel_regularizer=regularizers.l2(l2))(inp)
    x = L.ReLU()(x)
    if bn:
        x = L.BatchNormalization()(x)
    x = L.MaxPool2D()(x)

    # Body
    x = conv_block(x, 64,  bn=bn, l2=l2, dropout=dropout)
    x = conv_block(x, 128, bn=bn, l2=l2, dropout=dropout)
    x = conv_block(x, 128, bn=bn, l2=l2, dropout=dropout)

    # Head
    x = L.GlobalAveragePooling2D()(x)
    if dropout > 0:
        x = L.Dropout(dropout)(x)
    x = L.Dense(128, activation="relu",
                kernel_regularizer=regularizers.l2(l2))(x)
    if dropout > 0:
        x = L.Dropout(dropout)(x)

    out = L.Dense(1, activation="linear")(x)
    return models.Model(inp, out)


# ===========================
# Metrics (registered & reload-safe)
# ===========================
@register_keras_serializable(package="custom")
def rounded_mae(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1.0, 8.0)
    return tf.reduce_mean(tf.abs(y_true - tf.round(y_pred)))

@register_keras_serializable(package="custom")
def raw_mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


# ===========================
# Simple finite dataset builder for FT
# ===========================
def make_epoch_ds(files, batch, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle:
        ds = ds.shuffle(len(files), reshuffle_each_iteration=True)
    ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds


# ===========================
# Main
# ===========================
def main():

    ap = argparse.ArgumentParser()

    # Dataset
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--augment", action="store_true")

    # Training hyperparameters
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", choices=["mae", "huber"], default="mae")
    ap.add_argument("--init", choices=["glorot", "randn"], default="glorot")
    ap.add_argument("--l2", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--bn", action="store_true")
    ap.add_argument("--patience", type=int, default=15)

    # Fine-tuning
    ap.add_argument("--finetune_from", default="", help="Path to previous best_model.keras")
    ap.add_argument("--ft_epochs", type=int, default=0)
    ap.add_argument("--ft_lr", type=float, default=1e-4)
    ap.add_argument("--ft_use_val", action="store_true")  # include validation in fine-tuning dataset

    # Outputs
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--update_summary", action="store_true")

    args = ap.parse_args()

    # ----- Dataset loading -----
    train_ds = build_ds("training/*.jpg", batch=args.batch,
                        is_train=True, augment=args.augment)
    val_ds   = build_ds("validation/*.jpg", batch=args.batch,
                        is_train=False, augment=False)
    test_ds  = build_ds("test/*.jpg", batch=args.batch,
                        is_train=False, augment=False)

    train_files = sorted(tf.io.gfile.glob("training/*.jpg"))
    val_files   = sorted(tf.io.gfile.glob("validation/*.jpg"))
    test_files  = sorted(tf.io.gfile.glob("test/*.jpg"))

    steps_per_epoch = math.ceil(len(train_files) / args.batch)

    # ----- Create run folder -----
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_core = f"init-{args.init}_l2-{args.l2}_dropout-{args.dropout}_lr-{args.lr}_batch-{args.batch}"
    run_name = f"{ts}_{run_core}"
    os.makedirs(args.results_dir, exist_ok=True)
    run_dir = os.path.join(args.results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # ----- Model -----
    print("Building model...")

    loss_fn = tf.keras.losses.MeanAbsoluteError() if args.loss=="mae" else tf.keras.losses.Huber()

    if args.finetune_from:
        print(f"Loading model for fine-tuning: {args.finetune_from}")
        # Load without compiling (safer across TF/Keras versions)
        model = tf.keras.models.load_model(
            args.finetune_from,
            compile=False
        )
        # Re-compile fresh with our optimizer/metrics
        model.compile(optimizer=Adam(args.lr),
                      loss=loss_fn,
                      metrics=[raw_mae, rounded_mae])
    else:
        model = build_model(init=args.init, bn=args.bn, l2=args.l2, dropout=args.dropout)
        model.compile(optimizer=Adam(args.lr),
                      loss=loss_fn,
                      metrics=[raw_mae, rounded_mae])

    # Paths
    ckpt_path      = os.path.join(run_dir, "best_model.keras")
    csv_log_path   = os.path.join(run_dir, "train_log.csv")
    hist_json_path = os.path.join(run_dir, "history.json")

    # Callbacks
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, monitor="val_raw_mae",
                                           save_best_only=True, mode="min", verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_raw_mae", mode="min",
                                         patience=args.patience, 
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.CSVLogger(csv_log_path),
        EpochPrinter(),
    ]

    # -------- Training --------
    print("Training...")
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=cbs,
        verbose=1,
    )

    with open(hist_json_path, "w") as f:
        json.dump(hist.history, f)

    # Test set evaluation
    print("Evaluating...")
    print(model.evaluate(test_ds, return_dict=True))

    # Save predictions
    preds = model.predict(test_ds).reshape(-1)
    true_labels = [float(os.path.basename(x).split("_")[0]) for x in test_files]

    out_csv = os.path.join(run_dir, "test_predictions.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path","true","pred","pred_rounded"])
        for p, t, pr in zip(test_files, true_labels, preds):
            w.writerow([p, t, pr, round(pr)])

    # Save settings
    settings = {
        "run_name": run_name,
        "timestamp": ts,
        "batch": args.batch,
        "epochs": args.epochs,
        "lr": args.lr,
        "finetune_from": args.finetune_from,
    }
    with open(os.path.join(run_dir, "settings.json"), "w") as f:
        json.dump(settings, f, indent=2)

    # -------- Optional fine-tuning --------
    if args.ft_epochs > 0:
        print("\n===== Fine-tuning =====")

        # Load best weights first (compile=False, then compile)
        model = tf.keras.models.load_model(
            ckpt_path,
            compile=False
        )
        model.compile(optimizer=Adam(args.ft_lr),
                      loss=loss_fn,
                      metrics=[raw_mae, rounded_mae])

        # Build FT dataset
        ft_files = list(train_files)
        if args.ft_use_val:
            ft_files += list(val_files)

        ft_ds = make_epoch_ds(ft_files, args.batch, shuffle=True)
        ft_steps = math.ceil(len(ft_files) / args.batch)

        # FT callbacks
        cbs_ft = [
            tf.keras.callbacks.ModelCheckpoint(os.path.join(run_dir, "best_model_finetuned.keras"),
                                               monitor="val_raw_mae", save_best_only=True, mode="min"),
            tf.keras.callbacks.EarlyStopping(monitor="val_raw_mae", patience=args.patience,
                                             mode="min", restore_best_weights=True),
            EpochPrinter(),
        ]

        print("Fine-tuning...")
        model.fit(
            ft_ds,
            validation_data=val_ds,
            epochs=args.ft_epochs,
            steps_per_epoch=ft_steps,
            callbacks=cbs_ft,
            verbose=1,
        )

        # Evaluate FT
        print("Evaluating finetuned model...")
        print(model.evaluate(test_ds, return_dict=True))

        # Save FT predictions
        preds_ft = model.predict(test_ds).reshape(-1)
        out_ft_csv = os.path.join(run_dir, "test_predictions_finetuned.csv")
        with open(out_ft_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path","true","pred","pred_rounded"])
            for p, t, pr in zip(test_files, true_labels, preds_ft):
                w.writerow([p, t, pr, round(pr)])

    print(f"\nRun saved under: {run_dir}")

if __name__ == "__main__":
    main()
