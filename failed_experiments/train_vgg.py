import os, csv, argparse, datetime, shutil
import tensorflow as tf
from keras import layers as L, models, regularizers, initializers
import pandas as pd
from prepare_dataset import build_ds, IMG_SIZE
import glob, math
import json
from evaluator import evaluate_predictions_csv
from keras.applications import vgg16


class EpochPrinter(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        keys = [k for k in ["loss","raw_mae","rounded_mae","val_loss","val_raw_mae","val_rounded_mae"] if k in logs]
        tail = " | ".join(f"{k}={logs[k]:.4f}" for k in keys)
        print(f"Epoch {epoch+1}: {tail}", flush=True)

def conv_block(x, f, k=3, s=1, bn=False, l2=0.0, dropout=0.0):
    x = L.Conv2D(
        f, k, strides=s, padding="same",
        kernel_regularizer=regularizers.l2(l2) if l2 > 0 else None
    )(x)
    if bn:
        x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    if dropout > 0:
        x = L.Dropout(dropout)(x)
    return x

from keras import regularizers, models, initializers, layers as L

def build_model_vgg_backbone(init="glorot", l2=0.0, dropout=0.0, bn=False, train_last_block=False):
    inp = L.Input((*IMG_SIZE, 3))
    # VGG preprocessing: you can also plug this into your tf.data pipeline
    x = vgg16.preprocess_input(inp)

    # Pretrained VGG16 feature extractor
    base = vgg16.VGG16(include_top=False, weights="imagenet", input_tensor=x)
    for layer in base.layers:
        layer.trainable = False

    # Optionally unfreeze the last VGG block (block5) for fine-tuning
    if train_last_block:
        for layer in base.layers:
            if layer.name.startswith("block5"):
                layer.trainable = True

    x = base.output
    x = L.GlobalAveragePooling2D()(x)
    if bn:
        x = L.BatchNormalization()(x)
    x = L.Dense(
        256, activation="relu",
        kernel_regularizer=regularizers.l2(l2) if l2 > 0 else None
    )(x)
    if dropout > 0:
        x = L.Dropout(dropout)(x)
    out = L.Dense(1, activation="linear")(x)
    return models.Model(inp, out)

@tf.function
def rounded_mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - tf.round(y_pred)))

def raw_mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", choices=["mae", "huber"], default="mae")
    ap.add_argument("--init", choices=["glorot", "randn"], default="glorot")
    ap.add_argument("--bn", action="store_true")
    ap.add_argument("--l2", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--train_last_block", default=False)
    ap.add_argument("--model_out", default="best_model.keras")
    ap.add_argument("--log_csv", default="training_log.csv")
    ap.add_argument("--log_json", default="training_history.json")
    ap.add_argument("--tb_logdir", default=None, help="If None, will use run_dir/tb automatically")
    ap.add_argument("--results_dir", default="results_vgg")
    args = ap.parse_args()

    BATCH = args.batch

    # Run identity & folders
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_core = f"init-{args.init}_bn-{args.bn}_l2-{args.l2}_dropout-{args.dropout}_loss-{args.loss}_lr-{args.lr}_batch-{args.batch}_trainlastblock-{args.train_last_block}"
    run_name = f"{ts}_{run_core}"
    os.makedirs(args.results_dir, exist_ok=True)
    run_dir = os.path.join(args.results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Data (relative patterns now point at symlinks in project root)
    train_ds = build_ds("training/*.jpg", batch=BATCH, is_train=True)
    val_ds   = build_ds("validation/*.jpg", batch=BATCH, is_train=False)
    test_ds  = build_ds("test/*.jpg", batch=BATCH, is_train=False)

    train_paths = sorted(tf.io.gfile.glob("training/*.jpg"))
    if not train_paths:
        raise RuntimeError("No images found under training/*.jpg")
    steps_per_epoch = math.ceil(len(train_paths) / BATCH)

    # Model
    print("Building model...")
    model = build_model_vgg_backbone(init=args.init, bn=args.bn, l2=args.l2, dropout=args.dropout, train_last_block=False)
    loss_fn = tf.keras.losses.MeanAbsoluteError() if args.loss == "mae" else tf.keras.losses.Huber()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss=loss_fn, metrics=[raw_mae, rounded_mae])

    # Paths inside run folder
    ckpt_path      = os.path.join(run_dir, "best_model.keras")
    csv_log_path   = os.path.join(run_dir, "training_log.csv")
    hist_json_path = os.path.join(run_dir, "training_history.json")
    tb_dir         = args.tb_logdir or os.path.join(run_dir, "tb")

    # Callbacks
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, monitor="val_raw_mae",
                                           mode="min", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_raw_mae", mode="min",
                                         patience=max(10, args.patience), min_delta=0.02,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_raw_mae", mode="min",
                                             factor=0.5, patience=5, min_lr=1e-5, verbose=1),
        EpochPrinter(),
        tf.keras.callbacks.CSVLogger(csv_log_path, separator=",", append=False),
        tf.keras.callbacks.TensorBoard(log_dir=tb_dir, update_freq="epoch"),
    ]

    # Train
    print("Training...")
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        callbacks=cbs,
        verbose=1,
    )
    with open(hist_json_path, "w") as f:
        json.dump(history.history, f)

    # Evaluate & save predictions
    print("Evaluating on test set (best-val model)…")
    test_metrics = model.evaluate(test_ds, return_dict=True)
    print("TEST METRICS:", test_metrics)

    print("Saving test predictions…")
    preds = model.predict(test_ds).reshape(-1)
    test_paths = sorted(tf.io.gfile.glob("test/*.jpg"))
    true_labels = [float(os.path.basename(p).split("_")[0]) for p in test_paths]

    preds_csv_in_run = os.path.join(run_dir, "test_predictions.csv")
    with open(preds_csv_in_run, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "pred_rounded"])
        for p, t, pr in zip(test_paths, true_labels, preds):
            w.writerow([p, t, pr, round(pr)])

    # Flat copy for evaluator pattern *_test_predictions.csv
    flat_preds_csv = os.path.join(args.results_dir, f"{run_name}_test_predictions.csv")
    shutil.copyfile(preds_csv_in_run, flat_preds_csv)

    # Save run settings
    run_settings_csv = os.path.join(run_dir, "run_settings.csv")
    settings = {
        "run_name": run_name,
        "timestamp": ts,
        "batch": args.batch,
        "epochs": args.epochs,
        "lr": args.lr,
        "loss": args.loss,
        "init": args.init,
        "bn": args.bn,
        "l2": args.l2,
        "dropout": args.dropout,
        "patience": args.patience,
        "steps_per_epoch": steps_per_epoch,
        "n_train_images": len(train_paths),
        "model_checkpoint": os.path.relpath(ckpt_path, args.results_dir),
        "preds_csv_rel": os.path.relpath(flat_preds_csv, args.results_dir),
    }
    with open(run_settings_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(settings.keys()))
        w.writeheader(); w.writerow(settings)

    # Update results/summary_metrics.csv
    cm_out = os.path.join(run_dir, "confusion.csv")
    m = evaluate_predictions_csv(flat_preds_csv, label_set=None, save_confusion_to=cm_out)
    m["run_name"] = run_name

    summary_csv = os.path.join(args.results_dir, "summary_metrics.csv")
    cols = [
        "run_name","num_samples","mae_rounded","mae_raw","rmse_raw",
        "acc_exact","acc_within_1","acc_within_2",
        "balanced_accuracy","pearson_r","spearman_r","pred_entropy_bits"
    ]
    if os.path.exists(summary_csv):
        df = pd.read_csv(summary_csv)
        df = df[df["run_name"] != run_name]
        df = pd.concat([df, pd.DataFrame([{k: m.get(k, "") for k in cols}])], ignore_index=True)
        df = df[cols].sort_values("run_name")
        df.to_csv(summary_csv, index=False)
    else:
        import csv as _csv
        with open(summary_csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerow({k: m.get(k, "") for k in cols})

    print(f"Run artifacts saved under: {run_dir}")
    print(f"Predictions (flat copy): {flat_preds_csv}")
    print(f"Summary updated: {summary_csv}")

if __name__ == "__main__":
    main()