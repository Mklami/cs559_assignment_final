# evaluator.py
import argparse, os, glob, json
import numpy as np
import pandas as pd
import re, math

_TS_PREFIX = re.compile(r'^\d{8}-\d{6}_')  # e.g., 20251030-131237_
_TS_ONLY = re.compile(r'^\d{8}-\d{6}$')

def _normalize_name(name: str) -> str:
    # remove timestamp prefix if any
    return _TS_PREFIX.sub('', name)

def _collect_training_logs(results_dir: str):
    """
    Return a list of (run_key, path_to_training_log).
    run_key is both the raw base name AND a normalized (timestamp-stripped) variant,
    so we can match in multiple ways.
    """
    pairs = []

    # 1) Flat files: results/*_training_log.csv
    for p in glob.glob(os.path.join(results_dir, '*_training_log.csv')):
        base = os.path.basename(p).removesuffix('_training_log.csv')
        pairs.append((base, p))
        pairs.append((_normalize_name(base), p))

    # 2) Timestamped dirs: results/<dir>/training_log.csv
    for d in glob.glob(os.path.join(results_dir, '*')):
        if not os.path.isdir(d):
            continue
        tl = os.path.join(d, 'training_log.csv')
        if os.path.exists(tl):
            dirbase = os.path.basename(d)
            pairs.append((dirbase, tl))
            pairs.append((_normalize_name(dirbase), tl))

    return pairs  # may contain duplicated keys mapping to same path

def _best_path_for_run(run_name: str, key_paths: list[tuple[str, str]]) -> str | None:
    """
    Choose a training_log path for a given run_name using:
      exact match on key, then startswith/endswith overlap, then substring.
    Works with timestamp-normalized variants too (since pairs include both).
    """
    rn_raw = str(run_name)
    rn_norm = _normalize_name(rn_raw)
    keys = [k for k,_ in key_paths]

    # exact
    for k, p in key_paths:
        if k == rn_raw or k == rn_norm:
            return p

    # startswith/endswith (prefer longer keys)
    sw = [(k,p) for (k,p) in key_paths if k.startswith(rn_norm) or rn_norm.startswith(k)]
    if sw:
        return max(sw, key=lambda kp: len(kp[0]))[1]

    # substring
    sub = [(k,p) for (k,p) in key_paths if (rn_norm in k) or (k in rn_norm)]
    if sub:
        return max(sub, key=lambda kp: len(kp[0]))[1]

    return None

def attach_best_val_metrics(summary: pd.DataFrame, results_dir: str = "results") -> pd.DataFrame:
    """
    Adds columns:
      best_val_raw_mae, best_val_epoch, final_val_raw_mae, final_val_loss
    by reading per-run training logs from either:
      - results/<run_name>_training_log.csv
      - results/<timestamp>_<run_name>/training_log.csv
    """
    key_paths = _collect_training_logs(results_dir)

    # get run names from index or column
    if "run_name" in summary.columns:
        run_names = summary["run_name"].astype(str).tolist()
    else:
        run_names = [str(i) for i in summary.index.tolist()]

    best_val, best_ep, final_val, final_loss = [], [], [], []

    for rn in run_names:
        path = _best_path_for_run(rn, key_paths)
        if not path:
            # no log found for this run
            best_val.append(math.nan); best_ep.append(math.nan)
            final_val.append(math.nan); final_loss.append(math.nan)
            continue

        try:
            df = pd.read_csv(path)
            # pick the correct val metric column
            metric_col = "val_raw_mae" if "val_raw_mae" in df.columns else (
                         "val_mae"     if "val_mae"     in df.columns else None)

            if metric_col is None:
                best_val.append(math.nan); best_ep.append(math.nan)
                final_val.append(math.nan)
            else:
                idx = int(df[metric_col].idxmin())
                best_val.append(float(df.loc[idx, metric_col]))
                ep = int(df.loc[idx, "epoch"]) if "epoch" in df.columns else idx
                best_ep.append(ep)
                final_val.append(float(df[metric_col].iloc[-1]))

            final_loss.append(float(df["val_loss"].iloc[-1]) if "val_loss" in df.columns else math.nan)
        except Exception:
            best_val.append(math.nan); best_ep.append(math.nan)
            final_val.append(math.nan); final_loss.append(math.nan)

    summary["best_val_raw_mae"]  = best_val
    summary["best_val_epoch"]    = best_ep
    summary["final_val_raw_mae"] = final_val
    summary["final_val_loss"]    = final_loss
    return summary


def _parse_run_name_details(run_name: str) -> dict[str, object]:
    """Infer hyperparameters encoded in a run_name.

    Supports run names produced by both train_simple_cnn.py and train_simple_cnn_v2.py,
    and falls back gracefully when tokens are missing/unexpected.
    """
    details: dict[str, object] = {}

    if not run_name:
        return details

    tokens = run_name.split('_')
    if tokens and _TS_ONLY.match(tokens[0]):
        details["run_timestamp"] = tokens[0]
        tokens = tokens[1:]

    # Separate tail hyperparameter tokens (init, l2, drop, lr) from the rest
    tail_tokens = []
    remaining_tokens = []
    for tok in tokens:
        if tok in {"glorot", "randn", "he", "xavier"} or tok.startswith("l2-") \
           or tok.startswith("drop-") or tok.startswith("lr-"):
            tail_tokens.append(tok)
        else:
            remaining_tokens.append(tok)

    tokens_arch = remaining_tokens

    # Handle finetune runs (train_simple_cnn_v2 does not use this but keep for completeness)
    if tokens_arch and tokens_arch[0] == "finetune":
        details["model_type"] = "finetune"
        tokens_arch = tokens_arch[1:]
        if tokens_arch:
            details["finetune_base"] = tokens_arch[0]
            tokens_arch = tokens_arch[1:]

    # Parse architecture tokens
    for tok in tokens_arch:
        if tok.startswith("conv") and tok[4:].isdigit():
            details["conv_blocks"] = int(tok[4:])
            continue
        if tok.startswith("conv") and "x" in tok:
            # e.g., "conv4" token already handled; support combined forms like conv4x2
            m = re.match(r"conv(\d+)[xX](\d+)", tok)
            if m:
                details["conv_blocks"] = int(m.group(1))
                details["conv_per_block"] = int(m.group(2))
                continue
        if tok.startswith("x") and tok[1:].isdigit():
            details["conv_per_block"] = int(tok[1:])
            continue
        if tok == "bn":
            details["use_bn"] = True
            continue
        if tok == "nobn":
            details["use_bn"] = False
            continue
        if tok == "strided":
            details["use_strided_conv"] = True
            continue
        if tok == "augment" or tok == "aug":
            details["use_augment"] = True
            continue
        if tok.startswith("f") and len(tok) > 1:
            # Filter configuration, keep as string
            details["filters"] = tok[1:]
            continue
        # Unrecognized tokens are ignored to keep parser robust

    # Parse tail hyperparameter tokens
    for tok in tail_tokens:
        if tok in {"glorot", "randn", "he", "xavier"}:
            details["init"] = tok
        elif tok.startswith("l2-"):
            try:
                details["l2"] = float(tok.split("-", 1)[1])
            except ValueError:
                pass
        elif tok.startswith("drop-"):
            try:
                details["dropout"] = float(tok.split("-", 1)[1])
            except ValueError:
                pass
        elif tok.startswith("lr-"):
            try:
                details["lr"] = float(tok.split("-", 1)[1])
            except ValueError:
                pass

    return details


def ensure_int_labels(arr):
    # ground-truth labels are integers in this assignment
    return np.asarray(arr, dtype=float).astype(int)

def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))

def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def accuracy_exact(y_true, y_pred_round):
    return float(np.mean(y_true == y_pred_round))

def accuracy_within_k(y_true, y_pred_round, k=1):
    return float(np.mean(np.abs(y_true - y_pred_round) <= k))

def shannon_entropy(counts):
    """Shannon entropy (base 2) of a discrete distribution."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def spearmanr(y, x):
    """Spearman correlation without SciPy (simple rank-based Pearson)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    y_ranks = y.argsort().argsort()
    x_ranks = x.argsort().argsort()
    # Convert to 1-based average ranks for ties (simple handling):
    # For small datasets and mostly unique values, the below is fine.
    return float(np.corrcoef(y_ranks, x_ranks)[0,1])

def pearsonr(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.corrcoef(y, x)[0,1])

def confusion_matrix(y_true, y_pred_round, labels):
    L = len(labels)
    idx = {lab:i for i, lab in enumerate(labels)}
    cm = np.zeros((L, L), dtype=int)
    for t, p in zip(y_true, y_pred_round):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm

def balanced_accuracy(cm):
    # macro average of per-class recall
    recalls = []
    for i in range(cm.shape[0]):
        denom = cm[i,:].sum()
        if denom == 0:
            recalls.append(np.nan)
        else:
            recalls.append(cm[i,i] / denom)
    recalls = np.array(recalls, dtype=float)
    return float(np.nanmean(recalls))

def evaluate_predictions_csv(csv_path, label_set=None, save_confusion_to=None):
    df = pd.read_csv(csv_path)
    if "true" not in df or "pred" not in df:
        raise ValueError(f"{csv_path} must contain columns: true, pred (and optionally pred_rounded).")

    y_true = ensure_int_labels(df["true"].values)
    y_pred = df["pred"].values.astype(float)
    if "pred_rounded" in df:
        y_pred_round = ensure_int_labels(df["pred_rounded"].values)
    else:
        y_pred_round = np.rint(y_pred).astype(int)

    # Label set inferred if not provided
    if label_set is None:
        label_set = sorted(list({*y_true.tolist()}))
    label_set = list(label_set)

    # Metrics
    m = {}
    m["num_samples"] = int(len(y_true))
    m["mae_raw"] = mae(y_true, y_pred)                   # MAE without rounding (for analysis)
    m["mae_rounded"] = mae(y_true, y_pred_round)         # official rule (rounded preds)
    m["rmse_raw"] = rmse(y_true, y_pred)
    m["acc_exact"] = accuracy_exact(y_true, y_pred_round)
    m["acc_within_1"] = accuracy_within_k(y_true, y_pred_round, k=1)
    m["acc_within_2"] = accuracy_within_k(y_true, y_pred_round, k=2)
    m["pearson_r"] = pearsonr(y_true, y_pred)
    m["spearman_r"] = spearmanr(y_true, y_pred)

    # Distribution + entropy of predicted rounded labels
    counts = pd.Series(y_pred_round).value_counts().reindex(label_set, fill_value=0).values
    m["pred_entropy_bits"] = shannon_entropy(counts)

    # Confusion & balanced accuracy
    cm = confusion_matrix(y_true, y_pred_round, labels=label_set)
    m["balanced_accuracy"] = balanced_accuracy(cm)

    # Optionally save confusion matrix
    if save_confusion_to:
        cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in label_set],
                                columns=[f"pred_{l}" for l in label_set])
        cm_df.to_csv(save_confusion_to, index=True)

    return m

def _get_output_dir_for_run(predictions_path: str, run_name: str, results_dir: str) -> str:
    """
    Determine the appropriate output directory for files related to a run.
    - If predictions_path is in a subdirectory, use that subdirectory
    - Otherwise, use/create a folder named after the run_name in results_dir
    """
    predictions_dir = os.path.normpath(os.path.dirname(predictions_path))
    results_dir_norm = os.path.normpath(results_dir)
    
    # If the file is already in a subdirectory (not root), use that directory
    if predictions_dir != results_dir_norm:
        return predictions_dir
    
    # Otherwise, use/create the run_name folder
    run_dir = os.path.join(results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def main():
    ap = argparse.ArgumentParser(description="Evaluate CS559 results folder and summarize metrics.")
    ap.add_argument("--results_dir", default="results_homework", help="Directory containing test_predictions.csv files")
    ap.add_argument("--labels", default="", help="Comma-separated label set (e.g., 1,2,3,4,5). If empty, infer from data.")
    ap.add_argument("--pattern", default="test_predictions.csv", help="Glob pattern to find result CSVs")
    ap.add_argument("--out_csv", default="results_homework/summary_metrics.csv", help="Path to write summary CSV")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    # Optional fixed label set (recommended for consistent confusion matrices)
    label_set = None
    if args.labels.strip():
        label_set = [int(s) for s in args.labels.split(",")]

    # Search recursively in subdirectories as well
    pattern = os.path.join(args.results_dir, "**", args.pattern)
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        print(f"No result files found under {args.results_dir} matching {args.pattern}")
        return

    rows = []
    for p in paths:
        # Extract run_name from parent directory name (matching train_simple_cnn.py structure)
        # Path format: results_homework/<run_name>/test_predictions.csv
        parent_dir = os.path.normpath(os.path.dirname(p))
        results_dir_norm = os.path.normpath(args.results_dir)
        
        # If file is in a subdirectory, use the subdirectory name as run_name
        if parent_dir != results_dir_norm:
            run_name = os.path.basename(parent_dir)
        else:
            # Fallback: if file is directly in results_dir, extract from filename
            run_name = os.path.basename(p).replace("_test_predictions.csv", "").replace("test_predictions.csv", "")
        
        # Determine the appropriate output directory for this run
        output_dir = _get_output_dir_for_run(p, run_name, args.results_dir)
        cm_out = os.path.join(output_dir, f"{run_name}_confusion.csv")
        metrics = evaluate_predictions_csv(p, label_set=label_set, save_confusion_to=cm_out)
        metrics["run_name"] = run_name

        # Add parsed hyperparameters/details
        metrics.update(_parse_run_name_details(run_name))
        rows.append(metrics)
        print(f"[OK] {run_name}: MAE_rounded={metrics['mae_rounded']:.4f}, "
              f"Acc@1={metrics['acc_within_1']:.4f}, BalAcc={metrics['balanced_accuracy']:.4f}")

    summary = pd.DataFrame(rows).set_index("run_name")
    summary = attach_best_val_metrics(summary, results_dir=args.results_dir)
    base_cols = [
        "num_samples",
        "mae_rounded", "mae_raw", "rmse_raw",
        "acc_exact", "acc_within_1", "acc_within_2",
        "balanced_accuracy", "pearson_r", "spearman_r",
        "pred_entropy_bits",
    ]
    detail_cols = [c for c in [
        "run_timestamp", "conv_blocks", "conv_per_block",
        "use_bn", "use_strided_conv", "use_augment",
        "filters", "init", "l2", "dropout", "lr",
        "model_type", "finetune_base"
    ] if c in summary.columns]
    extra_cols = [c for c in ["best_val_raw_mae", "best_val_epoch",
                            "final_val_raw_mae", "final_val_loss"]
                if c in summary.columns]
    summary = summary[base_cols + detail_cols + extra_cols]


    summary.to_csv(args.out_csv)
    print(f"\nSaved summary to: {args.out_csv}")
    print(summary)

if __name__ == "__main__":
    main()
