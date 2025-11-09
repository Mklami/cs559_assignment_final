## CS559 Facial Attractiveness Project

Authors: Mayasah Kareem Lami - 22401352
Utku Boran Torun - 21901898

This repository contains the tooling used for the CS559 final assignment on facial attractiveness estimation. It includes multiple TensorFlow/Keras training scripts, dataset utilities, evaluation tooling, and analysis helpers to compare experiments and visualize results.

### Repository Layout (key scripts)

- `prepare_dataset.py` – builds TensorFlow datasets backed by the `training/`, `validation/`, and `test/` folders.
- `train_simple_cnn.py` – baseline CNN that mirrors the HW spec; logs metrics and predictions to `results_homework/<run_name>/`.
- `fine_tune_model.py` – resumes from a saved `.keras` checkpoint and fine-tunes with a smaller learning rate.
- `evaluator.py` – aggregates all `test_predictions.csv` files, computes metrics (including rounded MAE), attaches validation stats from `training_log.csv`, and writes `summary_metrics.csv`.
- `compare_experiments.py` – parses every run directory with `run_settings.json` + `training_log.csv`, ranks experiments, and writes `all_experiments_comparison.csv`.
- `plot_label_distribution.py` – visualizes attractiveness label histograms for the three dataset splits.
- `plot_training_curves.py` – quick utility to plot loss/MAE curves from a run’s `training_log.csv`.
- `best_worst_predictions.py` – loads a chosen model, re-evaluates the test set, saves best/worst examples, and exports `predictions_vs_true.csv`.
- `check_model_config.py` – sanity-checks a saved Keras model to confirm layer shapes and parameter counts.
- `failed_experiments/` – archived scripts and notebooks for approaches that did not outperform the main pipeline (e.g., early VGG trials, alternative strided convolution CNN variants). Useful for reference but not part of the primary workflow.
- `results_homework/` – default output directory for experiment artifacts (predictions, confusion matrices, logs, summaries, analysis CSVs).
- `report_figures/` – output folder for exported success/failure examples.

(*See inline `--help` for each script to discover optional flags.*)

### Getting Started

1. **Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install tensorflow pandas numpy pillow matplotlib seaborn
   ```
   - On Apple Silicon, `tensorflow-macos` can be installed instead of `tensorflow` if preferred.
   - Additional useful extras: `tensorflow-addons`, `jupyter`, `tqdm`.

2. **Dataset Layout**
   Place the assignment images in the provided folders:
   - `training/*.jpg`
   - `validation/*.jpg`
   - `test/*.jpg`

### Training Options

All training scripts emit a timestamped `run_name` folder inside `results_homework/` containing checkpoints, logs, predictions, and confusion matrices. Rounded predictions are stored by default so MAE-computation meets the assignment requirement.

**Baseline CNN**

```bash
python train_simple_cnn.py \
    --num_conv 3 \
    --filters 32 64 128 \
    --dense_units 128 \
    --lr 1e-3 \
    --results_dir results_homework
```

**Strided / high-capacity variant**

```bash
python train_simple_cnn_strConv.py \
    --num_blocks 4 \
    --filters 32 64 128 256 \
    --use_strided \
    --augment \
    --results_dir results_homework
```

**Fine-tuning an existing checkpoint**

```bash
python fine_tune_model.py \
    --model_path results_homework/<run_name>/best_model.keras \
    --lr 1e-4 \
    --epochs 30 \
    --results_dir results_homework
```

Each run directory typically contains:

- `best_model.keras` / `finetuned_model.keras` – best checkpoint by validation MAE.
- `training_log.csv` and `training_history.json` – epoch metrics used for plotting.
- `test_predictions.csv` – raw and rounded predictions per image (`pred_rounded` column).
- `<run_name>_confusion.csv` – confusion matrix for rounded predictions.
- `run_settings.json` (or `finetune_settings.json`) – hyperparameters used.
- Optional: `training_curves.png`/`.pdf`, `predictions_vs_true.csv` (from analysis scripts).

### Evaluating Results

Aggregate metrics for every run containing `test_predictions.csv` (works recursively through subdirectories):

```bash
python evaluator.py \
    --results_dir results_homework \
    --out_csv results_homework/summary_metrics.csv
```

`summary_metrics.csv` includes rounded MAE (`mae_rounded`), raw MAE, RMSE, accuracy thresholds, correlation metrics, per-run entropy, and any parsed hyperparameters that were inferred from run names. Validation metrics (`best_val_raw_mae`, `best_val_epoch`, etc.) are automatically joined when a matching `training_log.csv` exists.

### Tips

- Use `python <script>.py --help` to see the knobs available for each utility.
- Run `compare_experiments.py` to generate `all_experiments_comparison.csv` sorted by test MAE (rounded). This file includes the loss function, optimizer settings, and best-epoch information pulled from `run_settings.json`.
- `plot_label_distribution.py` provides a quick sanity check that each split covers the attractiveness range evenly.
- `best_worst_predictions.py` can generate publication-ready figures plus a CSV of per-image true vs predicted values for your report.
- All MAE metrics in this repo respect the assignment rule by using the rounded predictions stored in `pred_rounded`.



