## CS559 Facial Attractiveness Project

This repository contains the tooling used for the CS559 final assignment on facial attractiveness estimation. It includes a TensorFlow/Keras training pipeline, dataset utilities, and an evaluation script that aggregates model metrics across saved experiment runs.

### Repository Layout

- `train_simple_cnn.py` – trains the baseline Simple CNN, saves checkpoints, logs, and predictions under `results_homework/<run_name>/`.
- `prepare_dataset.py` – builds TensorFlow datasets for the training/validation/test splits found in the `training/`, `validation/`, and `test/` folders.
- `evaluator.py` – evaluates any `test_predictions.csv` files under `results_homework`, attaches validation metrics from matching `training_log.csv` files, and writes `summary_metrics.csv`.
- `results_homework/` – default output directory for experiment artifacts (predictions, confusion matrices, logs, summaries).
- `venv/` – optional Python virtual environment you may create locally.

### Getting Started

1. **Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install tensorflow pandas numpy
   ```
   (Install any additional packages you need for experiments.)

2. **Dataset Layout**
   Place the assignment images in the provided folders:
   - `training/*.jpg`
   - `validation/*.jpg`
   - `test/*.jpg`

### Training a Model

Run the trainer with any desired hyperparameters. A timestamped run directory will be created automatically under `results_homework/`.

```bash
python train_simple_cnn.py \
    --num_conv 3 \
    --filters 32 64 128 \
    --dense_units 128 \
    --lr 1e-3 \
    --results_dir results_homework
```

Key files produced per run:
- `best_model.keras` – best checkpoint by `val_raw_mae`.
- `training_log.csv` / `training_history.json` – epoch-level metrics.
- `test_predictions.csv` – predictions used by `evaluator.py`.
- `confusion.csv` – confusion matrix for rounded predictions.

### Evaluating Results

Aggregate metrics for every run with `test_predictions.csv`:

```bash
python evaluator.py \
    --results_dir results_homework \
    --out_csv results_homework/summary_metrics.csv
```

`summary_metrics.csv` will include accuracy/MAE metrics plus the best validation MAE and epoch retrieved from each run's training log. Additional confusion matrices are saved next to each run's predictions as `<run_name>_confusion.csv`.

### Tips

- Use `python train_simple_cnn.py --help` and `python evaluator.py --help` to view all available options.
- You can supply `--labels` to `evaluator.py` for a fixed label set when comparing confusion matrices across runs.
- To compare experiments programmatically, load `results_homework/summary_metrics.csv` into pandas and sort by `mae_rounded` or `best_val_raw_mae`.


