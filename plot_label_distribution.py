#!/usr/bin/env python3
"""
Plot the attractiveness label distribution for the training, validation,
and test folders used in the CS559 assignment.

The script assumes that each image filename begins with the attractiveness
score followed by an underscore (e.g. '1_AF52.jpg').
"""

import argparse
import collections
import glob
import os
from typing import Dict, List

import matplotlib.pyplot as plt


def _collect_counts(directory: str, pattern: str) -> Dict[int, int]:
    """
    Count how many images fall under each attractiveness score in a directory.
    Filenames are expected to start with '<score>_'.
    """
    counter: collections.Counter[int] = collections.Counter()

    paths = glob.glob(os.path.join(directory, pattern))
    for path in paths:
        base = os.path.basename(path)
        if "_" not in base:
            continue
        prefix = base.split("_", 1)[0]
        try:
            score = int(float(prefix))
        except ValueError:
            continue
        counter[score] += 1

    return dict(counter)


def _build_dataframe(counts_list: List[Dict[int, int]], label_names: List[str]):
    """
    Turn a list of count dictionaries into a structure that is easy to plot.
    Returns:
        labels: sorted list of attractiveness scores seen across all datasets
        values: list of lists (one per dataset) aligned with labels
    """
    scores = sorted({score for counts in counts_list for score in counts.keys()})
    values: List[List[int]] = []
    for counts in counts_list:
        values.append([counts.get(score, 0) for score in scores])
    return scores, values


def plot_distributions(directories: List[str], pattern: str, save_path: str | None = None, show: bool = True) -> None:
    """
    Plot the attractiveness distribution for each directory.
    """
    labels = ["Training", "Validation", "Test"]
    counts_list = [_collect_counts(d, pattern) for d in directories]

    # Print counts to console
    for name, counts in zip(labels, counts_list):
        if counts:
            print(f"{name} distribution:")
            for score in sorted(counts.keys()):
                print(f"  Score {score}: {counts[score]} images")
        else:
            print(f"{name} distribution: (no matching files)")
        print()

    scores, values = _build_dataframe(counts_list, labels)

    if not scores:
        print("No images found with the expected naming convention. Nothing to plot.")
        return

    bar_width = 0.25
    x = range(len(scores))

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, dataset_values in enumerate(values):
        offsets = [pos + (idx - 1) * bar_width for pos in x]
        ax.bar(offsets, dataset_values, width=bar_width, label=labels[idx])

    ax.set_xlabel("Attractiveness Score")
    ax.set_ylabel("Number of Images")
    ax.set_title("Attractiveness Distribution per Split")

    ax.set_xticks([pos for pos in x])
    ax.set_xticklabels([str(score) for score in scores])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot attractiveness label distribution for dataset splits.")
    parser.add_argument("--train_dir", default="training", help="Directory containing training images.")
    parser.add_argument("--val_dir", default="validation", help="Directory containing validation images.")
    parser.add_argument("--test_dir", default="test", help="Directory containing test images.")
    parser.add_argument("--pattern", default="*.jpg", help="Glob pattern for image files within each directory.")
    parser.add_argument("--save", default="", help="Optional path to save the plot as an image/PDF.")
    parser.add_argument("--no-show", action="store_true", help="Generate the plot without displaying it.")
    args = parser.parse_args()

    directories = [args.train_dir, args.val_dir, args.test_dir]
    save_path = args.save if args.save else None
    show_plot = not args.no_show

    plot_distributions(directories, args.pattern, save_path=save_path, show=show_plot)


if __name__ == "__main__":
    main()

