#!/usr/bin/env python3
"""Grid-search LightGCN lr/decay over CV folds for lgn and mf."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOSS_RE = re.compile(r"EPOCH\[(\d+)/(\d+)\]\s+loss([0-9]*\.?[0-9]+)-")
ARRAY_RE = r"array\(\[([^\]]+)\]\)"
METRICS_RE = re.compile(
    rf"\{{'precision':\s*{ARRAY_RE},\s*'recall':\s*{ARRAY_RE},\s*'ndcg':\s*{ARRAY_RE}\}}"
)

GRID_VALUES = [1.0, 1e-1, 1e-2, 1e-3, 1e-4]
MODELS = ["lgn", "mf"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search lr/decay for LightGCN and MF with CV.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name passed to LightGCN (e.g. ml-1m).")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per run.")
    parser.add_argument("--layer", type=int, default=1, help="LightGCN layer count.")
    parser.add_argument("--recdim", type=int, default=64, help="Embedding dimension.")
    parser.add_argument("--topks", type=str, default="[20]", help="LightGCN --topks string.")
    parser.add_argument("--seed", type=int, default=2020, help="Base random seed.")
    parser.add_argument(
        "--bpr-batch",
        type=int,
        default=1_000_000,
        help="BPR batch size. Requirement default is 1,000,000.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=None,
        help="Number of CV folds. If omitted, inferred from train_fold_*.txt in dataset directory.",
    )
    parser.add_argument(
        "--lightgcn-code-dir",
        type=Path,
        default=Path("LightGCN/code"),
        help="Directory containing LightGCN main.py.",
    )
    parser.add_argument(
        "--lightgcn-data-dir",
        type=Path,
        default=Path("LightGCN/data"),
        help="Root containing per-dataset subdirectories.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Output CSV path. Default: mixing/hyperparam/results/<dataset>.csv",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help=(
            "Output PDF path. Default: mixing/hyperparam/plots/<dataset>.pdf "
            "(kept to match requested path)."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on first failed LightGCN run.",
    )
    return parser.parse_args()


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_training_output(stdout_text: str, topks: list[int]) -> dict:
    loss_matches = list(LOSS_RE.finditer(stdout_text))
    if not loss_matches:
        raise RuntimeError("Could not parse final loss from LightGCN output.")

    metric_matches = list(METRICS_RE.finditer(stdout_text))
    if not metric_matches:
        raise RuntimeError("Could not parse evaluation metrics from LightGCN output.")

    last_metrics = metric_matches[-1]
    precision_vals = _parse_float_list(last_metrics.group(1))
    recall_vals = _parse_float_list(last_metrics.group(2))
    ndcg_vals = _parse_float_list(last_metrics.group(3))

    if not (len(precision_vals) == len(recall_vals) == len(ndcg_vals) == len(topks)):
        raise RuntimeError(
            "Metric length mismatch vs topks: "
            f"topks={topks}, got={(len(precision_vals), len(recall_vals), len(ndcg_vals))}"
        )

    return {
        "precision": {f"precision@{k}": v for k, v in zip(topks, precision_vals)},
        "recall": {f"recall@{k}": v for k, v in zip(topks, recall_vals)},
        "ndcg": {f"ndcg@{k}": v for k, v in zip(topks, ndcg_vals)},
    }


def infer_num_folds(dataset_dir: Path) -> int:
    fold_files = sorted(dataset_dir.glob("train_fold_*.txt"))
    if not fold_files:
        raise FileNotFoundError(
            f"No train_fold_*.txt files found in {dataset_dir}. "
            "Create CV splits first (e.g., mixing/data_preproc/movielens.py)."
        )
    return len(fold_files)


def run_one(
    code_dir: Path,
    dataset: str,
    model_name: str,
    lr: float,
    decay: float,
    fold_idx: int,
    epochs: int,
    layer: int,
    recdim: int,
    topks: str,
    bpr_batch: int,
    seed: int,
) -> tuple[dict, float]:
    cmd = [
        "python",
        "main.py",
        f"--model={model_name}",
        f"--dataset={dataset}",
        f"--lr={lr}",
        f"--decay={decay}",
        f"--layer={layer}",
        f"--recdim={recdim}",
        f"--topks={topks}",
        f"--bpr_batch={bpr_batch}",
        "--eval_split=val",
        f"--val_split_idx={fold_idx}",
        "--tensorboard=0",
        f"--seed={seed}",
        "--epochs",
        str(epochs),
    ]
    print(" ".join(cmd))
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(code_dir), capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - t0

    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Run failed (model={model_name}, lr={lr}, decay={decay}, fold={fold_idx}) "
            f"with exit code {proc.returncode}."
        )

    topks_list = [int(x.strip()) for x in topks.strip("[]").split(",") if x.strip()]
    parsed = parse_training_output(proc.stdout, topks=topks_list)
    return parsed, elapsed


def write_csv(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmaps(plot_path: Path, rows: list[dict], recall_key: str) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    x_labels = [f"1e-{i}" if i > 0 else "1" for i in range(6)]
    y_labels = [f"1e-{i}" if i > 0 else "1" for i in range(6)]

    for ax, model_name in zip(axes, MODELS):
        grid = np.full((len(GRID_VALUES), len(GRID_VALUES)), np.nan, dtype=float)
        for i, lr in enumerate(GRID_VALUES):
            for j, decay in enumerate(GRID_VALUES):
                match = [
                    r
                    for r in rows
                    if r["model"] == model_name
                    and math.isclose(float(r["lr"]), float(lr))
                    and math.isclose(float(r["decay"]), float(decay))
                ]
                if match:
                    grid[i, j] = float(match[0]["recall_cv_mean"])

        im = ax.imshow(grid, cmap="viridis", aspect="auto")
        ax.set_title(f"{model_name}: mean {recall_key}")
        ax.set_xlabel("decay")
        ax.set_ylabel("lr")
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.4f}", ha="center", va="center", color="white", fontsize=8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(recall_key)

    fig.suptitle("LightGCN Hyperparameter Grid Search (CV Mean)", fontsize=13)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    code_dir = args.lightgcn_code_dir if args.lightgcn_code_dir.is_absolute() else (repo_root / args.lightgcn_code_dir)
    data_root = args.lightgcn_data_dir if args.lightgcn_data_dir.is_absolute() else (repo_root / args.lightgcn_data_dir)
    dataset_dir = data_root / args.dataset

    csv_path = args.csv_path
    if csv_path is None:
        csv_path = repo_root / "mixing" / "hyperparam" / "results" / f"{args.dataset}.csv"
    elif not csv_path.is_absolute():
        csv_path = repo_root / csv_path

    plot_path = args.plot_path
    if plot_path is None:
        plot_path = repo_root / "mixing" / "hyperparm" / "plots" / f"{args.dataset}.pdf"
    elif not plot_path.is_absolute():
        plot_path = repo_root / plot_path

    if not code_dir.exists():
        raise FileNotFoundError(f"LightGCN code directory does not exist: {code_dir}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    num_folds = args.num_folds if args.num_folds is not None else infer_num_folds(dataset_dir)
    if num_folds <= 0:
        raise ValueError("num_folds must be positive.")

    topks_list = [int(x.strip()) for x in args.topks.strip("[]").split(",") if x.strip()]
    if not topks_list:
        raise ValueError("--topks must contain at least one integer k value.")
    recall_key = f"recall@{topks_list[0]}"
    print(f"Running grid search on dataset={args.dataset}, folds={num_folds}, recall metric={recall_key}")

    rows: list[dict] = []
    for model_name in MODELS:
        for lr in GRID_VALUES:
            for decay in GRID_VALUES:
                fold_scores: list[float] = []
                fold_times: list[float] = []
                failed = False
                for fold_idx in range(num_folds):
                    print(
                        f"[run] model={model_name} lr={lr:g} decay={decay:g} "
                        f"fold={fold_idx}/{num_folds-1}"
                    )
                    try:
                        parsed, elapsed = run_one(
                            code_dir=code_dir,
                            dataset=args.dataset,
                            model_name=model_name,
                            lr=lr,
                            decay=decay,
                            fold_idx=fold_idx,
                            epochs=args.epochs,
                            layer=args.layer,
                            recdim=args.recdim,
                            topks=args.topks,
                            bpr_batch=args.bpr_batch,
                            seed=args.seed + fold_idx,
                        )
                    except Exception as exc:
                        failed = True
                        print(f"[error] {exc}", file=sys.stderr)
                        if args.fail_fast:
                            raise
                        break

                    if recall_key not in parsed["recall"]:
                        raise KeyError(
                            f"Requested {recall_key} not found in parsed recall metrics: "
                            f"{sorted(parsed['recall'].keys())}."
                        )
                    fold_scores.append(float(parsed["recall"][recall_key]))
                    fold_times.append(float(elapsed))

                row = {
                    "dataset": args.dataset,
                    "model": model_name,
                    "lr": lr,
                    "decay": decay,
                    "num_folds": num_folds,
                    "recall_metric": recall_key,
                    "status": "ok" if (not failed and len(fold_scores) == num_folds) else "failed",
                    "recall_cv_mean": float(np.mean(fold_scores)) if fold_scores else np.nan,
                    "recall_cv_std": float(np.std(fold_scores)) if fold_scores else np.nan,
                    "fold_scores_json": json.dumps(fold_scores),
                    "fold_times_seconds_json": json.dumps(fold_times),
                    "total_time_seconds": float(np.sum(fold_times)) if fold_times else np.nan,
                }
                rows.append(row)
                print(
                    f"[done] model={model_name} lr={lr:g} decay={decay:g} "
                    f"status={row['status']} recall_mean={row['recall_cv_mean']}"
                )

    if not rows:
        raise RuntimeError("No grid search rows produced.")

    write_csv(csv_path, rows)
    ok_rows = [r for r in rows if r["status"] == "ok"]
    if not ok_rows:
        raise RuntimeError(f"All runs failed. CSV written to {csv_path}, no heatmap produced.")

    plot_heatmaps(plot_path, ok_rows, recall_key=recall_key)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved heatmap PDF: {plot_path}")


if __name__ == "__main__":
    main()
