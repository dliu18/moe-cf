#!/usr/bin/env python3
"""Train proportional vs stratified baselines across embedding dimensions."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path


LOSS_RE = re.compile(r"EPOCH\[(\d+)/(\d+)\]\s+loss([0-9]*\.?[0-9]+)-")
ARRAY_RE = r"array\(\[([^\]]+)\]\)"
METRICS_RE = re.compile(
    rf"\{{'precision':\s*{ARRAY_RE},\s*'recall':\s*{ARRAY_RE},\s*'ndcg':\s*{ARRAY_RE}\}}"
)
REC_DIMS = [2**i for i in range(1, 9)]  # 2..256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare proportional vs stratified baseline performance over embedding "
            "dimensions for a dataset/model/source group."
        )
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. ml-1m, lastfm-asia.")
    parser.add_argument("--model", type=str, default="lgn", choices=["lgn", "mf"])
    parser.add_argument("--source-label", type=str, required=True, help="Source group label.")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--decay", type=float, required=True)
    parser.add_argument("--trials", type=int, default=5, help="Number of random-seed trials per baseline/recdim.")
    parser.add_argument("--seed", type=int, default=2020, help="Base seed.")
    parser.add_argument("--topks", type=str, default="", help="Override LightGCN --topks. Default depends on dataset.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--feature-name", type=str, default="", help="Override feature name. Default depends on dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Default: LightGCN/data/<dataset>")
    parser.add_argument("--lightgcn-code-dir", type=Path, default=Path("LightGCN/code"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mixing/experiments/prop_vs_stratified/results"),
        help="Directory for result CSVs.",
    )
    return parser.parse_args()


def _resolve(repo_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_metrics(stdout_text: str, topks: list[int]) -> dict:
    loss_matches = list(LOSS_RE.finditer(stdout_text))
    if not loss_matches:
        raise RuntimeError("Could not parse final loss from LightGCN output.")
    final_loss = float(loss_matches[-1].group(3))
    final_epoch = int(loss_matches[-1].group(1))

    metric_matches = list(METRICS_RE.finditer(stdout_text))
    if not metric_matches:
        raise RuntimeError("Could not parse ranking metrics from LightGCN output.")
    m = metric_matches[-1]
    precision_vals = _parse_float_list(m.group(1))
    recall_vals = _parse_float_list(m.group(2))
    ndcg_vals = _parse_float_list(m.group(3))

    if not (len(precision_vals) == len(recall_vals) == len(ndcg_vals) == len(topks)):
        raise RuntimeError("Metric length mismatch vs --topks.")

    return {
        "final_epoch": final_epoch,
        "final_loss": final_loss,
        "precision": {f"precision@{k}": v for k, v in zip(topks, precision_vals)},
        "recall": {f"recall@{k}": v for k, v in zip(topks, recall_vals)},
        "ndcg": {f"ndcg@{k}": v for k, v in zip(topks, ndcg_vals)},
    }


def _load_labels_or_build(repo_root: Path, data_dir: Path, dataset_dir: Path) -> Path:
    labels_path = dataset_dir / "user_labels.pkl"
    if labels_path.is_file():
        return labels_path

    if dataset_dir.name != "ml-1m":
        raise FileNotFoundError(
            f"Missing {labels_path}. For non-ml-1m datasets, create labels during preprocessing first."
        )

    sys.path.insert(0, str(repo_root))
    from loaders.movielens import movielens  # pylint: disable=import-error

    ml = movielens(min_ratings=0, min_users=0, binary=True, data_dir=str(data_dir) + "/")
    labels = {"Gender": ml.get_user_labels("Gender"), "Age": ml.get_user_labels("Age")}
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("wb") as f:
        pickle.dump(labels, f)
    return labels_path


def _ensure_dataset_files(dataset_dir: Path) -> None:
    required = [dataset_dir / "train.txt", dataset_dir / "val.txt", dataset_dir / "test.txt"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing dataset files: " + ", ".join(missing))


def _compute_baselines_from_labels(labels_path: Path, feature_name: str, source_label: str) -> list[dict]:
    with labels_path.open("rb") as f:
        labels = pickle.load(f)

    if feature_name not in labels:
        raise KeyError(f"Feature '{feature_name}' not found in labels. Available: {sorted(labels.keys())}")

    label_to_users = {str(k): v for k, v in labels[feature_name].items()}
    if source_label not in label_to_users:
        raise KeyError(
            f"Source label '{source_label}' not found in feature '{feature_name}'. "
            f"Available: {sorted(label_to_users.keys())}"
        )

    aug_labels = sorted([l for l in label_to_users if l != source_label])
    n_aug = float(sum(len(label_to_users[l]) for l in aug_labels))
    proportional = {l: float(len(label_to_users[l]) / n_aug) for l in aug_labels}
    stratified = {l: float(1.0 / len(aug_labels)) for l in aug_labels}

    return [
        {"trial_type": "proportional", "alpha_aug": 1.0, "alpha_mix": proportional},
        {"trial_type": "stratified", "alpha_aug": 1.0, "alpha_mix": stratified},
    ]


def _run_lightgcn_once(
    code_dir: Path,
    labels_pkl: Path,
    dataset_name: str,
    model_name: str,
    feature_name: str,
    source_label: str,
    alpha_aug: float,
    alpha_mix: dict[str, float],
    topks_str: str,
    recdim: int,
    lr: float,
    decay: float,
    epochs: int,
    layer: int,
    seed: int,
) -> dict:
    cmd = [
        "python",
        "main.py",
        "--bpr_batch=1_000_000",
        f"--model={model_name}",
        f"--decay={decay}",
        f"--lr={lr}",
        f"--layer={layer}",
        f"--seed={seed}",
        f"--dataset={dataset_name}",
        f"--topks={topks_str}",
        f"--recdim={recdim}",
        "--epochs",
        str(epochs),
        "--tensorboard=0",
        "--eval_split=test",
        "--group_mixing=1",
        f"--group_labels_pkl={labels_pkl}",
        f"--feature_name={feature_name}",
        f"--source_group={source_label}",
        f"--alpha_aug={alpha_aug}",
        f"--alpha_mix={json.dumps(alpha_mix, sort_keys=True)}",
    ]

    print(" ".join(cmd))
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(code_dir), capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - t0

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"LightGCN run failed with exit code {result.returncode}.")

    topks = [int(x.strip()) for x in topks_str.strip("[]").split(",") if x.strip()]
    metrics = _parse_metrics(result.stdout, topks)
    metrics["training_time_seconds"] = elapsed
    return metrics


def _append_csv(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.is_file()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def _load_completed_runs(csv_path: Path, source_label: str) -> set[tuple[str, str, int, int]]:
    """
    Resume key format:
      (source_label, trial_type, recdim, trial_idx)
    """
    completed: set[tuple[str, str, int, int]] = set()
    if not csv_path.is_file():
        print(f"No existing results CSV at {csv_path}. Starting fresh.")
        return completed

    rows_total = 0
    rows_loaded = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_total += 1
            if str(row.get("source_label", "")) != str(source_label):
                continue
            trial_type = str(row.get("trial_type", ""))
            try:
                recdim = int(row.get("recdim", ""))
                trial_idx = int(row.get("trial_idx", ""))
            except (TypeError, ValueError):
                continue
            completed.add((str(source_label), trial_type, recdim, trial_idx))
            rows_loaded += 1

    print(
        "Resume scan summary: "
        f"rows_total={rows_total}, rows_loaded_for_source={rows_loaded}, "
        f"completed_keys={len(completed)}"
    )
    return completed


def _default_feature_and_topks(dataset: str) -> tuple[str, str]:
    if dataset == "lastfm-asia":
        return "Country", "[100]"
    if dataset in {"movielens", "ml-1m"}:
        return "Age", "[20]"
    return "Age", "[20]"


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]

    dataset = str(args.dataset)
    model = str(args.model).lower()
    source_label = str(args.source_label)

    feature_default, topks_default = _default_feature_and_topks(dataset)
    feature_name = str(args.feature_name) if args.feature_name else feature_default
    topks = str(args.topks) if args.topks else topks_default

    dataset_dir_raw = args.dataset_dir if args.dataset_dir is not None else Path("LightGCN") / "data" / dataset
    code_dir = _resolve(repo_root, args.lightgcn_code_dir)
    data_dir = _resolve(repo_root, args.data_dir)
    dataset_dir = _resolve(repo_root, dataset_dir_raw)
    output_dir = _resolve(repo_root, args.output_dir)

    _ensure_dataset_files(dataset_dir)
    labels_pkl = _load_labels_or_build(repo_root, data_dir, dataset_dir)

    out_csv = output_dir / f"{dataset}__{model}__source-{source_label}.csv"
    baselines = _compute_baselines_from_labels(labels_pkl, feature_name, source_label)
    completed = _load_completed_runs(out_csv, source_label=source_label)

    print(f"Writing results to: {out_csv}")
    print(f"Dataset={dataset}, Model={model}, Source={source_label}, Feature={feature_name}, Topks={topks}")

    for recdim in REC_DIMS:
        for baseline in baselines:
            for trial_idx in range(1, int(args.trials) + 1):
                key = (str(source_label), str(baseline["trial_type"]), int(recdim), int(trial_idx))
                if key in completed:
                    print(
                        f"Skipping completed run: source={source_label} "
                        f"baseline={baseline['trial_type']} recdim={recdim} trial_idx={trial_idx}"
                    )
                    continue
                seed = int(args.seed) + recdim * 100 + trial_idx
                print(
                    f"=== recdim={recdim} baseline={baseline['trial_type']} "
                    f"trial={trial_idx}/{args.trials} seed={seed} ==="
                )
                metrics = _run_lightgcn_once(
                    code_dir=code_dir,
                    labels_pkl=labels_pkl,
                    dataset_name=dataset,
                    model_name=model,
                    feature_name=feature_name,
                    source_label=source_label,
                    alpha_aug=float(baseline["alpha_aug"]),
                    alpha_mix=baseline["alpha_mix"],
                    topks_str=topks,
                    recdim=recdim,
                    lr=float(args.lr),
                    decay=float(args.decay),
                    epochs=int(args.epochs),
                    layer=int(args.layer),
                    seed=seed,
                )

                row = {
                    "dataset": dataset,
                    "model": model,
                    "source_label": source_label,
                    "feature_name": feature_name,
                    "trial_type": baseline["trial_type"],
                    "alpha_aug": baseline["alpha_aug"],
                    "alpha_mix_json": json.dumps(baseline["alpha_mix"], sort_keys=True),
                    "recdim": recdim,
                    "seed": seed,
                    "trial_idx": trial_idx,
                    "lr": float(args.lr),
                    "decay": float(args.decay),
                    "topks": topks,
                    "final_epoch": metrics["final_epoch"],
                    "final_loss": metrics["final_loss"],
                    "training_time_seconds": metrics["training_time_seconds"],
                }
                row.update(metrics["precision"])
                row.update(metrics["recall"])
                row.update(metrics["ndcg"])
                _append_csv(out_csv, row)
                completed.add(key)


if __name__ == "__main__":
    main()
