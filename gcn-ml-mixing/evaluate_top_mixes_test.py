#!/usr/bin/env python3
"""Evaluate top validation mixes for a source group on the test split."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


LOSS_RE = re.compile(r"EPOCH\[(\d+)/(\d+)\]\s+loss([0-9]*\.?[0-9]+)-")
ARRAY_RE = r"array\(\[([^\]]+)\]\)"
METRICS_RE = re.compile(
    rf"\{{'precision':\s*{ARRAY_RE},\s*'recall':\s*{ARRAY_RE},\s*'ndcg':\s*{ARRAY_RE}\}}"
)
BASELINE_TYPES = {"no_augmentation", "proportional", "stratified"}
MODE_TOPK = "topk_and_baselines"
MODE_PROP_PERTURB = "proportional_perturbation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read validation sweep CSVs, select top-k mixes, and evaluate them "
            "on test split along with baselines."
        )
    )
    parser.add_argument("--feature-name", type=str, required=True, help="Feature key, e.g. Age.")
    parser.add_argument("--source-label", type=str, required=True, help="Source group label.")
    parser.add_argument("--top-k", type=int, default=10, help="How many top validation mixes to test.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[MODE_TOPK, MODE_PROP_PERTURB],
        default=MODE_TOPK,
        help=(
            f"Run mode: '{MODE_TOPK}' evaluates baselines + top-k validation mixes; "
            f"'{MODE_PROP_PERTURB}' evaluates only proportional perturbation trials."
        ),
    )
    parser.add_argument(
        "--proportional-perturbation-trials",
        type=int,
        default=0,
        help=(
            "Number of proportional_perturbation trials to evaluate when "
            f"--mode={MODE_PROP_PERTURB}."
        ),
    )
    parser.add_argument(
        "--perturbation-scale",
        type=float,
        default=0.1,
        help=(
            "Scale factor applied after centering delta in proportional_perturbation mode. "
            "alpha_mix = x + scale * (delta - mean(delta))."
        ),
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path("gcn-ml-mixing/validation"),
        help="Directory containing validation CSV outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gcn-ml-mixing/test"),
        help="Directory for test evaluation CSV outputs.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing ml-1m data for label regeneration fallback.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("LightGCN/data/ml-1m"),
        help="Directory containing train/val/test and user_labels.pkl for ml-1m.",
    )
    parser.add_argument(
        "--lightgcn-code-dir",
        type=Path,
        default=Path("LightGCN/code"),
        help="Directory containing main.py",
    )
    parser.add_argument("--topks", type=str, default="[20]", help="LightGCN --topks value.")
    parser.add_argument("--recdim", type=int, default=64, help="LightGCN --recdim value.")
    parser.add_argument("--lr", type=float, default=0.001, help="LightGCN --lr value.")
    parser.add_argument("--decay", type=float, default=1e-4, help="LightGCN --decay value.")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs per test run.")
    parser.add_argument("--layer", type=int, default=1, help="LightGCN layer count.")
    parser.add_argument("--seed", type=int, default=2020, help="Base seed for test evaluations.")
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="",
        help="Validation metric column for top-k selection (default: first recall@k column).",
    )
    return parser.parse_args()


def _resolve(repo_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _parse_alpha_mix(v: str) -> dict[str, float]:
    try:
        parsed = json.loads(v)
    except Exception:
        parsed = ast.literal_eval(v)
    return {str(k): float(val) for k, val in parsed.items()}


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
        raise FileNotFoundError(
            "Missing dataset files for test evaluation. "
            "Run make-splits first. Missing: " + ", ".join(missing)
        )


def _load_validation_runs(validation_dir: Path, feature_name: str, source_label: str) -> pd.DataFrame:
    pattern = f"*__feature-{feature_name}__source-{source_label}.csv"
    files = sorted(validation_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No validation CSV files found in {validation_dir} matching {pattern}"
        )

    frames = []
    for fp in files:
        df = pd.read_csv(fp)
        if "feature_name" in df.columns:
            df = df[df["feature_name"].astype(str) == str(feature_name)]
        if "source_label" in df.columns:
            df = df[df["source_label"].astype(str) == str(source_label)]
        if not df.empty:
            df["validation_csv"] = fp.name
            frames.append(df)
    if not frames:
        raise ValueError("Validation CSV files found, but no rows matched feature/source.")
    return pd.concat(frames, ignore_index=True)


def _select_metric_column(df: pd.DataFrame, selection_metric: str) -> str:
    if selection_metric:
        if selection_metric not in df.columns:
            raise KeyError(f"--selection-metric '{selection_metric}' not found in validation CSV.")
        return selection_metric
    recall_cols = sorted(
        [c for c in df.columns if c.startswith("recall@")],
        key=lambda x: int(x.split("@")[1]),
    )
    if not recall_cols:
        raise KeyError("No recall@k column found in validation CSV.")
    return recall_cols[0]


def _compute_baselines_from_labels(labels_path: Path, feature_name: str, source_label: str) -> list[dict]:
    with labels_path.open("rb") as f:
        labels = pickle.load(f)
    if feature_name not in labels:
        raise KeyError(f"Feature '{feature_name}' not in labels pickle.")
    label_to_users_raw = labels[feature_name]
    label_to_users = {str(k): v for k, v in label_to_users_raw.items()}
    if source_label not in label_to_users:
        raise KeyError(
            f"Source label '{source_label}' not in feature '{feature_name}'. "
            f"Available: {sorted(label_to_users.keys())}"
        )

    aug_labels = sorted([l for l in label_to_users.keys() if l != source_label])
    n_aug = float(sum(len(label_to_users[l]) for l in aug_labels))
    proportional = {l: float(len(label_to_users[l]) / n_aug) for l in aug_labels}
    stratified = {l: float(1.0 / len(aug_labels)) for l in aug_labels}
    return [
        {"trial_type": "no_augmentation", "alpha_aug": 0.0, "alpha_mix": proportional},
        {"trial_type": "proportional", "alpha_aug": 1.0, "alpha_mix": proportional},
        {"trial_type": "stratified", "alpha_aug": 1.0, "alpha_mix": stratified},
    ]


def _sample_proportional_perturbation_mix(
    proportional_mix: dict[str, float],
    perturbation_scale: float,
    rng: np.random.Generator,
    max_attempts: int = 10000,
) -> dict[str, float]:
    labels = sorted(proportional_mix.keys())
    x = np.array([float(proportional_mix[l]) for l in labels], dtype=float)

    for _ in range(max_attempts):
        delta = rng.uniform(0.0, 1.0, size=len(labels))
        delta_centered_scaled = (delta - float(np.mean(delta))) * float(perturbation_scale)
        mixed = x + delta_centered_scaled
        if np.all(mixed >= 0.0):
            # Keep on simplex after numerical noise.
            mixed = mixed / float(np.sum(mixed))
            return {label: float(v) for label, v in zip(labels, mixed)}

    raise RuntimeError(
        "Failed to sample a valid proportional_perturbation alpha_mix with non-negative "
        f"components after {max_attempts} attempts."
    )


def _run_lightgcn_once(
    code_dir: Path,
    labels_pkl: Path,
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
        f"--decay={decay}",
        f"--lr={lr}",
        f"--layer={layer}",
        f"--seed={seed}",
        "--dataset=ml-1m",
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


def _load_completed_test_runs(
    csv_path: Path, feature_name: str, source_label: str
) -> set[tuple[str, str, str]]:
    """
    Return completed run keys as:
      (trial_type, alpha_aug_str, alpha_mix_json_sorted)
    scoped to feature/source.
    """
    completed: set[tuple[str, str, str]] = set()
    if not csv_path.is_file():
        print(f"No existing test CSV at {csv_path}. Starting fresh.")
        return completed

    print(f"Found existing test CSV at {csv_path}. Scanning for completed rows...")
    rows_total = 0
    rows_matched = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_total += 1
            if str(row.get("feature_name", "")) != str(feature_name):
                continue
            if str(row.get("source_label", "")) != str(source_label):
                continue
            rows_matched += 1
            trial_type = str(row.get("trial_type", ""))
            alpha_aug = str(row.get("alpha_aug", ""))
            alpha_mix_json = str(row.get("alpha_mix_json", ""))
            completed.add((trial_type, alpha_aug, alpha_mix_json))

    print(
        "Test resume scan summary: "
        f"rows_total={rows_total}, rows_matched_feature_source={rows_matched}, "
        f"completed_keys={len(completed)}"
    )
    return completed


def _count_existing_proportional_perturbation_rows(
    csv_path: Path, feature_name: str, source_label: str, perturbation_scale: float
) -> int:
    if not csv_path.is_file():
        print(f"No existing test CSV at {csv_path} for perturbation resume. Starting fresh.")
        return 0

    count = 0
    rows_total = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_total += 1
            if str(row.get("feature_name", "")) != str(feature_name):
                continue
            if str(row.get("source_label", "")) != str(source_label):
                continue
            if str(row.get("trial_type", "")) != MODE_PROP_PERTURB:
                continue
            row_scale_raw = row.get("perturbation_scale", "")
            try:
                row_scale = float(row_scale_raw)
            except (TypeError, ValueError):
                continue
            if np.isclose(row_scale, perturbation_scale, atol=1e-12):
                count += 1

    print(
        "Perturbation resume scan summary: "
        f"rows_total={rows_total}, completed_for_feature_source_scale={count}"
    )
    return count


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    feature_name = str(args.feature_name)
    source_label = str(args.source_label)
    validation_dir = _resolve(repo_root, args.validation_dir)
    output_dir = _resolve(repo_root, args.output_dir)
    data_dir = _resolve(repo_root, args.data_dir)
    dataset_dir = _resolve(repo_root, args.dataset_dir)
    code_dir = _resolve(repo_root, args.lightgcn_code_dir)

    _ensure_dataset_files(dataset_dir)
    labels_pkl = _load_labels_or_build(repo_root, data_dir, dataset_dir)
    out_csv = output_dir / f"ml1m_test_eval__feature-{feature_name}__source-{source_label}.csv"
    print(f"Writing test evaluations to: {out_csv}")

    if args.mode == MODE_PROP_PERTURB:
        if args.proportional_perturbation_trials <= 0:
            raise ValueError(
                "--proportional-perturbation-trials must be > 0 when "
                f"--mode={MODE_PROP_PERTURB}."
            )
        if args.perturbation_scale < 0:
            raise ValueError("--perturbation-scale must be non-negative.")

        baseline_specs = _compute_baselines_from_labels(labels_pkl, feature_name, source_label)
        proportional_spec = next(b for b in baseline_specs if b["trial_type"] == "proportional")
        proportional_mix = proportional_spec["alpha_mix"]

        completed_count = _count_existing_proportional_perturbation_rows(
            out_csv,
            feature_name=feature_name,
            source_label=source_label,
            perturbation_scale=args.perturbation_scale,
        )
        requested = int(args.proportional_perturbation_trials)
        if completed_count >= requested:
            print(
                "Requested proportional_perturbation trials already completed for this "
                "feature/source/scale. Nothing to run."
            )
            return

        remaining = requested - completed_count
        print(
            f"Running {remaining} new {MODE_PROP_PERTURB} trials "
            f"(completed={completed_count}, requested={requested}, scale={args.perturbation_scale})."
        )

        rng = np.random.default_rng(args.seed)
        for _ in range(completed_count):
            _sample_proportional_perturbation_mix(
                proportional_mix=proportional_mix,
                perturbation_scale=args.perturbation_scale,
                rng=rng,
            )

        for offset in range(remaining):
            perturbation_trial = completed_count + offset + 1
            alpha_mix = _sample_proportional_perturbation_mix(
                proportional_mix=proportional_mix,
                perturbation_scale=args.perturbation_scale,
                rng=rng,
            )
            alpha_mix_json = json.dumps(alpha_mix, sort_keys=True)

            print(
                f"=== Test Eval {offset + 1}/{remaining}: {MODE_PROP_PERTURB} "
                f"(trial={perturbation_trial}) ==="
            )
            metrics = _run_lightgcn_once(
                code_dir=code_dir,
                labels_pkl=labels_pkl,
                feature_name=feature_name,
                source_label=source_label,
                alpha_aug=1.0,
                alpha_mix=alpha_mix,
                topks_str=args.topks,
                recdim=args.recdim,
                lr=args.lr,
                decay=args.decay,
                epochs=args.epochs,
                layer=args.layer,
                seed=args.seed + perturbation_trial,
            )

            row = {
                "trial_type": MODE_PROP_PERTURB,
                "feature_name": feature_name,
                "source_label": source_label,
                "alpha_aug": 1.0,
                "alpha_mix_json": alpha_mix_json,
                "validation_rank": "",
                "validation_metric_name": "",
                "validation_metric_value": "",
                "validation_trial_type": "proportional",
                "perturbation_scale": float(args.perturbation_scale),
                "perturbation_trial": perturbation_trial,
                "final_epoch": metrics["final_epoch"],
                "final_loss": metrics["final_loss"],
                "training_time_seconds": metrics["training_time_seconds"],
            }
            row.update(metrics["precision"])
            row.update(metrics["recall"])
            row.update(metrics["ndcg"])
            _append_csv(out_csv, row)
            print(f"Appended: {MODE_PROP_PERTURB} trial={perturbation_trial}")

        return

    val_df = _load_validation_runs(validation_dir, feature_name, source_label)
    metric_col = _select_metric_column(val_df, args.selection_metric)

    non_baseline = val_df[~val_df["trial_type"].isin(BASELINE_TYPES)].copy()
    if non_baseline.empty:
        raise ValueError("No non-baseline validation rows found for top-k selection.")

    group_cols = ["source_label", "trial_type", "alpha_aug", "alpha_mix_json"]
    metric_cols = [
        c
        for c in val_df.columns
        if c.startswith("precision@") or c.startswith("recall@") or c.startswith("ndcg@")
    ]
    agg_map = {c: "mean" for c in metric_cols}
    if "final_loss" in val_df.columns:
        agg_map["final_loss"] = "mean"
    cv_agg = non_baseline.groupby(group_cols, as_index=False).agg(agg_map)

    cv_agg = cv_agg.sort_values(metric_col, ascending=False)
    top_df = cv_agg.head(args.top_k).copy()

    baseline_specs = _compute_baselines_from_labels(labels_pkl, feature_name, source_label)
    top_specs = []
    for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
        top_specs.append(
            {
                "trial_type": f"top_{rank}",
                "alpha_aug": float(row["alpha_aug"]),
                "alpha_mix": _parse_alpha_mix(row["alpha_mix_json"]),
                "validation_rank": rank,
                "validation_metric_name": metric_col,
                "validation_metric_value": float(row[metric_col]),
                "validation_trial_type": str(row.get("trial_type", "")),
            }
        )

    all_specs = []
    for b in baseline_specs:
        b.update(
            {
                "validation_rank": "",
                "validation_metric_name": metric_col,
                "validation_metric_value": "",
                "validation_trial_type": b["trial_type"],
            }
        )
        all_specs.append(b)
    all_specs.extend(top_specs)

    completed = _load_completed_test_runs(
        out_csv, feature_name=feature_name, source_label=source_label
    )

    for i, spec in enumerate(all_specs, start=1):
        key = (
            str(spec["trial_type"]),
            str(spec["alpha_aug"]),
            json.dumps(spec["alpha_mix"], sort_keys=True),
        )
        if key in completed:
            print(
                f"=== Test Eval {i}/{len(all_specs)}: {spec['trial_type']} "
                "(already completed, skipping) ==="
            )
            continue

        print(f"=== Test Eval {i}/{len(all_specs)}: {spec['trial_type']} ===")
        metrics = _run_lightgcn_once(
            code_dir=code_dir,
            labels_pkl=labels_pkl,
            feature_name=feature_name,
            source_label=source_label,
            alpha_aug=spec["alpha_aug"],
            alpha_mix=spec["alpha_mix"],
            topks_str=args.topks,
            recdim=args.recdim,
            lr=args.lr,
            decay=args.decay,
            epochs=args.epochs,
            layer=args.layer,
            seed=args.seed + i,
        )

        row = {
            "trial_type": spec["trial_type"],
            "feature_name": feature_name,
            "source_label": source_label,
            "alpha_aug": spec["alpha_aug"],
            "alpha_mix_json": key[2],
            "validation_rank": spec["validation_rank"],
            "validation_metric_name": spec["validation_metric_name"],
            "validation_metric_value": spec["validation_metric_value"],
            "validation_trial_type": spec["validation_trial_type"],
            "perturbation_scale": "",
            "perturbation_trial": "",
            "final_epoch": metrics["final_epoch"],
            "final_loss": metrics["final_loss"],
            "training_time_seconds": metrics["training_time_seconds"],
        }
        row.update(metrics["precision"])
        row.update(metrics["recall"])
        row.update(metrics["ndcg"])
        _append_csv(out_csv, row)
        completed.add(key)
        print(f"Appended: {spec['trial_type']}")


if __name__ == "__main__":
    main()
