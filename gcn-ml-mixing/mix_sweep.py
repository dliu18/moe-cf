#!/usr/bin/env python3
"""Run baseline + random group-mixing trials on ml-1m validation split."""

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

import numpy as np

from create_ml1m_lightgcn_splits import ensure_splits_and_labels, _resolve


LOSS_RE = re.compile(r"EPOCH\[(\d+)/(\d+)\]\s+loss([0-9]*\.?[0-9]+)-")
ARRAY_RE = r"array\(\[([^\]]+)\]\)"
METRICS_RE = re.compile(
    rf"\{{'precision':\s*{ARRAY_RE},\s*'recall':\s*{ARRAY_RE},\s*'ndcg':\s*{ARRAY_RE}\}}"
)

SAMPLER_DIRICHLET = "dirichlet"
SAMPLER_PROP_PERTURB = "proportional_perturbation"
SAMPLER_LOGNORMAL = "lognormal"
SAMPLERS = [SAMPLER_DIRICHLET, SAMPLER_PROP_PERTURB, SAMPLER_LOGNORMAL]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ml-1m LightGCN data-mixing sweep.")
    parser.add_argument("--feature-name", type=str, required=True, help="Feature key, e.g. Age.")
    parser.add_argument("--source-label", type=str, required=True, help="Source group label.")
    parser.add_argument("--trials", type=int, default=10, help="Number of random trials.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for trials.")
    parser.add_argument("--num-folds", type=int, default=5, help="Number of validation folds.")
    parser.add_argument("--topks", type=str, default="[20]", help="LightGCN --topks value.")
    parser.add_argument("--recdim", type=int, default=64, help="LightGCN --recdim value.")
    parser.add_argument("--lr", type=float, default=0.001, help="LightGCN --lr value.")
    parser.add_argument("--decay", type=float, default=1e-4, help="LightGCN --decay value.")
    parser.add_argument(
        "--model",
        type=str,
        default="lgn",
        choices=["lgn", "mf"],
        help="LightGCN --model value.",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs per trial.")
    parser.add_argument("--layer", type=int, default=1, help="LightGCN layer count.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("gcn-ml-mixing/validation/ml1m_mixing_results.csv"),
        help="CSV output path (appends trial-by-trial).",
    )
    parser.add_argument(
        "--lightgcn-code-dir",
        type=Path,
        default=Path("LightGCN/code"),
        help="Directory containing LightGCN main.py.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing ml-1m/ratings.dat.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("LightGCN/data/ml-1m"),
        help="Output directory for LightGCN ml-1m files.",
    )

    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.7,
        help="Dirichlet concentration for the dirichlet sampler.",
    )
    parser.add_argument(
        "--proportional-perturbation-scale",
        type=float,
        default=0.25,
        help=(
            "Scale for proportional_perturbation sampler: "
            "x + scale * (delta - mean(delta)), delta~Uniform([0,1])."
        ),
    )
    parser.add_argument(
        "--lognormal-mu",
        type=float,
        default=0.0,
        help="Mu parameter for log-normal alpha' sampler.",
    )
    parser.add_argument(
        "--lognormal-sigma",
        type=float,
        default=0.5,
        help="Sigma parameter for log-normal alpha' sampler.",
    )
    parser.add_argument(
        "--max-sample-attempts",
        type=int,
        default=10000,
        help="Maximum retries when re-sampling invalid alpha_mix.",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Only sample alpha_mixes and generate diagnostics plot; skip LightGCN runs.",
    )
    parser.add_argument(
        "--smoke-plot-path",
        type=Path,
        default=Path("gcn-ml-mixing/validation/mix_sweep_smoke.png"),
        help="Output image path for smoke-test alpha_mix distribution plot.",
    )

    return parser.parse_args()


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_lgn_stdout(stdout_text: str, topks: list[int]) -> dict:
    loss_matches = list(LOSS_RE.finditer(stdout_text))
    if not loss_matches:
        raise RuntimeError("Could not parse final loss from LightGCN stdout.")
    final_loss = float(loss_matches[-1].group(3))
    final_epoch = int(loss_matches[-1].group(1))

    metric_matches = list(METRICS_RE.finditer(stdout_text))
    if not metric_matches:
        raise RuntimeError("Could not parse validation metrics from LightGCN stdout.")
    m = metric_matches[-1]
    precision_vals = _parse_float_list(m.group(1))
    recall_vals = _parse_float_list(m.group(2))
    ndcg_vals = _parse_float_list(m.group(3))
    if not (len(precision_vals) == len(recall_vals) == len(ndcg_vals) == len(topks)):
        raise RuntimeError("Metric length mismatch vs topks.")
    return {
        "final_epoch": final_epoch,
        "final_loss": final_loss,
        "precision": {f"precision@{k}": v for k, v in zip(topks, precision_vals)},
        "recall": {f"recall@{k}": v for k, v in zip(topks, recall_vals)},
        "ndcg": {f"ndcg@{k}": v for k, v in zip(topks, ndcg_vals)},
    }


def run_lightgcn_trial(
    code_dir: Path,
    labels_pkl: Path,
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
    val_split_idx: int,
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
        "--dataset=ml-1m",
        f"--topks={topks_str}",
        f"--recdim={recdim}",
        "--epochs",
        str(epochs),
        "--tensorboard=0",
        "--eval_split=val",
        f"--val_split_idx={val_split_idx}",
        "--group_mixing=1",
        f"--group_labels_pkl={labels_pkl}",
        f"--feature_name={feature_name}",
        f"--source_group={source_label}",
        f"--alpha_aug={alpha_aug}",
        f"--alpha_mix={json.dumps(alpha_mix, sort_keys=True)}",
    ]
    print(" ".join(cmd))
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(code_dir), capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - start
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"LightGCN trial failed with exit code {result.returncode}.")

    topks = [int(x.strip()) for x in topks_str.strip("[]").split(",") if x.strip()]
    parsed = parse_lgn_stdout(result.stdout, topks)
    parsed["training_time_seconds"] = elapsed
    return parsed


def append_csv_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_completed_runs(
    csv_path: Path, feature_name: str, source_label: str, model_name: str
) -> set[tuple[int, int]]:
    """
    Return completed (trial_index, cv_fold) pairs already present in CSV
    for the same feature/source.
    """
    completed: set[tuple[int, int]] = set()
    if not csv_path.is_file():
        print(f"No existing CSV found at {csv_path}. Starting a fresh sweep.")
        return completed

    print(f"Found existing CSV at {csv_path}. Checking completed runs for resume...")
    rows_total = 0
    rows_matched = 0
    rows_malformed = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_total += 1
            if str(row.get("feature_name", "")) != str(feature_name):
                continue
            if str(row.get("source_label", "")) != str(source_label):
                continue
            # Backward compatibility: legacy rows may not have "model"; treat them as lgn.
            row_model = str(row.get("model", "lgn")).strip().lower()
            if row_model != str(model_name).strip().lower():
                continue
            rows_matched += 1
            try:
                trial_idx = int(row["trial_index"])
                fold_idx = int(row["cv_fold"])
            except Exception:
                # Ignore malformed/legacy rows.
                rows_malformed += 1
                continue
            completed.add((trial_idx, fold_idx))
    print(
        "Resume scan summary: "
        f"rows_total={rows_total}, rows_matched_feature_source={rows_matched}, "
        f"rows_malformed={rows_malformed}, completed_pairs={len(completed)}"
    )
    return completed


def ensure_feature_source_in_filename(csv_path: Path, feature_name: str, source_label: str) -> Path:
    feature_token = feature_name.replace("/", "-").replace(" ", "_")
    source_token = str(source_label).replace("/", "-").replace(" ", "_")
    stem = csv_path.stem
    if feature_token in stem and source_token in stem:
        return csv_path
    return csv_path.with_name(f"{stem}__feature-{feature_token}__source-{source_token}{csv_path.suffix}")


def _max_indices(split_path: Path) -> tuple[int, int]:
    max_user = -1
    max_item = -1
    with split_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            uid = int(parts[0])
            items = [int(x) for x in parts[1:]]
            if not items:
                raise AssertionError(f"{split_path.name} has a user with zero items: {uid}")
            max_user = max(max_user, uid)
            max_item = max(max_item, max(items))
    if max_user < 0 or max_item < 0:
        raise AssertionError(f"{split_path.name} appears empty.")
    return max_user, max_item


def assert_split_index_alignment(output_dir: Path) -> None:
    split_files = [output_dir / "train_full.txt"]
    alias_train = output_dir / "train.txt"
    if alias_train.is_file():
        split_files.append(alias_train)
    fold_files = sorted(output_dir.glob("train_fold_*.txt"))
    split_files.extend(fold_files)
    if not fold_files:
        raise AssertionError("No training k-fold files found (train_fold_*.txt).")
    per_split = {p.name: _max_indices(p) for p in split_files}
    user_maxes = {name: pair[0] for name, pair in per_split.items()}
    item_maxes = {name: pair[1] for name, pair in per_split.items()}
    if len(set(user_maxes.values())) != 1:
        raise AssertionError(f"Max user index mismatch across splits: {user_maxes}")
    if len(set(item_maxes.values())) != 1:
        raise AssertionError(f"Max item index mismatch across splits: {item_maxes}")
    print(
        "Verified training-split index alignment: "
        f"user_max={next(iter(user_maxes.values()))}, item_max={next(iter(item_maxes.values()))}"
    )


def _is_valid_alpha_mix(alpha_mix_vec: np.ndarray, tol: float = 1e-8) -> bool:
    if alpha_mix_vec.ndim != 1:
        return False
    if np.any(alpha_mix_vec < -tol):
        return False
    return abs(float(alpha_mix_vec.sum()) - 1.0) <= 1e-6


def _normalize_nonnegative(alpha_mix_vec: np.ndarray) -> np.ndarray:
    clipped = np.where(alpha_mix_vec < 0.0, 0.0, alpha_mix_vec)
    denom = float(clipped.sum())
    if denom <= 0.0:
        return clipped
    return clipped / denom


def _sample_dirichlet(rng: np.random.Generator, d: int, alpha: float) -> np.ndarray:
    return rng.dirichlet(np.full(d, alpha, dtype=float))


def _sample_prop_perturb(
    rng: np.random.Generator, base_proportional: np.ndarray, scale: float
) -> np.ndarray:
    delta = rng.uniform(0.0, 1.0, size=base_proportional.shape[0])
    delta_p = (delta - float(delta.mean())) * float(scale)
    return base_proportional + delta_p


def _sample_lognormal_size_adjusted(
    rng: np.random.Generator,
    group_sizes: np.ndarray,
    mu: float,
    sigma: float,
) -> np.ndarray:
    alpha_prime = rng.lognormal(mean=mu, sigma=sigma, size=group_sizes.shape[0])
    denom = float(np.dot(group_sizes, alpha_prime))
    if denom <= 0.0:
        return np.full(group_sizes.shape[0], -1.0, dtype=float)
    return (group_sizes * alpha_prime) / denom


def _sample_alpha_mix_with_retry(
    sampler_name: str,
    rng: np.random.Generator,
    aug_labels: list[str],
    proportional_vec: np.ndarray,
    group_sizes: np.ndarray,
    dirichlet_alpha: float,
    pp_scale: float,
    lognormal_mu: float,
    lognormal_sigma: float,
    max_attempts: int,
) -> dict[str, float]:
    d = len(aug_labels)
    for _ in range(max_attempts):
        if sampler_name == SAMPLER_DIRICHLET:
            mix_vec = _sample_dirichlet(rng, d=d, alpha=dirichlet_alpha)
        elif sampler_name == SAMPLER_PROP_PERTURB:
            mix_vec = _sample_prop_perturb(rng, proportional_vec, scale=pp_scale)
        elif sampler_name == SAMPLER_LOGNORMAL:
            mix_vec = _sample_lognormal_size_adjusted(
                rng, group_sizes=group_sizes, mu=lognormal_mu, sigma=lognormal_sigma
            )
        else:
            raise ValueError(f"Unknown sampler '{sampler_name}'.")

        # Normalize before checks for numerical stability.
        mix_vec = _normalize_nonnegative(np.asarray(mix_vec, dtype=float))
        if _is_valid_alpha_mix(mix_vec):
            mix_dict = {label: float(mix_vec[j]) for j, label in enumerate(aug_labels)}
            assert abs(sum(mix_dict.values()) - 1.0) <= 1e-6
            assert all(v >= -1e-8 for v in mix_dict.values())
            return mix_dict

    raise RuntimeError(
        f"Failed to sample a valid alpha_mix for sampler={sampler_name} after {max_attempts} attempts."
    )


def _allocate_sampler_counts(trials: int) -> dict[str, int]:
    base = trials // len(SAMPLERS)
    counts = {name: base for name in SAMPLERS}
    remainder = trials % len(SAMPLERS)
    for i in range(remainder):
        counts[SAMPLERS[i]] += 1
    return counts


def _make_sampled_specs(
    trials: int,
    rng: np.random.Generator,
    aug_labels: list[str],
    proportional: dict[str, float],
    group_sizes: np.ndarray,
    dirichlet_alpha: float,
    pp_scale: float,
    lognormal_mu: float,
    lognormal_sigma: float,
    max_attempts: int,
) -> list[tuple[str, float, dict[str, float]]]:
    counts = _allocate_sampler_counts(trials)
    print(f"Sampler allocation for {trials} trials: {counts}")

    proportional_vec = np.array([float(proportional[l]) for l in aug_labels], dtype=float)
    sampled_specs: list[tuple[str, float, dict[str, float]]] = []

    for sampler_name in SAMPLERS:
        n_sampler = counts[sampler_name]
        for idx in range(1, n_sampler + 1):
            alpha_aug = float(rng.uniform(0.5, 2.0))
            alpha_mix = _sample_alpha_mix_with_retry(
                sampler_name=sampler_name,
                rng=rng,
                aug_labels=aug_labels,
                proportional_vec=proportional_vec,
                group_sizes=group_sizes,
                dirichlet_alpha=dirichlet_alpha,
                pp_scale=pp_scale,
                lognormal_mu=lognormal_mu,
                lognormal_sigma=lognormal_sigma,
                max_attempts=max_attempts,
            )
            sampled_specs.append((f"random_{sampler_name}_{idx}", alpha_aug, alpha_mix))

    assert len(sampled_specs) == trials, (
        f"Sampled {len(sampled_specs)} alpha_mixes but expected exactly {trials}."
    )
    return sampled_specs


def _make_smoke_plot(
    sampled_specs: list[tuple[str, float, dict[str, float]]],
    aug_labels: list[str],
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise RuntimeError(
            "Smoke-test plotting requires matplotlib and seaborn installed."
        ) from exc

    import math

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    sns.set_style("whitegrid")

    # Collect values by sampler and component.
    rows = []
    for trial_type, _, alpha_mix in sampled_specs:
        if trial_type.startswith("random_dirichlet_"):
            sampler = SAMPLER_DIRICHLET
        elif trial_type.startswith("random_proportional_perturbation_"):
            sampler = SAMPLER_PROP_PERTURB
        elif trial_type.startswith("random_lognormal_"):
            sampler = SAMPLER_LOGNORMAL
        else:
            continue
        for j, g in enumerate(aug_labels):
            rows.append((sampler, g, float(alpha_mix[g]), j))

    if not rows:
        raise RuntimeError("No sampled rows found to plot in smoke-test mode.")

    ncols = len(aug_labels)
    nrows = len(SAMPLERS)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.8 * max(1, ncols), 3.0 * max(1, nrows)),
        sharex=True,
        sharey=True,
    )
    axes = np.array(axes).reshape(nrows, ncols)

    color_map = {
        SAMPLER_DIRICHLET: "#2563EB",
        SAMPLER_PROP_PERTURB: "#D97706",
        SAMPLER_LOGNORMAL: "#059669",
    }

    for r, sampler in enumerate(SAMPLERS):
        for c, g in enumerate(aug_labels):
            ax = axes[r, c]
            vals = [v for s, gg, v, _ in rows if s == sampler and gg == g]
            vals = np.asarray(vals, dtype=float)
            if len(vals) > 1 and np.std(vals) > 0:
                sns.kdeplot(vals, ax=ax, fill=True, alpha=0.28, linewidth=1.6, color=color_map[sampler])
            elif len(vals) > 0:
                ax.hist(vals, bins=12, density=True, color=color_map[sampler], alpha=0.6)

            if r == 0:
                ax.set_title(f"Aug Group {g}")
            if c == 0:
                ax.set_ylabel(f"{sampler}\nDensity")
            else:
                ax.set_ylabel("Density")
            if r == nrows - 1:
                ax.set_xlabel(r"$\\alpha_g$")

    fig.suptitle("Smoke Test: Sampled alpha_mix Component Distributions by Sampler", y=1.02, fontsize=14)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved smoke-test figure: {output_path}")


def run_mixing_sweep(
    repo_root: Path,
    data_dir: Path,
    output_dir: Path,
    feature_name: str,
    source_label: str,
    trials: int,
    seed: int,
    topks: str,
    recdim: int,
    lr: float,
    decay: float,
    model_name: str,
    epochs: int,
    layer: int,
    csv_path: Path,
    code_dir: Path,
    num_folds: int,
    dirichlet_alpha: float,
    pp_scale: float,
    lognormal_mu: float,
    lognormal_sigma: float,
    max_sample_attempts: int,
    smoke_test: bool,
    smoke_plot_path: Path,
) -> None:
    labels_pkl = ensure_splits_and_labels(
        repo_root, data_dir, output_dir, seed=seed, num_folds=num_folds
    )
    assert_split_index_alignment(output_dir)

    labels = pickle.loads(labels_pkl.read_bytes())
    if feature_name not in labels:
        raise KeyError(f"Feature '{feature_name}' not found in labels pickle.")
    label_to_users = {str(k): v for k, v in labels[feature_name].items()}
    source_label = str(source_label)
    if source_label not in label_to_users:
        raise KeyError(
            f"Source label '{source_label}' not in feature '{feature_name}'. "
            f"Available: {sorted(label_to_users.keys())}"
        )

    aug_labels = sorted([l for l in label_to_users if l != source_label])
    if not aug_labels:
        raise ValueError("Need at least one augmentation group.")

    n_aug = float(sum(len(label_to_users[l]) for l in aug_labels))
    proportional = {l: len(label_to_users[l]) / n_aug for l in aug_labels}
    stratified = {l: 1.0 / len(aug_labels) for l in aug_labels}
    group_sizes = np.array([float(len(label_to_users[l])) for l in aug_labels], dtype=float)

    rng = np.random.default_rng(seed)
    sampled_specs = _make_sampled_specs(
        trials=trials,
        rng=rng,
        aug_labels=aug_labels,
        proportional=proportional,
        group_sizes=group_sizes,
        dirichlet_alpha=dirichlet_alpha,
        pp_scale=pp_scale,
        lognormal_mu=lognormal_mu,
        lognormal_sigma=lognormal_sigma,
        max_attempts=max_sample_attempts,
    )

    if smoke_test:
        print("Smoke-test mode enabled. Skipping LightGCN training/evaluation.")
        _make_smoke_plot(
            sampled_specs=sampled_specs,
            aug_labels=aug_labels,
            output_path=smoke_plot_path,
        )
        return

    all_specs = [
        ("no_augmentation", 0.0, proportional),
        ("proportional", 1.0, proportional),
        ("stratified", 1.0, stratified),
    ]
    all_specs.extend(sampled_specs)

    csv_path = ensure_feature_source_in_filename(csv_path, feature_name, source_label)
    print(f"CSV output path: {csv_path}")
    completed = load_completed_runs(
        csv_path,
        feature_name=feature_name,
        source_label=source_label,
        model_name=model_name,
    )
    if completed:
        print(
            f"Resuming from existing CSV for model={model_name}; "
            f"found {len(completed)} completed (trial, fold) runs."
        )
    else:
        print(f"No completed (trial, fold) entries found for this feature/source/model={model_name}.")

    for trial_idx, (trial_type, alpha_aug, alpha_mix) in enumerate(all_specs, start=1):
        print(f"=== Trial {trial_idx}/{len(all_specs)}: {trial_type} ===")
        for fold_idx in range(num_folds):
            if (trial_idx, fold_idx) in completed:
                print(f"--- Fold {fold_idx}/{num_folds-1}: already completed, skipping ---")
                continue
            print(f"--- Fold {fold_idx}/{num_folds-1} ---")
            metrics = run_lightgcn_trial(
                code_dir=code_dir,
                labels_pkl=labels_pkl,
                model_name=model_name,
                feature_name=feature_name,
                source_label=source_label,
                alpha_aug=alpha_aug,
                alpha_mix=alpha_mix,
                topks_str=topks,
                recdim=recdim,
                lr=lr,
                decay=decay,
                epochs=epochs,
                layer=layer,
                seed=seed + trial_idx * 100 + fold_idx,
                val_split_idx=fold_idx,
            )
            row = {
                "trial_index": trial_idx,
                "cv_fold": fold_idx,
                "num_folds": num_folds,
                "model": model_name,
                "trial_type": trial_type,
                "feature_name": feature_name,
                "source_label": source_label,
                "alpha_aug": alpha_aug,
                "alpha_mix_json": json.dumps(alpha_mix, sort_keys=True),
                "final_epoch": metrics["final_epoch"],
                "final_loss": metrics["final_loss"],
                "training_time_seconds": metrics["training_time_seconds"],
            }
            row.update(metrics["precision"])
            row.update(metrics["recall"])
            row.update(metrics["ndcg"])
            append_csv_row(csv_path, row)
            completed.add((trial_idx, fold_idx))
            print(f"Appended fold {fold_idx} results to: {csv_path}")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    run_mixing_sweep(
        repo_root=repo_root,
        data_dir=_resolve(repo_root, args.data_dir),
        output_dir=_resolve(repo_root, args.output_dir),
        feature_name=args.feature_name,
        source_label=args.source_label,
        trials=args.trials,
        seed=args.seed,
        topks=args.topks,
        recdim=args.recdim,
        lr=args.lr,
        decay=args.decay,
        model_name=args.model,
        epochs=args.epochs,
        layer=args.layer,
        csv_path=_resolve(repo_root, args.csv_path),
        code_dir=_resolve(repo_root, args.lightgcn_code_dir),
        num_folds=args.num_folds,
        dirichlet_alpha=args.dirichlet_alpha,
        pp_scale=args.proportional_perturbation_scale,
        lognormal_mu=args.lognormal_mu,
        lognormal_sigma=args.lognormal_sigma,
        max_sample_attempts=args.max_sample_attempts,
        smoke_test=args.smoke_test,
        smoke_plot_path=_resolve(repo_root, args.smoke_plot_path),
    )


if __name__ == "__main__":
    main()
