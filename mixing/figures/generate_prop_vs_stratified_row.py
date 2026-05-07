#!/usr/bin/env python3
"""Generate prop-vs-stratified relative-change figure as a single row of subplots.

Reads experiment result CSVs from:
    mixing/experiments/prop_vs_stratified/results

Reads dataset/source-group pairs from JSON config:
    mixing/figures/prop_vs_stratified_pairs.json

Outputs PDF to:
    mixing/figures/pdfs/prop_vs_stratified
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_DISPLAY = {
    "ml-1m": "MovieLens-1M",
    "lastfm-asia": "LastFM-Asia",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot stratified relative change vs proportional across recdim for configured pairs."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["lgn", "mf"],
        help="Model to plot.",
    )
    parser.add_argument(
        "--metric",
        required=True,
        help=(
            "Metric base or full column name. Examples: recall, ndcg, precision, recall@20, ndcg@100."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("mixing/figures/prop_vs_stratified_pairs.json"),
        help="JSON file with `pairs` list of {dataset, source_group}.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("mixing/experiments/prop_vs_stratified/results"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("mixing/figures/pdfs/prop_vs_stratified"),
    )
    parser.add_argument(
        "--ci-bands",
        action="store_true",
        help="Add bootstrap confidence-interval bands (light gray).",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=5000,
        help="Number of bootstrap iterations for CI estimation.",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=95.0,
        help="Confidence level for CI bands, e.g. 95.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for bootstrap CI.",
    )
    return parser.parse_args()


def _resolve(repo_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _load_pairs(config_path: Path) -> list[dict[str, str]]:
    payload = json.loads(config_path.read_text())
    pairs = payload.get("pairs", [])
    if not pairs:
        raise ValueError(f"No pairs found in config: {config_path}")

    normalized = []
    for p in pairs:
        if "dataset" not in p or "source_group" not in p:
            raise ValueError("Each pair must include `dataset` and `source_group`.")
        normalized.append({"dataset": str(p["dataset"]), "source_group": str(p["source_group"])})
    return normalized


def _metric_col_for_dataset(df: pd.DataFrame, metric_input: str, dataset: str) -> str:
    metric_input = metric_input.strip().lower()
    if "@" in metric_input:
        if metric_input not in df.columns:
            raise KeyError(f"Metric column `{metric_input}` not found for dataset={dataset}.")
        return metric_input

    # dataset-specific k convention requested by user
    target_k = 20 if dataset == "ml-1m" else 100 if dataset == "lastfm-asia" else None
    if target_k is not None:
        candidate = f"{metric_input}@{target_k}"
        if candidate in df.columns:
            return candidate

    # fallback to any available metric family column
    cols = sorted(
        [c for c in df.columns if c.startswith(f"{metric_input}@")],
        key=lambda x: int(x.split("@")[1]),
    )
    if cols:
        return cols[0]

    available = [c for c in df.columns if c.startswith(("precision@", "recall@", "ndcg@"))]
    raise KeyError(
        f"No metric column found for metric input `{metric_input}` and dataset `{dataset}`. "
        f"Available metric columns: {available}"
    )


def _metric_label(metric_col: str) -> str:
    base, kval = metric_col.split("@")
    base_label = "NDCG" if base.lower() == "ndcg" else base.title()
    return f"{base_label}@{kval}"


def _bootstrap_pct_ci(
    prop_vals: np.ndarray,
    strat_vals: np.ndarray,
    iters: int,
    ci_level: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    prop_vals = np.asarray(prop_vals, dtype=float)
    strat_vals = np.asarray(strat_vals, dtype=float)
    prop_vals = prop_vals[np.isfinite(prop_vals)]
    strat_vals = strat_vals[np.isfinite(strat_vals)]
    if len(prop_vals) == 0 or len(strat_vals) == 0:
        return (np.nan, np.nan)

    boot = np.empty(iters, dtype=float)
    for i in range(iters):
        p = rng.choice(prop_vals, size=len(prop_vals), replace=True).mean()
        s = rng.choice(strat_vals, size=len(strat_vals), replace=True).mean()
        boot[i] = np.nan if p == 0 else (s - p) / p * 100.0

    alpha = (100.0 - ci_level) / 2.0
    lo = float(np.nanpercentile(boot, alpha))
    hi = float(np.nanpercentile(boot, 100.0 - alpha))
    return (lo, hi)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    repo_root = Path(__file__).resolve().parents[2]
    results_dir = _resolve(repo_root, args.results_dir)
    out_dir = _resolve(repo_root, args.out_dir)
    config_path = _resolve(repo_root, args.config)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _load_pairs(config_path)
    files = sorted(results_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No result CSV files found in {results_dir}")

    frames = [pd.read_csv(fp) for fp in files]
    df = pd.concat(frames, ignore_index=True)

    # Keep only the two baselines needed for relative change.
    df = df[df["trial_type"].astype(str).isin(["proportional", "stratified"])].copy()
    if df.empty:
        raise ValueError("No proportional/stratified rows found in results.")
    df = df[df["model"].astype(str).str.lower() == args.model.lower()].copy()
    if df.empty:
        raise ValueError(f"No rows found for model={args.model}.")

    # Same font system as overall figure.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 14,
        }
    )

    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.5), squeeze=False)
    axes = axes[0]

    for i, pair in enumerate(pairs):
        dataset = pair["dataset"]
        source_group = pair["source_group"]

        sdf = df[
            (df["dataset"].astype(str) == dataset)
            & (df["source_label"].astype(str) == source_group)
        ].copy()
        if sdf.empty:
            raise ValueError(f"No rows for dataset={dataset}, source_group={source_group}.")

        metric_col = _metric_col_for_dataset(sdf, args.metric, dataset)
        sdf[metric_col] = pd.to_numeric(sdf[metric_col], errors="coerce")
        sdf["recdim"] = pd.to_numeric(sdf["recdim"], errors="coerce")

        agg = (
            sdf.groupby(["trial_type", "recdim"], as_index=False)[metric_col]
            .mean(numeric_only=True)
            .dropna(subset=["recdim", metric_col])
        )

        prop = agg[agg["trial_type"] == "proportional"][["recdim", metric_col]].rename(
            columns={metric_col: "prop"}
        )
        strat = agg[agg["trial_type"] == "stratified"][["recdim", metric_col]].rename(
            columns={metric_col: "strat"}
        )

        cmp_df = strat.merge(prop, on="recdim", how="inner").sort_values("recdim")
        if cmp_df.empty:
            raise ValueError(
                f"No overlapping recdim values between proportional and stratified for "
                f"dataset={dataset}, source_group={source_group}."
            )

        cmp_df["pct_change"] = (cmp_df["strat"] - cmp_df["prop"]) / cmp_df["prop"] * 100.0
        if args.ci_bands:
            lo_list = []
            hi_list = []
            for recdim_val in cmp_df["recdim"].tolist():
                prop_trials = sdf[
                    (sdf["trial_type"] == "proportional") & (sdf["recdim"] == recdim_val)
                ][metric_col].to_numpy()
                strat_trials = sdf[
                    (sdf["trial_type"] == "stratified") & (sdf["recdim"] == recdim_val)
                ][metric_col].to_numpy()
                lo, hi = _bootstrap_pct_ci(
                    prop_trials,
                    strat_trials,
                    iters=args.bootstrap_iters,
                    ci_level=args.ci_level,
                    rng=rng,
                )
                lo_list.append(lo)
                hi_list.append(hi)
            cmp_df["pct_ci_lo"] = lo_list
            cmp_df["pct_ci_hi"] = hi_list

        ax = axes[i]
        if args.ci_bands and "pct_ci_lo" in cmp_df.columns and "pct_ci_hi" in cmp_df.columns:
            ax.fill_between(
                cmp_df["recdim"].to_numpy(dtype=float),
                cmp_df["pct_ci_lo"].to_numpy(dtype=float),
                cmp_df["pct_ci_hi"].to_numpy(dtype=float),
                color="#d3d3d3",
                alpha=0.6,
                linewidth=0,
                zorder=1,
            )
        ax.plot(
            cmp_df["recdim"],
            cmp_df["pct_change"],
            color="black",
            marker="o",
            linewidth=1.8,
            markersize=3.0,
            zorder=2,
        )
        try:
            ax.set_xscale("log", base=2)
        except TypeError:
            ax.set_xscale("log", basex=2)

        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)

        dataset_title = DATASET_DISPLAY.get(dataset, dataset)
        ax.set_title(f"{dataset_title}\nTarget Group: {source_group}")

        ax.set_xlabel("Embedding Dimension")
        metric_label = _metric_label(metric_col)
        ax.set_ylabel(f"{metric_label} $\\Delta$%")

        # show only actual tested recdims as ticks
        xticks = sorted(cmp_df["recdim"].astype(float).unique().tolist())
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(x)) if float(x).is_integer() else f"{x:g}" for x in xticks])

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.tight_layout()

    metric_slug = args.metric.strip().lower().replace("@", "at")
    out_file = out_dir / f"prop_vs_stratified__{args.model}__{metric_slug}.pdf"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {out_file}")


if __name__ == "__main__":
    main()
