#!/usr/bin/env python3
"""Generate an overall grouped-bar figure from exported plotting CSV files.

Expected input CSV layout:
    mixing/figures/plotting_data/overall/{dataset}/{model}/recdim_{recdim}/{metric}__grouped_bars_test.csv

Example:
    python mixing/figures/generate_overall_grouped_bars.py --recdim 64 --metric ndcg
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_ORDER = ["ml-1m", "lastfm-asia"]
DATASET_DISPLAY = {
    "ml-1m": "MovieLens-1M",
    "lastfm-asia": "LastFM-Asia",
}
DATASET_K = {
    "ml-1m": 20,
    "lastfm-asia": 100,
}

MODEL_ORDER = ["mf", "lgn"]
MODEL_DISPLAY = {
    "mf": "Matrix Factorization",
    "lgn": "LightGCN",
}

BAR_ORDER = ["proportional", "no_augmentation", "stratified", "top_data_mix"]
BAR_DISPLAY = {
    "proportional": "Proportional",
    "no_augmentation": "No Augmentation",
    "stratified": "Stratified",
    "top_data_mix": "Data Mix",
}
BAR_COLORS = {
    "proportional": "#000000",
    "no_augmentation": "#808080",
    "stratified": "#7570b3",
    "top_data_mix": "#d95f02",
}
SIGNIFICANCE_ALPHA = 0.05
BOOTSTRAP_ITERS = 3000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate overall grouped-bar metric figure.")
    parser.add_argument("--recdim", required=True, help="Embedding/recommendation dimension (e.g., 4 or 64).")
    parser.add_argument(
        "--metric",
        required=True,
        help="Metric name, e.g. precision, recall, ndcg (case-insensitive).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debugging statements about loaded files and plotted data.",
    )
    return parser.parse_args()


def _metric_csv_name(metric: str) -> str:
    m = metric.strip().lower()
    aliases = {
        "precision": "precision",
        "recall": "recall",
        "ndcg": "ndcg",
    }
    if m not in aliases:
        raise ValueError("Unsupported metric. Use one of: precision, recall, ndcg.")
    return f"{aliases[m]}__grouped_bars_test.csv"


def _load_metric_df(base: Path, dataset: str, model: str, recdim: str, metric_csv: str) -> pd.DataFrame:
    csv_path = base / dataset / model / f"recdim_{recdim}" / metric_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    needed = {"source_label", "bar_key", "value", "pct_change_vs_proportional"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")

    df = df[df["bar_key"].isin(BAR_ORDER)].copy()
    df["source_label"] = df["source_label"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["pct_change_vs_proportional"] = pd.to_numeric(df["pct_change_vs_proportional"], errors="coerce")
    return df


def _source_order(frames: List[pd.DataFrame]) -> List[str]:
    seen = []
    for frame in frames:
        for src in frame["source_label"].astype(str).tolist():
            if src not in seen:
                seen.append(src)

    def _maybe_numeric(x: str):
        try:
            return (0, float(x))
        except ValueError:
            return (1, x)

    return sorted(seen, key=_maybe_numeric)


def _metric_col_for_dataset(metric: str, dataset: str) -> str:
    k = DATASET_K[dataset]
    return f"{metric}@{k}"


def _load_trial_samples(
    repo_root: Path,
    dataset: str,
    model: str,
    recdim: str,
    metric_col: str,
) -> dict[str, dict[str, np.ndarray]]:
    test_dir = repo_root / "mixing" / "test" / dataset / model / str(recdim)
    files = sorted(test_dir.glob("test_eval__feature-*__source-*.csv"))
    if not files:
        raise FileNotFoundError(f"No test eval files found in {test_dir}")

    out: dict[str, dict[str, list[float]]] = {}
    for fp in files:
        df = pd.read_csv(fp)
        if metric_col not in df.columns:
            continue
        if "source_label" not in df.columns:
            src = fp.stem.split("__source-")[-1]
            df["source_label"] = src
        df["source_label"] = df["source_label"].astype(str)
        vals = pd.to_numeric(df[metric_col], errors="coerce")
        tt = df["trial_type"].astype(str)
        for src in df["source_label"].unique().tolist():
            m = df["source_label"] == src
            out.setdefault(src, {})
            for key, mask in {
                "proportional": tt == "proportional",
                "no_augmentation": tt == "no_augmentation",
                "stratified": tt == "stratified",
                "top_data_mix": tt == "top_1",
                "avg_data_mix": tt.str.startswith("top_"),
            }.items():
                arr = vals[m & mask].dropna().astype(float).to_numpy()
                out[src].setdefault(key, [])
                if arr.size:
                    out[src][key].extend(arr.tolist())

    return {
        src: {k: np.asarray(v, dtype=float) for k, v in kv.items()}
        for src, kv in out.items()
    }


def _significant_vs_proportional(
    prop_vals: np.ndarray,
    comp_vals: np.ndarray,
    *,
    alpha: float = SIGNIFICANCE_ALPHA,
    iters: int = BOOTSTRAP_ITERS,
    seed: int = 42,
) -> bool:
    prop = np.asarray(prop_vals, dtype=float)
    comp = np.asarray(comp_vals, dtype=float)
    prop = prop[np.isfinite(prop)]
    comp = comp[np.isfinite(comp)]
    if len(prop) < 2 or len(comp) < 2:
        return False

    rng = np.random.default_rng(seed)
    diffs = np.empty(iters, dtype=float)
    for i in range(iters):
        p = rng.choice(prop, size=len(prop), replace=True).mean()
        c = rng.choice(comp, size=len(comp), replace=True).mean()
        diffs[i] = c - p
    lo = float(np.percentile(diffs, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(diffs, 100.0 * (1.0 - alpha / 2.0)))
    return lo > 0.0 or hi < 0.0


def main() -> None:
    args = _parse_args()
    recdim = str(args.recdim)
    metric = args.metric.strip().lower()
    debug = bool(args.debug)

    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "mixing" / "figures" / "plotting_data" / "overall"
    out_dir = repo_root / "mixing" / "figures" / "pdfs" / "overall"
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_csv = _metric_csv_name(metric)

    dfs: Dict[tuple[str, str], pd.DataFrame] = {}
    trial_cache: Dict[tuple[str, str], dict[str, dict[str, np.ndarray]]] = {}
    dataset_source_counts: Dict[str, int] = {}
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            df = _load_metric_df(data_root, dataset, model, recdim, metric_csv)
            dfs[(model, dataset)] = df
            metric_col = _metric_col_for_dataset(metric, dataset)
            trial_cache[(model, dataset)] = _load_trial_samples(
                repo_root, dataset, model, recdim, metric_col
            )
            n_src = len(_source_order([df]))
            dataset_source_counts[dataset] = max(dataset_source_counts.get(dataset, 0), n_src)
            if debug:
                print(
                    f"[DEBUG] Loaded {len(df):,} rows for dataset={dataset}, model={model}, "
                    f"recdim={recdim}, metric={metric}"
                )
                print(
                    f"[DEBUG]   source groups ({df['source_label'].nunique()}): "
                    f"{sorted(df['source_label'].unique().tolist())}"
                )

    width_ratios = [max(1, dataset_source_counts.get(ds, 1)) for ds in DATASET_ORDER]
    fig_width = 12.5
    fig_height = 6.0
    if debug:
        print(f"[DEBUG] Column width ratios by dataset: {dict(zip(DATASET_ORDER, width_ratios))}")

    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, num=len(BAR_ORDER))

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

    fig, axes = plt.subplots(
        nrows=len(MODEL_ORDER),
        ncols=len(DATASET_ORDER),
        figsize=(fig_width, fig_height),
        sharey=False,
        constrained_layout=False,
        gridspec_kw={"width_ratios": width_ratios},
    )

    if len(MODEL_ORDER) == 1 and len(DATASET_ORDER) == 1:
        axes = np.array([[axes]])
    elif len(MODEL_ORDER) == 1:
        axes = np.array([axes])
    elif len(DATASET_ORDER) == 1:
        axes = np.array([[ax] for ax in axes])

    for r, model in enumerate(MODEL_ORDER):
        for c, dataset in enumerate(DATASET_ORDER):
            ax = axes[r, c]
            df = dfs[(model, dataset)].copy()
            sources = _source_order([df])
            x = np.arange(len(sources), dtype=float)

            pivot = (
                df.pivot_table(index="source_label", columns="bar_key", values="value", aggfunc="mean")
                .reindex(index=sources, columns=BAR_ORDER)
            )

            mat = pivot.to_numpy(dtype=float)
            finite_vals = mat[np.isfinite(mat)]
            if finite_vals.size > 0:
                y_min = float(np.min(finite_vals))
                y_max = float(np.max(finite_vals))
                span = max(y_max - y_min, 1e-6)
                y_low = y_min - 0.08 * span
                y_high = y_max + 0.18 * span
            else:
                y_low, y_high, span = 0.0, 1.0, 1.0
            if debug:
                print(
                    f"[DEBUG] Plot panel dataset={dataset}, model={model}: "
                    f"sources={sources}, finite_bars={finite_vals.size}"
                )

            for j, bar_key in enumerate(BAR_ORDER):
                heights = mat[:, j]
                bars = ax.bar(
                    x + offsets[j],
                    heights,
                    width=width,
                    color=BAR_COLORS[bar_key],
                    edgecolor="white",
                    linewidth=0.7,
                    label=BAR_DISPLAY[bar_key],
                    zorder=3,
                )

                if bar_key == "top_data_mix":
                    pct_lookup = (
                        df[df["bar_key"] == "top_data_mix"]
                        .drop_duplicates(subset=["source_label"])
                        .set_index("source_label")["pct_change_vs_proportional"]
                    )
                    for i, rect in enumerate(bars):
                        src = sources[i]
                        y = rect.get_height()
                        pct = pct_lookup.get(src, np.nan)
                        sig = False
                        src_trials = trial_cache[(model, dataset)].get(src, {})
                        sig = _significant_vs_proportional(
                            src_trials.get("proportional", np.asarray([])),
                            src_trials.get("top_data_mix", np.asarray([])),
                        )
                        if np.isfinite(y) and np.isfinite(pct):
                            ax.text(
                                rect.get_x() + rect.get_width() / 2.0,
                                y + max(0.003, 0.02 * span),
                                f"{pct:+.1f}%{'*' if sig else ''}",
                                ha="center",
                                va="bottom",
                                fontsize=12,
                                rotation=0,
                            )

            ax.set_title(f"{DATASET_DISPLAY[dataset]} {MODEL_DISPLAY[model]}")
            ax.set_xticks(x)
            ax.set_xticklabels(sources)
            ax.set_xlabel("Target Group")
            ylabel_base = "NDCG" if metric == "ndcg" else metric.title()
            ylabel = f"{ylabel_base}@{DATASET_K[dataset]}"
            ax.set_ylabel(ylabel)
            ax.set_ylim(y_low, y_high)
            ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7, zorder=0)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=BAR_COLORS[k]) for k in BAR_ORDER]
    labels = [BAR_DISPLAY[k] for k in BAR_ORDER]
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])

    out_file = out_dir / f"overall_{metric}_recdim_{recdim}.pdf"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {out_file}")


if __name__ == "__main__":
    main()
