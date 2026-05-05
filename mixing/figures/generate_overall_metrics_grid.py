#!/usr/bin/env python3
"""Generate a variant of the Overall grouped-bar figure.

Layout:
- Rows: datasets
- Columns: evaluation metrics (Precision, Recall, NDCG)

Differences vs overall:
- CLI takes model and dataset input.
- Annotates percentage change vs proportional for all non-proportional bars.
- No subplot titles.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_DISPLAY = {
    "ml-1m": "MovieLens-1M",
    "lastfm-asia": "LastFM-Asia",
}

DATASET_K = {
    "ml-1m": 20,
    "lastfm-asia": 100,
}

METRIC_ORDER = ["precision", "recall", "ndcg"]
METRIC_DISPLAY = {
    "precision": "Precision",
    "recall": "Recall",
    "ndcg": "NDCG",
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate grouped bars for all metrics with dataset rows.")
    parser.add_argument("--recdim", required=True, help="Embedding/recommendation dimension (e.g., 4 or 64).")
    parser.add_argument("--model", required=True, choices=["lgn", "mf"], help="Model name.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name or comma-separated dataset names (e.g., ml-1m,lastfm-asia).",
    )
    parser.add_argument("--debug", action="store_true", help="Print debug statements.")
    return parser.parse_args()


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


def _load_metric_df(base: Path, dataset: str, model: str, recdim: str, metric_key: str) -> pd.DataFrame:
    metric_csv = f"{metric_key}__grouped_bars_test.csv"
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


def _y_label(metric_key: str, dataset: str) -> str:
    base = METRIC_DISPLAY[metric_key]
    k = DATASET_K.get(dataset)
    return f"{base}@{k}" if k is not None else base


def main() -> None:
    args = _parse_args()
    recdim = str(args.recdim)
    model = str(args.model).lower()
    datasets = [d.strip() for d in str(args.dataset).split(",") if d.strip()]
    debug = bool(args.debug)

    for d in datasets:
        if d not in DATASET_DISPLAY:
            raise ValueError(f"Unsupported dataset `{d}`. Supported: {sorted(DATASET_DISPLAY.keys())}")

    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "mixing" / "figures" / "plotting_data" / "overall"
    out_dir = repo_root / "mixing" / "figures" / "pdfs" / "overall"
    out_dir.mkdir(parents=True, exist_ok=True)

    # load all required dataframes
    dfs: Dict[tuple[str, str], pd.DataFrame] = {}
    dataset_source_counts: Dict[str, int] = {}
    for dataset in datasets:
        metric_frames = []
        for metric in METRIC_ORDER:
            df = _load_metric_df(data_root, dataset, model, recdim, metric)
            dfs[(dataset, metric)] = df
            metric_frames.append(df)
            if debug:
                print(
                    f"[DEBUG] Loaded rows={len(df):,} dataset={dataset} model={model} recdim={recdim} metric={metric}"
                )
        dataset_source_counts[dataset] = max(1, len(_source_order(metric_frames)))

    width_ratios = [max(1, dataset_source_counts[d]) for d in datasets]
    fig_width = 14.0
    fig_height = 3.6 * len(datasets)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 12,
        }
    )

    fig, axes = plt.subplots(
        nrows=len(datasets),
        ncols=len(METRIC_ORDER),
        figsize=(fig_width, fig_height),
        sharey=False,
        constrained_layout=False,
        gridspec_kw={"width_ratios": [1, 1, 1]},
    )

    if len(datasets) == 1 and len(METRIC_ORDER) == 1:
        axes = np.array([[axes]])
    elif len(datasets) == 1:
        axes = np.array([axes])
    elif len(METRIC_ORDER) == 1:
        axes = np.array([[ax] for ax in axes])

    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, num=len(BAR_ORDER))

    for r, dataset in enumerate(datasets):
        # dataset-specific source ordering across metrics
        src_frames = [dfs[(dataset, m)] for m in METRIC_ORDER]
        sources = _source_order(src_frames)
        x = np.arange(len(sources), dtype=float)

        for c, metric in enumerate(METRIC_ORDER):
            ax = axes[r, c]
            df = dfs[(dataset, metric)]

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
                y_low = y_min - 0.10 * span
                y_high = y_max + 0.24 * span
            else:
                y_low, y_high, span = 0.0, 1.0, 1.0

            for j, bar_key in enumerate(BAR_ORDER):
                heights = mat[:, j]
                bars = ax.bar(
                    x + offsets[j],
                    heights,
                    width=width,
                    color=BAR_COLORS[bar_key],
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=3,
                )

                # Annotate % change for all non-proportional bars.
                if bar_key != "proportional":
                    pct_lookup = (
                        df[df["bar_key"] == bar_key]
                        .drop_duplicates(subset=["source_label"])
                        .set_index("source_label")["pct_change_vs_proportional"]
                    )
                    for i_src, rect in enumerate(bars):
                        src = sources[i_src]
                        y = rect.get_height()
                        pct = pct_lookup.get(src, np.nan)
                        if np.isfinite(y) and np.isfinite(pct):
                            ax.text(
                                rect.get_x() + rect.get_width() / 2.0,
                                y + max(0.0015, 0.010 * span),
                                f"{pct:+.1f}%",
                                ha="center",
                                va="bottom",
                                fontsize=12,
                                rotation=90,
                            )

            # No subplot titles (requested).
            ax.set_xticks(x)
            ax.set_xticklabels(sources)
            ax.set_xlabel("Source Group")
            ax.set_ylabel(_y_label(metric, dataset))
            ax.set_ylim(y_low, y_high)
            ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7, zorder=0)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

            if debug:
                print(
                    f"[DEBUG] panel dataset={dataset}, metric={metric}, sources={sources}, finite_bars={finite_vals.size}"
                )

    handles = [plt.Rectangle((0, 0), 1, 1, color=BAR_COLORS[k]) for k in BAR_ORDER]
    labels = [BAR_DISPLAY[k] for k in BAR_ORDER]
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])

    dataset_slug = "-".join(datasets)
    out_file = out_dir / f"overall_metrics_grid__{dataset_slug}__{model}__recdim_{recdim}.pdf"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {out_file}")


if __name__ == "__main__":
    main()
