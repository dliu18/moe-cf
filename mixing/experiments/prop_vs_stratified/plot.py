#!/usr/bin/env python3
"""Plot proportional vs stratified performance vs embedding dimension."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot prop-vs-stratified experiment results.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["lgn", "mf"])
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("mixing/experiments/prop_vs_stratified/results"),
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("mixing/experiments/prop_vs_stratified/plots"),
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="",
        help=(
            "Comma-separated metric columns to plot (e.g. recall@20,ndcg@20). "
            "If omitted, auto-detects one available k for each of precision/recall/ndcg."
        ),
    )
    return parser.parse_args()


def _resolve(repo_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    results_dir = _resolve(repo_root, args.results_dir)
    plots_dir = _resolve(repo_root, args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(results_dir.glob(f"{args.dataset}__{args.model}__*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No result files found for dataset={args.dataset}, model={args.model} in {results_dir}"
        )

    frames = [pd.read_csv(fp) for fp in files]
    df = pd.concat(frames, ignore_index=True)

    available_metric_cols = [c for c in df.columns if c.startswith(("precision@", "recall@", "ndcg@"))]
    if args.metrics.strip():
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
        missing = [m for m in metrics if m not in df.columns]
        if missing:
            raise KeyError(
                f"Missing metric columns: {missing}. Available metrics: {available_metric_cols}"
            )
    else:
        metrics = []
        for base in ["recall", "ndcg", "precision"]:
            cols = sorted(
                [c for c in available_metric_cols if c.startswith(f"{base}@")],
                key=lambda x: int(x.split("@")[1]),
            )
            if cols:
                metrics.append(cols[0])
        if not metrics:
            raise KeyError(
                f"No ranking metric columns found. Available columns: {list(df.columns)}"
            )
        print(f"Auto-selected metrics based on available headers: {metrics}")

    df = df[df["trial_type"].isin(["proportional", "stratified"])].copy()
    if df.empty:
        raise ValueError("No proportional/stratified rows found after filtering.")

    df = df[df["model"].astype(str).str.lower() == str(args.model).lower()].copy()
    if df.empty:
        raise ValueError(f"No rows found for model={args.model}.")

    agg = (
        df.groupby(["source_label", "trial_type", "recdim"], as_index=False)[metrics]
        .mean(numeric_only=True)
        .sort_values(["source_label", "trial_type", "recdim"])
    )

    sources = sorted(agg["source_label"].astype(str).unique().tolist(), key=lambda x: (len(x), x))
    n_rows = len(sources)
    n_cols = len(metrics)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    colors = {"proportional": "tab:blue", "stratified": "tab:orange"}

    for r, source in enumerate(sources):
        source_df = agg[agg["source_label"].astype(str) == str(source)]
        for c, metric in enumerate(metrics):
            ax = axes[r][c]
            for baseline in ["proportional", "stratified"]:
                sdf = source_df[source_df["trial_type"] == baseline].sort_values("recdim")
                if sdf.empty:
                    continue
                ax.plot(
                    sdf["recdim"],
                    sdf[metric],
                    marker="o",
                    linewidth=2,
                    label=baseline,
                    color=colors[baseline],
                )
            try:
                ax.set_xscale("log", base=2)
            except TypeError:
                # Backward compatibility for older matplotlib versions.
                ax.set_xscale("log", basex=2)
            ax.set_xlabel("Embedding dimension (recdim)")
            ax.set_ylabel(metric)
            ax.set_title(f"source={source} | metric={metric}")
            ax.grid(alpha=0.3)
            if r == 0 and c == 0:
                ax.legend()

    out_png = plots_dir / f"{args.dataset}__{args.model}__prop_vs_stratified.png"
    out_pdf = plots_dir / f"{args.dataset}__{args.model}__prop_vs_stratified.pdf"
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)

    # Additional page: stratified percent change relative to proportional.
    prop = agg[agg["trial_type"] == "proportional"].copy()
    strat = agg[agg["trial_type"] == "stratified"].copy()
    merge_cols = ["source_label", "recdim"]
    prop = prop.rename(columns={m: f"{m}_prop" for m in metrics})
    strat = strat.rename(columns={m: f"{m}_strat" for m in metrics})
    cmp_df = strat.merge(
        prop[merge_cols + [f"{m}_prop" for m in metrics]],
        on=merge_cols,
        how="inner",
    )

    for metric in metrics:
        pcol = f"{metric}_prop"
        scol = f"{metric}_strat"
        out_col = f"{metric}_pct_change"
        cmp_df[out_col] = (cmp_df[scol] - cmp_df[pcol]) / cmp_df[pcol] * 100.0
        cmp_df[f"{metric}_raw_change"] = cmp_df[scol] - cmp_df[pcol]

    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    for r, source in enumerate(sources):
        source_cmp = cmp_df[cmp_df["source_label"].astype(str) == str(source)].sort_values("recdim")
        for c, metric in enumerate(metrics):
            ax = axes2[r][c]
            ycol = f"{metric}_pct_change"
            if not source_cmp.empty and ycol in source_cmp.columns:
                recdims = source_cmp["recdim"].astype(int).tolist()
                yvals = source_cmp[ycol].tolist()
                x = np.arange(len(recdims))
                ax.bar(x, yvals, color="tab:orange", width=0.8, label="stratified vs proportional")
                ax.set_xticks(x)
                ax.set_xticklabels([str(d) for d in recdims], rotation=45, ha="right")
            ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
            ax.set_xlabel("Embedding dimension (recdim)")
            ax.set_ylabel("% change vs proportional")
            ax.set_title(f"source={source} | metric={metric}")
            ax.grid(alpha=0.3)
            if r == 0 and c == 0:
                ax.legend()
    fig2.tight_layout()

    # Third page: stratified raw change relative to proportional.
    fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    for r, source in enumerate(sources):
        source_cmp = cmp_df[cmp_df["source_label"].astype(str) == str(source)].sort_values("recdim")
        for c, metric in enumerate(metrics):
            ax = axes3[r][c]
            ycol = f"{metric}_raw_change"
            if not source_cmp.empty and ycol in source_cmp.columns:
                recdims = source_cmp["recdim"].astype(int).tolist()
                yvals = source_cmp[ycol].tolist()
                x = np.arange(len(recdims))
                ax.bar(x, yvals, color="tab:red", width=0.8, label="stratified - proportional")
                ax.set_xticks(x)
                ax.set_xticklabels([str(d) for d in recdims], rotation=45, ha="right")
            ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
            ax.set_xlabel("Embedding dimension (recdim)")
            ax.set_ylabel("Raw change vs proportional")
            ax.set_title(f"source={source} | metric={metric}")
            ax.grid(alpha=0.3)
            if r == 0 and c == 0:
                ax.legend()
    fig3.tight_layout()

    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig)
        pdf.savefig(fig2)
        pdf.savefig(fig3)

    plt.close(fig)
    plt.close(fig2)
    plt.close(fig3)
    print(f"Saved plot: {out_png}")
    print(f"Saved multi-page PDF: {out_pdf}")


if __name__ == "__main__":
    main()
