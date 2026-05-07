#!/usr/bin/env python3
"""Plot source-group similarity vs top-mix composition.

Creates two figures (row of subplots over configured dataset/source pairs):
1) Grouped bars by augmentation group:
   - bar A: avg user-level Jaccard similarity(source_group, augmentation_group)
   - bar B: alpha_mix component under top_1 data mix
2) Scatter per augmentation group:
   - x: similarity
   - y: alpha_mix ratio

Uses pairs config shared with prop-vs-stratified figure.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    SCIPY_AVAILABLE = False


DATASET_LABEL_KEY = {
    "ml-1m": "Age",
    "lastfm-asia": "Country",
}
DATASET_DISPLAY = {
    "ml-1m": "MovieLens-1M",
    "lastfm-asia": "LastFM-Asia",
}
DISTANCE_RANK_D = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate similarity vs top-mix figures.")
    parser.add_argument("--recdim", required=True, help="Embedding dimension.")
    parser.add_argument("--model", required=True, choices=["lgn", "mf"], help="Model name.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("mixing/figures/prop_vs_stratified_pairs.json"),
        help="JSON with `pairs: [{dataset, source_group}, ...]`.",
    )
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=Path("LightGCN/data"),
        help="Root directory containing dataset split outputs.",
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=Path("mixing/test"),
        help="Root directory containing test evaluation CSVs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("mixing/figures/pdfs/similarity_vs_topmix"),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Use average alpha_mix component across top_1..top_k mixes (default: 1).",
    )
    parser.add_argument(
        "--similarity-cache",
        type=Path,
        default=Path("mixing/figures/similarity_cache.json"),
        help="JSON cache for computed group-pair similarities.",
    )
    parser.add_argument(
        "--two-column",
        action="store_true",
        help=(
            "Generate two-column-friendly outputs: "
            "(1) scatter-only panels with dataset rows, "
            "(2) separate target-level augmentation-ratio bar chart."
        ),
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _resolve(repo_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _load_pairs(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text())
    pairs = payload.get("pairs", [])
    if not pairs:
        raise ValueError(f"No pairs found in config: {path}")
    out = []
    for p in pairs:
        out.append({"dataset": str(p["dataset"]), "source_group": str(p["source_group"])})
    return out


def _read_lightgcn_user_items(path: Path) -> dict[int, set[int]]:
    user_items: dict[int, set[int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            uid = int(parts[0])
            items = set(int(x) for x in parts[1:])
            user_items[uid] = items
    return user_items


def _load_user_group_map(labels_pkl: Path, label_key: str) -> dict[str, list[int]]:
    payload = pickle.load(labels_pkl.open("rb"))
    if label_key not in payload:
        raise KeyError(f"Label key `{label_key}` not found in {labels_pkl}. Keys={list(payload.keys())}")
    grp = payload[label_key]
    # expected shape: {group_label: [user_ids]}
    if not isinstance(grp, dict):
        raise TypeError(f"Expected dict for payload[{label_key}], got {type(grp)}")
    return {str(k): [int(u) for u in v] for k, v in grp.items()}


def _build_group_matrix(users: list[int], all_items: dict[int, set[int]], n_items: int):
    if SCIPY_AVAILABLE:
        data = []
        rows = []
        cols = []
        for r, uid in enumerate(users):
            for it in all_items[uid]:
                rows.append(r)
                cols.append(it)
                data.append(1.0)
        if not rows:
            return csr_matrix((len(users), n_items), dtype=np.float64)
        return csr_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols))), shape=(len(users), n_items))

    mat = np.zeros((len(users), n_items), dtype=np.float64)
    for r, uid in enumerate(users):
        for it in all_items[uid]:
            mat[r, it] = 1.0
    return mat


def _top_svd_v(group_matrix, max_rank: int) -> np.ndarray:
    if SCIPY_AVAILABLE:
        m, n = group_matrix.shape
        k = min(max_rank, min(m, n) - 1)
        if k < 1:
            return np.zeros((n, 0), dtype=np.float64)
        _, s, vt = svds(group_matrix, k=k)
        order = np.argsort(-s)
        vt = vt[order, :]
        return vt.T.astype(np.float64, copy=False)

    arr = np.asarray(group_matrix, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((arr.shape[1], 0), dtype=np.float64)
    _, _, vt = np.linalg.svd(arr, full_matrices=False)
    k = min(max_rank, vt.shape[0])
    return vt[:k, :].T


def _normalized_discrepancy_vvt(v_i: np.ndarray, v_j: np.ndarray, d: int) -> float:
    di = min(d, v_i.shape[1])
    dj = min(d, v_j.shape[1])
    if di == 0:
        return np.nan
    vi = v_i[:, :di]
    vj = v_j[:, :dj]
    m = vi.T @ vj
    # Pi=ViVi^T, Pj=VjVj^T
    # ||Pi||_F^2 = di, ||Pj||_F^2 = dj, tr(PiPj)=||Vi^TVj||_F^2
    cross = float(np.sum(m * m))
    return float((di + dj - 2.0 * cross) / di)


def _top1_alpha_mix_for_source(test_file: Path) -> dict[str, float]:
    df = pd.read_csv(test_file)
    top1 = df[df["trial_type"].astype(str) == "top_1"].copy()
    if top1.empty:
        raise ValueError(f"No top_1 rows found in {test_file}")

    mixes: list[dict[str, float]] = []
    for raw in top1["alpha_mix_json"].astype(str).tolist():
        mix = {str(k): float(v) for k, v in json.loads(raw).items()}
        mixes.append(mix)
    if not mixes:
        raise ValueError(f"Could not parse top_1 alpha_mix_json in {test_file}")

    ref = mixes[0]
    ref_keys = set(ref.keys())
    atol = 1e-10
    for i, mix in enumerate(mixes[1:], start=1):
        if set(mix.keys()) != ref_keys:
            raise AssertionError(
                f"top_1 alpha_mix keys mismatch across trials in {test_file}. "
                f"trial0_keys={sorted(ref_keys)}, trial{i}_keys={sorted(set(mix.keys()))}"
            )
        for k in ref_keys:
            if not np.isclose(mix[k], ref[k], atol=atol, rtol=0.0):
                raise AssertionError(
                    f"top_1 alpha_mix value mismatch across trials in {test_file} for key={k}: "
                    f"trial0={ref[k]:.16f}, trial{i}={mix[k]:.16f}"
                )

    return ref


def _avg_alpha_mix_across_topk_for_source(test_file: Path, top_k: int) -> dict[str, float]:
    if top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if top_k == 1:
        # Preserve prior behavior including consistency assertion.
        return _top1_alpha_mix_for_source(test_file)

    df = pd.read_csv(test_file)
    tt = df["trial_type"].astype(str)
    top_rows = df[tt.str.startswith("top_")].copy()
    if top_rows.empty:
        raise ValueError(f"No top_* rows found in {test_file}")

    top_rows["top_rank"] = pd.to_numeric(top_rows["trial_type"].str.replace("top_", "", regex=False), errors="coerce")
    top_rows = top_rows[(top_rows["top_rank"] >= 1) & (top_rows["top_rank"] <= top_k)].copy()
    if top_rows.empty:
        raise ValueError(f"No rows found in top_1..top_{top_k} for {test_file}")

    acc: dict[str, list[float]] = {}
    for raw in top_rows["alpha_mix_json"].astype(str).tolist():
        mix = {str(k): float(v) for k, v in json.loads(raw).items()}
        for k, v in mix.items():
            acc.setdefault(k, []).append(v)

    if not acc:
        raise ValueError(f"Could not parse alpha_mix_json in top_1..top_{top_k} for {test_file}")
    return {k: float(np.mean(vs)) for k, vs in acc.items() if len(vs) > 0}


def _avg_alpha_aug_across_topk_for_source(test_file: Path, top_k: int) -> float:
    if top_k < 1:
        raise ValueError("--top-k must be >= 1")
    df = pd.read_csv(test_file)
    tt = df["trial_type"].astype(str)
    top_rows = df[tt.str.startswith("top_")].copy()
    if top_rows.empty:
        raise ValueError(f"No top_* rows found in {test_file}")
    top_rows["top_rank"] = pd.to_numeric(
        top_rows["trial_type"].str.replace("top_", "", regex=False),
        errors="coerce",
    )
    top_rows = top_rows[(top_rows["top_rank"] >= 1) & (top_rows["top_rank"] <= top_k)].copy()
    if top_rows.empty:
        raise ValueError(f"No rows found in top_1..top_{top_k} for {test_file}")
    vals = pd.to_numeric(top_rows["alpha_aug"], errors="coerce").dropna()
    if len(vals) == 0:
        raise ValueError(f"No valid alpha_aug values in top_1..top_{top_k} for {test_file}")
    return float(vals.mean())


def _find_test_eval_file(test_dir: Path, source_group: str) -> Path:
    matches = sorted(test_dir.glob(f"test_eval__feature-*__source-{source_group}.csv"))
    if not matches:
        raise FileNotFoundError(f"No test eval file for source={source_group} in {test_dir}")
    return matches[0]


def _sort_group_labels(labels: list[str]) -> list[str]:
    def key_fn(x: str):
        try:
            return (0, int(x))
        except ValueError:
            return (1, x)

    return sorted(labels, key=key_fn)


def _pair_key(dataset: str, g1: str, g2: str) -> str:
    a, b = sorted([str(g1), str(g2)], key=lambda x: (len(x), x))
    return f"{dataset}::{a}::{b}"


def main() -> None:
    args = parse_args()
    recdim = str(args.recdim)
    model = str(args.model).lower()
    top_k = int(args.top_k)
    if top_k < 1:
        raise ValueError("--top-k must be >= 1")

    repo_root = Path(__file__).resolve().parents[2]
    config_path = _resolve(repo_root, args.config)
    splits_root = _resolve(repo_root, args.splits_root)
    test_root = _resolve(repo_root, args.test_root)
    out_dir = _resolve(repo_root, args.out_dir)
    cache_path = _resolve(repo_root, args.similarity_cache)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        similarity_cache = json.loads(cache_path.read_text())
        if not isinstance(similarity_cache, dict):
            raise TypeError(f"Similarity cache must be a JSON object: {cache_path}")
    else:
        similarity_cache = {}

    pairs = _load_pairs(config_path)

    # same font family as overall plot
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 14,
            "axes.titlesize": 17,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 14,
        }
    )

    # Cache per-dataset structures
    dataset_cache: Dict[str, dict] = {}

    # Precompute subplot payloads
    panel_data = []
    for pair in pairs:
        dataset = pair["dataset"]
        source_group = pair["source_group"]
        if dataset not in dataset_cache:
            ddir = splits_root / dataset
            train_full = _read_lightgcn_user_items(ddir / "train_full.txt")
            test_items = _read_lightgcn_user_items(ddir / "test.txt")
            all_users = sorted(train_full.keys())
            all_items = {u: (train_full[u] | test_items[u]) for u in all_users}
            n_items = 1 + max((max(items) if items else -1) for items in all_items.values())
            label_key = DATASET_LABEL_KEY.get(dataset)
            if label_key is None:
                raise KeyError(f"No label-key mapping for dataset={dataset}")
            group_to_users = _load_user_group_map(ddir / "user_labels.pkl", label_key=label_key)
            group_v: dict[str, np.ndarray] = {}
            for g, users in group_to_users.items():
                grp_mat = _build_group_matrix(users, all_items, n_items)
                group_v[g] = _top_svd_v(grp_mat, max_rank=256)
            dataset_cache[dataset] = {
                "all_items": all_items,
                "group_to_users": group_to_users,
                "group_v": group_v,
            }
            if args.debug:
                print(f"[DEBUG] Loaded dataset={dataset}, users={len(all_users)}, groups={len(group_to_users)}")

        cache = dataset_cache[dataset]
        group_to_users = cache["group_to_users"]
        if source_group not in group_to_users:
            raise KeyError(f"source_group={source_group} not in labels for dataset={dataset}")

        test_dir = test_root / dataset / model / recdim
        top1_file = _find_test_eval_file(test_dir, source_group)
        alpha_map = _avg_alpha_mix_across_topk_for_source(top1_file, top_k=top_k)
        alpha_aug_avg = _avg_alpha_aug_across_topk_for_source(top1_file, top_k=top_k)

        aug_groups = _sort_group_labels(list(alpha_map.keys()))
        # Print pair-count summary before heavy similarity computation.
        print(
            f"[INFO] dataset={dataset} source_group={source_group} "
            f"n_source_users={len(group_to_users[source_group])}"
        )
        for aug in aug_groups:
            n_pairs = len(group_to_users[source_group]) * len(group_to_users.get(aug, []))
            print(
                f"[INFO]   source={source_group} vs aug={aug}: "
                f"n_aug_users={len(group_to_users.get(aug, []))}, n_user_pairs={n_pairs}"
            )

        sims = []
        alphas = []
        source_v = cache["group_v"][source_group]
        for aug in aug_groups:
            if aug not in group_to_users:
                raise KeyError(f"Aug group={aug} missing in labels for dataset={dataset}")
            key = _pair_key(dataset, source_group, aug)
            if key in similarity_cache:
                sim = float(similarity_cache[key])
                if args.debug:
                    print(f"[DEBUG] cache hit: {key} -> {sim:.8f}")
            else:
                aug_v = cache["group_v"][aug]
                sim = _normalized_discrepancy_vvt(source_v, aug_v, d=DISTANCE_RANK_D)
                similarity_cache[key] = float(sim)
                if args.debug:
                    print(f"[DEBUG] cache miss: {key}; computed {sim:.8f}")
            sims.append(sim)
            alphas.append(float(alpha_map[aug]))

        panel_data.append(
            {
                "dataset": dataset,
                "source_group": source_group,
                "aug_groups": aug_groups,
                "similarities": np.array(sims, dtype=float),
                "alpha_mix": np.array(alphas, dtype=float),
                "alpha_aug_avg": float(alpha_aug_avg),
            }
        )

        if args.debug:
            print(
                f"[DEBUG] panel dataset={dataset} source={source_group} aug_groups={aug_groups} "
                f"sim_range=({np.nanmin(sims):.4f},{np.nanmax(sims):.4f})"
            )

    n = len(panel_data)

    if args.two_column:
        # Figure A (two-column mode): scatter only, with dataset rows.
        dataset_order = []
        seen = set()
        for p in panel_data:
            ds = p["dataset"]
            if ds not in seen:
                dataset_order.append(ds)
                seen.add(ds)
        by_dataset: dict[str, list[dict]] = {ds: [] for ds in dataset_order}
        for p in panel_data:
            by_dataset[p["dataset"]].append(p)

        n_rows = len(dataset_order)
        n_cols = max(len(v) for v in by_dataset.values())
        fig_sc, axes_sc = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.3 * n_cols, 3.6 * n_rows),
            squeeze=False,
        )
        for r, ds in enumerate(dataset_order):
            row_panels = by_dataset[ds]
            for c in range(n_cols):
                ax = axes_sc[r, c]
                if c >= len(row_panels):
                    ax.axis("off")
                    continue
                panel = row_panels[c]
                sim = panel["similarities"]
                alpha = panel["alpha_mix"]
                groups = panel["aug_groups"]

                ax.scatter(sim, alpha, color="black", s=36)
                title_ds = DATASET_DISPLAY.get(panel["dataset"], panel["dataset"])
                ax.set_title(f"{title_ds}\nTarget Group: {panel['source_group']}")
                ax.set_xlabel("Normalized Discrepancy")
                ax.set_ylabel("Mixing Ratio")
                ax.grid(axis="both", linestyle="--", alpha=0.35, linewidth=0.7)
                for spine in ["top", "right"]:
                    ax.spines[spine].set_visible(False)

                for x, y, g in zip(sim, alpha, groups):
                    ax.annotate(
                        str(g),
                        xy=(x, y),
                        xytext=(6, 0),
                        textcoords="offset points",
                        fontsize=16,
                        ha="left",
                        va="center",
                    )

        fig_sc.tight_layout()
        out_sc = (
            out_dir
            / f"similarity_vs_topmix_scatter__{model}__recdim_{recdim}__topk_{top_k}__two_column.pdf"
        )
        fig_sc.savefig(out_sc, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {out_sc}")

        # Figure B (two-column mode): separate target-level alpha_aug bar chart.
        fig_aug, ax_aug = plt.subplots(1, 1, figsize=(max(6.0, 1.1 * n), 3.6))
        target_labels = [str(p["source_group"]) for p in panel_data]
        alpha_vals = [float(p["alpha_aug_avg"]) for p in panel_data]
        x_aug = np.arange(len(target_labels), dtype=float)
        ax_aug.bar(x_aug, alpha_vals, width=0.68, color="#6b7280", alpha=0.95)
        ax_aug.set_xticks(x_aug)
        ax_aug.set_xticklabels(target_labels)
        ax_aug.set_xlabel("Target Group")
        ax_aug.set_ylabel("Augmentation Ratio")
        ax_aug.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)
        for spine in ["top", "right"]:
            ax_aug.spines[spine].set_visible(False)

        fig_aug.tight_layout()
        out_aug = (
            out_dir
            / f"similarity_vs_topmix_target_aug_bar__{model}__recdim_{recdim}__topk_{top_k}__two_column.pdf"
        )
        fig_aug.savefig(out_aug, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {out_aug}")
    else:
        # Figure A: grouped bars
        fig_bar, axes_bar = plt.subplots(1, n + 1, figsize=(4.2 * n + 3.2, 4.3), squeeze=False)
        axes_bar = axes_bar[0]

        for i, panel in enumerate(panel_data):
            ax = axes_bar[i]
            groups = panel["aug_groups"]
            sim = panel["similarities"]
            alpha = panel["alpha_mix"]

            x = np.arange(len(groups), dtype=float)
            w = 0.38
            ax.bar(x - w / 2, sim, width=w, color="#111111", label="Similarity", alpha=0.95)
            ax.bar(
                x + w / 2,
                alpha,
                width=w,
                color="#d95f02",
                label=f"Avg Top-{top_k} Mix Alpha",
                alpha=0.90,
            )

            title_ds = DATASET_DISPLAY.get(panel["dataset"], panel["dataset"])
            ax.set_title(f"{title_ds}\nTarget Group: {panel['source_group']}")
            ax.set_xticks(x)
            ax.set_xticklabels(groups)
            ax.set_xlabel("Augmentation Group")
            ax.set_ylabel("Value")
            ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

            if i == 0:
                ax.legend(frameon=False)

        # Rightmost subplot: alpha_aug by target group.
        ax_aug = axes_bar[-1]
        target_labels = [str(p["source_group"]) for p in panel_data]
        alpha_vals = [float(p["alpha_aug_avg"]) for p in panel_data]
        x_aug = np.arange(len(target_labels), dtype=float)
        ax_aug.bar(x_aug, alpha_vals, width=0.68, color="#6b7280", alpha=0.95)
        ax_aug.set_xticks(x_aug)
        ax_aug.set_xticklabels(target_labels)
        ax_aug.set_xlabel("Target Group")
        ax_aug.set_ylabel("Augmentation Ratio")
        ax_aug.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)
        for spine in ["top", "right"]:
            ax_aug.spines[spine].set_visible(False)

        fig_bar.tight_layout()
        out_bar = out_dir / f"similarity_vs_topmix_grouped_bars__{model}__recdim_{recdim}__topk_{top_k}.pdf"
        fig_bar.savefig(out_bar, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {out_bar}")

        # Figure B: scatter
        fig_sc, axes_sc = plt.subplots(1, n + 1, figsize=(4.2 * n + 3.2, 4.3), squeeze=False)
        axes_sc = axes_sc[0]

        for i, panel in enumerate(panel_data):
            ax = axes_sc[i]
            sim = panel["similarities"]
            alpha = panel["alpha_mix"]
            groups = panel["aug_groups"]

            ax.scatter(sim, alpha, color="black", s=36)

            title_ds = DATASET_DISPLAY.get(panel["dataset"], panel["dataset"])
            ax.set_title(f"{title_ds}\nTarget Group: {panel['source_group']}")
            ax.set_xlabel("Normalized Discrepancy")
            ax.set_ylabel("Mixing Ratio")
            ax.grid(axis="both", linestyle="--", alpha=0.35, linewidth=0.7)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

            # light text labels for each augmentation group
            for x, y, g in zip(sim, alpha, groups):
                ax.annotate(
                    str(g),
                    xy=(x, y),
                    xytext=(6, 0),
                    textcoords="offset points",
                    fontsize=16,
                    ha="left",
                    va="center",
                )

        # Rightmost subplot: alpha_aug by target group.
        ax_aug2 = axes_sc[-1]
        target_labels = [str(p["source_group"]) for p in panel_data]
        alpha_vals = [float(p["alpha_aug_avg"]) for p in panel_data]
        x_aug = np.arange(len(target_labels), dtype=float)
        ax_aug2.bar(x_aug, alpha_vals, width=0.68, color="#6b7280", alpha=0.95)
        ax_aug2.set_xticks(x_aug)
        ax_aug2.set_xticklabels(target_labels)
        ax_aug2.set_xlabel("Target Group")
        ax_aug2.set_ylabel("Augmentation Ratio")
        ax_aug2.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)
        for spine in ["top", "right"]:
            ax_aug2.spines[spine].set_visible(False)

        fig_sc.tight_layout()
        out_sc = out_dir / f"similarity_vs_topmix_scatter__{model}__recdim_{recdim}__topk_{top_k}.pdf"
        fig_sc.savefig(out_sc, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {out_sc}")

    cache_path.write_text(json.dumps(similarity_cache, indent=2, sort_keys=True))
    print(f"Saved similarity cache: {cache_path}")


if __name__ == "__main__":
    main()
