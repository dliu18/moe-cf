#!/usr/bin/env python3
"""Intergroup representation distance vs embedding dimension.

For each dataset, constructs user-item interactions from train_full ∪ test,
computes group-level truncated SVDs once (up to rank 256), and plots
normalized projection distance curves:

    dist(i, j) = ||P_i - P_j||_F^2 / ||P_i||_F^2

where P_g = V_g Σ_g V_g^T from rank-d SVD of group adjacency A_g.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

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

SWEEP_DIMS = list(range(2, 257))
XTICK_DIMS = [2, 4, 8, 16, 32, 64, 128, 256]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot intergroup representation distance vs embedding dimension.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="ml-1m,lastfm-asia",
        help="Comma-separated datasets to run (default: ml-1m,lastfm-asia).",
    )
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=Path("LightGCN/data"),
        help="Root directory containing {dataset}/train_full.txt, test.txt, user_labels.pkl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("mixing/figures/pdfs/similarity_vs_dim"),
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _resolve(repo_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _read_lightgcn_file(path: Path) -> Dict[int, set[int]]:
    user_items: Dict[int, set[int]] = {}
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


def _load_group_to_users(labels_pkl: Path, label_key: str) -> Dict[str, List[int]]:
    payload = pickle.load(labels_pkl.open("rb"))
    if label_key not in payload:
        raise KeyError(f"{labels_pkl} missing label key {label_key}; keys={list(payload.keys())}")
    group_map = payload[label_key]
    if not isinstance(group_map, dict):
        raise TypeError(f"Expected dict at payload[{label_key}], got {type(group_map)}")
    return {str(k): [int(u) for u in v] for k, v in group_map.items()}


def _build_group_matrix(
    users: List[int],
    all_user_items: Dict[int, set[int]],
    n_items: int,
):
    """Return sparse CSR matrix rows=users, cols=items, values in {0,1}."""
    if SCIPY_AVAILABLE:
        data = []
        rows = []
        cols = []
        for r, uid in enumerate(users):
            items = all_user_items[uid]
            for it in items:
                rows.append(r)
                cols.append(it)
                data.append(1.0)
        if len(rows) == 0:
            return csr_matrix((len(users), n_items), dtype=np.float64)
        return csr_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols))), shape=(len(users), n_items))

    # Dense fallback when scipy is unavailable.
    mat = np.zeros((len(users), n_items), dtype=np.float64)
    for r, uid in enumerate(users):
        for it in all_user_items[uid]:
            mat[r, it] = 1.0
    return mat


def _top_svd_v_s(group_matrix, max_rank: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return V (n_items x k), s (k,) sorted descending; k <= max_rank."""
    if SCIPY_AVAILABLE:
        m, n = group_matrix.shape
        min_dim = min(m, n)
        if min_dim <= 1:
            return np.zeros((n, 0), dtype=np.float64), np.zeros((0,), dtype=np.float64)

        k = min(max_rank, min_dim - 1)
        if k < 1:
            return np.zeros((n, 0), dtype=np.float64), np.zeros((0,), dtype=np.float64)

        u, s, vt = svds(group_matrix, k=k)
        order = np.argsort(-s)
        s = s[order]
        vt = vt[order, :]
        v = vt.T
        return v.astype(np.float64, copy=False), s.astype(np.float64, copy=False)

    # Dense fallback: full SVD then truncate.
    arr = np.asarray(group_matrix, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((arr.shape[1], 0), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    u, s, vt = np.linalg.svd(arr, full_matrices=False)
    k = min(max_rank, len(s))
    return vt[:k, :].T, s[:k]


def _distance_from_truncated_factors(
    v_i: np.ndarray,
    s_i: np.ndarray,
    v_j: np.ndarray,
    s_j: np.ndarray,
    d: int,
    projection_mode: str,
) -> float:
    """Compute normalized projection distance using rank-d truncations.

    projection_mode:
      - "weighted": Pi = Vi Si Vi^T   (VΣV^T)
      - "subspace": Pi = Vi Vi^T      (VV^T)
    """
    di = min(d, len(s_i))
    dj = min(d, len(s_j))

    if di == 0:
        return np.nan

    vi = v_i[:, :di]
    vj = v_j[:, :dj]

    # M = Vi^T Vj, shape di x dj
    m = vi.T @ vj

    if projection_mode == "weighted":
        si = s_i[:di]
        sj = s_j[:dj]
        fro_i_sq = float(np.sum(si * si))
        if fro_i_sq == 0.0:
            return np.nan
        fro_j_sq = float(np.sum(sj * sj))
        # tr(Pi Pj) = tr(Si M Sj M^T) = sum_{a,b} si[a] * sj[b] * M[a,b]^2
        cross = float(np.sum((si[:, None] * sj[None, :]) * (m * m)))
        num = fro_i_sq + fro_j_sq - 2.0 * cross
        return num / fro_i_sq

    if projection_mode == "subspace":
        # Pi = Vi Vi^T, Pj = Vj Vj^T
        # ||Pi||_F^2 = tr(Pi^2) = tr(Pi) = di
        fro_i_sq = float(di)
        if fro_i_sq == 0.0:
            return np.nan
        fro_j_sq = float(dj)
        # tr(Pi Pj) = ||Vi^T Vj||_F^2
        cross = float(np.sum(m * m))
        num = fro_i_sq + fro_j_sq - 2.0 * cross
        return num / fro_i_sq

    raise ValueError(f"Unknown projection_mode: {projection_mode}")


def _sorted_group_labels(labels: List[str]) -> List[str]:
    def key_fn(x: str):
        try:
            return (0, int(x))
        except ValueError:
            return (1, x)

    return sorted(labels, key=key_fn)


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    splits_root = _resolve(repo_root, args.splits_root)
    out_dir = _resolve(repo_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if not datasets:
        raise ValueError("No datasets provided.")
    variance_store: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {
        "weighted": {},
        "subspace": {},
    }

    for dataset in datasets:
        if dataset not in DATASET_LABEL_KEY:
            raise ValueError(
                f"Unsupported dataset `{dataset}`. Supported: {sorted(DATASET_LABEL_KEY.keys())}"
            )

    # Figure style consistent with other figures.
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

    for dataset in datasets:
        label_key = DATASET_LABEL_KEY.get(dataset)
        if label_key is None:
            raise KeyError(f"Unsupported dataset in config: {dataset}")

        ds_dir = splits_root / dataset
        train_full = _read_lightgcn_file(ds_dir / "train_full.txt")
        test_items = _read_lightgcn_file(ds_dir / "test.txt")

        if set(train_full.keys()) != set(test_items.keys()):
            raise AssertionError(f"User mismatch between train_full and test for dataset={dataset}")

        # Union of train and test per user.
        all_user_items: Dict[int, set[int]] = {
            u: (train_full[u] | test_items[u]) for u in train_full.keys()
        }
        n_items = 1 + max((max(items) if items else -1) for items in all_user_items.values())
        if n_items <= 0:
            raise ValueError(f"No items found for dataset={dataset}")

        group_to_users = _load_group_to_users(ds_dir / "user_labels.pkl", label_key)
        group_labels = _sorted_group_labels(list(group_to_users.keys()))
        source_groups = group_labels.copy()

        if args.debug:
            print(f"[DEBUG] dataset={dataset}: users={len(all_user_items)}, items={n_items}, groups={len(group_labels)}")

        # Precompute rank-256-ish SVD for each group once; smaller d uses truncation.
        group_svd: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for g in group_labels:
            users = group_to_users[g]
            grp_mat = _build_group_matrix(users, all_user_items, n_items)
            v, s = _top_svd_v_s(grp_mat, max_rank=256)
            group_svd[g] = (v, s)
            if args.debug:
                print(f"[DEBUG] group={g}: n_users={len(users)}, svd_rank={len(s)}")

        # Stable color per augmentation group across subplots.
        cmap = plt.get_cmap("tab20")
        color_map = {g: cmap(i % 20) for i, g in enumerate(group_labels)}

        for projection_mode in ["weighted", "subspace"]:
            n_cols = len(source_groups)
            fig, axes = plt.subplots(1, n_cols, figsize=(5.0 * n_cols, 4.8), squeeze=False)
            axes = axes[0]
            variance_store[projection_mode][dataset] = {}

            for idx, source in enumerate(source_groups):
                ax = axes[idx]
                v_i, s_i = group_svd[source]

                aug_groups = [g for g in group_labels if g != source]
                aug_curves: List[np.ndarray] = []
                for aug in aug_groups:
                    v_j, s_j = group_svd[aug]
                    ys = [
                        _distance_from_truncated_factors(
                            v_i, s_i, v_j, s_j, d, projection_mode=projection_mode
                        )
                        for d in SWEEP_DIMS
                    ]
                    ys_arr = np.asarray(ys, dtype=np.float64)
                    aug_curves.append(ys_arr)
                    ax.plot(
                        SWEEP_DIMS,
                        ys_arr,
                        marker="o",
                        linewidth=1.8,
                        markersize=4.2,
                        color=color_map[aug],
                        label=aug,
                    )

                try:
                    ax.set_xscale("log", base=2)
                except TypeError:
                    # Backward compatibility for older matplotlib versions.
                    ax.set_xscale("log", basex=2)
                ax.set_xticks(XTICK_DIMS)
                ax.set_xticklabels([str(d) for d in XTICK_DIMS])
                ax.set_xlabel("Embedding Dimension")
                ax.set_ylabel("Normalized Group Distance")
                ax.set_title(f"Source Group: {source}", fontweight="bold")
                ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)
                for spine in ["top", "right"]:
                    ax.spines[spine].set_visible(False)

                if aug_curves:
                    variance_store[projection_mode][dataset][source] = np.nanvar(
                        np.vstack(aug_curves), axis=0
                    )
                else:
                    variance_store[projection_mode][dataset][source] = np.full(
                        len(SWEEP_DIMS), np.nan, dtype=np.float64
                    )

            # Shared legend at top, labels in ascending group order.
            legend_labels = group_labels
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    color=color_map[g],
                    marker="o",
                    linewidth=1.8,
                    markersize=4.2,
                )
                for g in legend_labels
            ]
            fig.legend(
                handles,
                legend_labels,
                loc="upper center",
                ncol=min(8, max(1, len(legend_labels))),
                frameon=False,
                bbox_to_anchor=(0.5, 0.995),
            )
            fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])

            suffix = "vsvt" if projection_mode == "weighted" else "vvt"
            out_file = out_dir / f"intergroup_similarity_vs_dim__{dataset}__{suffix}.pdf"
            fig.savefig(out_file, dpi=300, bbox_inches="tight")
            print(f"Saved figure: {out_file}")

    # Additional output: variance among augmentation-group distances per source group.
    for projection_mode in ["weighted", "subspace"]:
        n_cols = len(datasets)
        fig_var, axes_var = plt.subplots(1, n_cols, figsize=(5.6 * n_cols, 4.8), squeeze=False)
        axes_var = axes_var[0]

        for idx, dataset in enumerate(datasets):
            ax = axes_var[idx]
            dataset_sources = _sorted_group_labels(
                list(variance_store[projection_mode].get(dataset, {}).keys())
            )
            cmap = plt.get_cmap("tab20")
            source_color = {g: cmap(i % 20) for i, g in enumerate(dataset_sources)}

            for src in dataset_sources:
                ys = variance_store[projection_mode][dataset][src]
                ax.plot(
                    SWEEP_DIMS,
                    ys,
                    marker="o",
                    linewidth=1.8,
                    markersize=4.2,
                    color=source_color[src],
                    label=src,
                )

            try:
                ax.set_xscale("log", base=2)
            except TypeError:
                ax.set_xscale("log", basex=2)
            ax.set_xticks(XTICK_DIMS)
            ax.set_xticklabels([str(d) for d in XTICK_DIMS])
            ax.set_xlabel("Embedding Dimension")
            ax.set_ylabel("Variance of Group Distances")
            ax.set_title(DATASET_DISPLAY.get(dataset, dataset), fontweight="bold")
            ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.legend(
                loc="upper right",
                frameon=False,
                title="Source Group",
                ncol=1,
            )

        fig_var.tight_layout()
        suffix = "vsvt" if projection_mode == "weighted" else "vvt"
        out_var = out_dir / f"intergroup_distance_variance_vs_dim__{suffix}.pdf"
        fig_var.savefig(out_var, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {out_var}")


if __name__ == "__main__":
    main()
