#!/usr/bin/env python3
"""Prepare LastFM-Asia LightGCN splits from top-k largest country groups."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create LastFM-Asia train/val/test LightGCN splits after filtering "
            "to users from the top-k largest country labels."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/lastfm_asia"),
        help=(
            "Directory containing lastfm_asia_features.json and "
            "lastfm_asia_target.csv (default: data/lastfm_asia)."
        ),
    )
    parser.add_argument(
        "--top-k-countries",
        type=int,
        default=4,
        help="Keep only users from the top-k largest country labels (default: 4). Use -1 to keep all countries.",
    )
    parser.add_argument(
        "--countries-to-keep",
        type=str,
        default="",
        help=(
            "Comma-separated explicit country labels to keep (e.g. '17,10,0,6'). "
            "When provided, this overrides --top-k-countries."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("LightGCN/data/lastfm-asia"),
        help="Output directory for LightGCN split files.",
    )
    parser.add_argument(
        "--matrix-image-path",
        type=Path,
        default=None,
        help=(
            "Where to save the user-item interaction matrix image. "
            "Default: <output-dir>/interaction_matrix_by_country.png"
        ),
    )
    parser.add_argument(
        "--country-top-items-path",
        type=Path,
        default=None,
        help=(
            "Where to save JSON with per-country top-20 popular items. "
            "Default: <output-dir>/country_top20_items.json"
        ),
    )
    parser.add_argument(
        "--country-overlap-heatmap-path",
        type=Path,
        default=None,
        help=(
            "Where to save country top-20 overlap heatmap. "
            "Default: <output-dir>/country_top20_overlap_heatmap.png"
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random split seed.")
    parser.add_argument("--num-folds", type=int, default=5, help="Number of CV folds.")
    return parser.parse_args()


def _resolve(repo_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _resolve_data_dir(repo_root: Path, requested: Path) -> Path:
    resolved = _resolve(repo_root, requested)
    if resolved.is_dir():
        return resolved

    # Backward-compatible fallback for existing repo typo.
    typo_fallback = _resolve(repo_root, Path("data/lasftm_asia"))
    if typo_fallback.is_dir():
        print(
            f"[note] Requested data dir does not exist: {resolved}\n"
            f"       Falling back to: {typo_fallback}"
        )
        return typo_fallback
    return resolved


def load_interactions(features_path: Path) -> dict[int, list[int]]:
    with features_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    interactions: dict[int, list[int]] = {}
    for raw_uid, raw_items in raw.items():
        uid = int(raw_uid)
        if not isinstance(raw_items, list):
            raise AssertionError(f"User {uid} value is not a list.")
        items = [int(i) for i in raw_items]
        interactions[uid] = items
    return interactions


def load_user_labels(target_path: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    with target_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"id", "target"}
        if not reader.fieldnames or not required_cols.issubset(set(reader.fieldnames)):
            raise AssertionError(
                f"{target_path.name} must have headers including {sorted(required_cols)}."
            )
        for row in reader:
            uid = int(row["id"])
            label = str(row["target"])
            if uid in labels:
                raise AssertionError(f"Duplicate label entry for user {uid}.")
            labels[uid] = label
    return labels


def _assert_contiguous_indices(index_set: set[int], kind: str) -> int:
    if not index_set:
        raise AssertionError(f"No {kind} found.")
    min_idx = min(index_set)
    max_idx = max(index_set)
    if min_idx != 0:
        raise AssertionError(f"{kind} indices must start at 0, found min={min_idx}.")
    expected = set(range(max_idx + 1))
    missing = sorted(expected - index_set)
    if missing:
        preview = ", ".join(str(x) for x in missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise AssertionError(
            f"{kind} indices are not contiguous in [0, {max_idx}]. "
            f"Missing {len(missing)} indices: {preview}{suffix}"
        )
    return max_idx


def split_counts_test_only(n_items: int, num_folds: int) -> tuple[int, int]:
    if n_items < num_folds + 1:
        raise ValueError(
            "Cannot guarantee one held-out test item and non-empty val folds "
            f"for user with {n_items} interactions and num_folds={num_folds}."
        )
    test_n = max(1, int(round(0.2 * n_items)))
    test_n = min(test_n, n_items - num_folds)
    non_test_n = n_items - test_n
    if non_test_n < num_folds:
        raise ValueError(
            f"non_test_n={non_test_n} < num_folds={num_folds} for n_items={n_items}"
        )
    return non_test_n, test_n


def write_lightgcn_file(user_to_items: dict[int, list[int]], output_path: Path, n_users: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for user_idx in range(n_users):
            if user_idx not in user_to_items:
                raise AssertionError(f"User {user_idx} missing in {output_path.name}.")
            items = user_to_items[user_idx]
            if len(items) == 0:
                raise AssertionError(f"User {user_idx} has zero entries in {output_path.name}.")
            f.write(f"{user_idx} {' '.join(str(i) for i in items)}\n")


def assert_file_user_contiguity(path: Path, n_users: int) -> None:
    present_users: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            uid = int(line.split()[0])
            present_users.append(uid)
    if sorted(present_users) != list(range(n_users)):
        raise AssertionError(f"{path.name} users are not contiguous 0..{n_users - 1}.")


def prepare_lastfm_asia(
    data_dir: Path,
    output_dir: Path,
    matrix_image_path: Path | None = None,
    country_top_items_path: Path | None = None,
    country_overlap_heatmap_path: Path | None = None,
    countries_to_keep: list[str] | None = None,
    *,
    top_k_countries: int = 4,
    seed: int = 42,
    num_folds: int = 5,
) -> None:
    features_path = data_dir / "lastfm_asia_features.json"
    target_path = data_dir / "lastfm_asia_target.csv"
    if not features_path.is_file():
        raise FileNotFoundError(f"Missing file: {features_path}")
    if not target_path.is_file():
        raise FileNotFoundError(f"Missing file: {target_path}")

    if top_k_countries == -1:
        pass
    elif top_k_countries <= 0:
        raise ValueError("--top-k-countries must be >= 1, or -1 to keep all countries")
    if num_folds < 2:
        raise ValueError("--num-folds must be >= 2")

    interactions_raw = load_interactions(features_path)
    labels_raw = load_user_labels(target_path)

    # Keep only users with interactions.
    active_user_ids = sorted(uid for uid, items in interactions_raw.items() if len(items) > 0)
    if not active_user_ids:
        raise AssertionError("No users with interactions found.")

    # Select top-k country groups among active users.
    active_country_sizes = Counter(labels_raw[uid] for uid in active_user_ids if uid in labels_raw)
    if not active_country_sizes:
        raise AssertionError("No country labels found for users with interactions.")

    sorted_country_labels = [
        label for label, _ in sorted(active_country_sizes.items(), key=lambda x: (-x[1], x[0]))
    ]
    if countries_to_keep:
        # Preserve user-specified order while deduplicating.
        requested_labels: list[str] = []
        for label in countries_to_keep:
            label = str(label).strip()
            if label and label not in requested_labels:
                requested_labels.append(label)
        missing_labels = [label for label in requested_labels if label not in active_country_sizes]
        if missing_labels:
            raise KeyError(
                "Requested countries are not present among users with interactions: "
                f"{missing_labels}. Available: {sorted(sorted_country_labels)}"
            )
        top_country_labels = requested_labels
    else:
        top_country_labels = (
            sorted_country_labels if top_k_countries == -1 else sorted_country_labels[:top_k_countries]
        )
    top_country_set = set(top_country_labels)
    raw_user_ids = [uid for uid in active_user_ids if labels_raw.get(uid) in top_country_set]
    before_min_filter_n_users = len(raw_user_ids)
    min_required_interactions = num_folds + 1
    raw_user_ids = [
        uid for uid in raw_user_ids if len(set(interactions_raw.get(uid, []))) >= min_required_interactions
    ]
    if not raw_user_ids:
        raise AssertionError("No users remaining after top-country filtering.")

    missing_filtered_labels = [uid for uid in raw_user_ids if uid not in labels_raw]
    if missing_filtered_labels:
        preview = ", ".join(str(x) for x in missing_filtered_labels[:10])
        suffix = " ..." if len(missing_filtered_labels) > 10 else ""
        raise AssertionError(
            f"Missing group labels for {len(missing_filtered_labels)} filtered user(s): "
            f"{preview}{suffix}"
        )

    # Remap users to contiguous indices.
    user_id_to_idx = {old_uid: new_uid for new_uid, old_uid in enumerate(raw_user_ids)}
    interactions: dict[int, list[int]] = {user_id_to_idx[uid]: [] for uid in raw_user_ids}
    all_raw_item_ids: set[int] = set()

    for old_uid, raw_items in interactions_raw.items():
        if old_uid not in user_id_to_idx:
            continue
        new_uid = user_id_to_idx[old_uid]
        dedup_items = sorted({int(i) for i in raw_items})
        interactions[new_uid] = dedup_items
        all_raw_item_ids.update(dedup_items)

    if not all_raw_item_ids:
        raise AssertionError("No items found in interactions.")

    # Remap items to contiguous indices.
    raw_item_ids = sorted(all_raw_item_ids)
    item_id_to_idx = {old_iid: new_iid for new_iid, old_iid in enumerate(raw_item_ids)}
    for uid, raw_items in interactions.items():
        interactions[uid] = [item_id_to_idx[iid] for iid in raw_items]

    labels = {user_id_to_idx[old_uid]: labels_raw[old_uid] for old_uid in raw_user_ids}

    user_ids = set(interactions.keys())
    max_user_idx = _assert_contiguous_indices(user_ids, kind="User")
    n_users = max_user_idx + 1

    item_ids = {i for items in interactions.values() for i in items}
    max_item_idx = _assert_contiguous_indices(item_ids, kind="Item")
    n_items = max_item_idx + 1

    missing_labels = sorted(set(range(n_users)) - set(labels.keys()))
    if missing_labels:
        preview = ", ".join(str(x) for x in missing_labels[:10])
        suffix = " ..." if len(missing_labels) > 10 else ""
        raise AssertionError(
            f"Missing group labels for {len(missing_labels)} user(s): {preview}{suffix}"
        )

    # Build train/test and k-fold train/val splits.
    rng = np.random.default_rng(seed)
    train_full: dict[int, list[int]] = {}
    test: dict[int, list[int]] = {}
    fold_trains: list[dict[int, list[int]]] = [{} for _ in range(num_folds)]
    fold_vals: list[dict[int, list[int]]] = [{} for _ in range(num_folds)]

    for uid in range(n_users):
        user_items = np.array(interactions[uid], dtype=np.int64)
        non_test_n, test_n = split_counts_test_only(user_items.size, num_folds)
        shuffled = rng.permutation(user_items)
        non_test_items = shuffled[:non_test_n]
        test_items = shuffled[non_test_n : non_test_n + test_n]

        train_full[uid] = sorted(non_test_items.tolist())
        test[uid] = sorted(test_items.tolist())

        chunks = np.array_split(non_test_items, num_folds)
        if any(len(c) == 0 for c in chunks):
            raise AssertionError(f"User {uid} has empty fold chunk; increase interactions.")
        for fold_idx in range(num_folds):
            val_items = chunks[fold_idx]
            train_items = np.concatenate([chunks[i] for i in range(num_folds) if i != fold_idx])
            fold_vals[fold_idx][uid] = sorted(val_items.tolist())
            fold_trains[fold_idx][uid] = sorted(train_items.tolist())

            if len(fold_vals[fold_idx][uid]) == 0:
                raise AssertionError(f"Fold {fold_idx}: user {uid} has empty val split.")
            if len(fold_trains[fold_idx][uid]) == 0:
                raise AssertionError(f"Fold {fold_idx}: user {uid} has empty train split.")
            partition = (
                set(fold_vals[fold_idx][uid])
                | set(fold_trains[fold_idx][uid])
                | set(test[uid])
            )
            if partition != set(interactions[uid]):
                raise AssertionError(f"Fold {fold_idx}: partition mismatch for user {uid}.")

    # Global assertions on contiguous indices after splitting.
    all_items_train_test = {i for arr in train_full.values() for i in arr} | {
        i for arr in test.values() for i in arr
    }
    if all_items_train_test != set(range(n_items)):
        raise AssertionError("Item indices are not contiguous in train_full + test outputs.")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_lightgcn_file(train_full, output_dir / "train_full.txt", n_users=n_users)
    write_lightgcn_file(test, output_dir / "test.txt", n_users=n_users)
    write_lightgcn_file(fold_trains[0], output_dir / "train.txt", n_users=n_users)
    write_lightgcn_file(fold_vals[0], output_dir / "val.txt", n_users=n_users)
    for fold_idx in range(num_folds):
        write_lightgcn_file(
            fold_trains[fold_idx], output_dir / f"train_fold_{fold_idx}.txt", n_users=n_users
        )
        write_lightgcn_file(
            fold_vals[fold_idx], output_dir / f"val_fold_{fold_idx}.txt", n_users=n_users
        )

    # Explicit contiguity assertions on output user indices and non-empty fold rows.
    output_files = [
        output_dir / "train_full.txt",
        output_dir / "test.txt",
        output_dir / "train.txt",
        output_dir / "val.txt",
    ] + [output_dir / f"train_fold_{i}.txt" for i in range(num_folds)] + [
        output_dir / f"val_fold_{i}.txt" for i in range(num_folds)
    ]
    for p in output_files:
        assert_file_user_contiguity(p, n_users=n_users)

    # Write user labels pickle.
    label_to_users: dict[str, list[int]] = {label: [] for label in sorted(set(labels.values()))}
    for uid in range(n_users):
        label_to_users[labels[uid]].append(uid)
    labels_payload = {"Country": label_to_users}

    labels_path = output_dir / "user_labels.pkl"
    with labels_path.open("wb") as f:
        pickle.dump(labels_payload, f)

    country_sizes = Counter(labels[uid] for uid in range(n_users))

    # Save user-item interaction matrix image with users grouped by country.
    if matrix_image_path is None:
        matrix_image_path = output_dir / "interaction_matrix_by_country.png"
    matrix_image_path.parent.mkdir(parents=True, exist_ok=True)
    if country_top_items_path is None:
        country_top_items_path = output_dir / "country_top20_items.json"
    country_top_items_path.parent.mkdir(parents=True, exist_ok=True)
    if country_overlap_heatmap_path is None:
        country_overlap_heatmap_path = output_dir / "country_top20_overlap_heatmap.png"
    country_overlap_heatmap_path.parent.mkdir(parents=True, exist_ok=True)

    label_order = sorted(country_sizes.keys(), key=lambda x: (-country_sizes[x], str(x)))
    ordered_users: list[int] = []
    for label in label_order:
        ordered_users.extend(sorted(label_to_users[label]))

    if len(ordered_users) != n_users:
        raise AssertionError("Grouped user ordering does not cover all users.")

    img = np.full((n_users, n_items), 255, dtype=np.uint8)  # white background
    for row_idx, uid in enumerate(ordered_users):
        items = interactions[uid]
        img[row_idx, np.asarray(items, dtype=np.int64)] = 0  # black for interactions

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img, cmap="gray", aspect="auto", interpolation="nearest", vmin=0, vmax=255)

    # Draw horizontal lines between country blocks and label each block.
    boundaries: list[int] = []
    centers: list[float] = []
    ylabels: list[str] = []
    start = 0
    for label in label_order:
        size = len(label_to_users[label])
        end = start + size
        boundaries.append(end)
        centers.append((start + end - 1) / 2.0)
        ylabels.append(f"country {label} (n={size})")
        start = end

    for b in boundaries[:-1]:
        ax.axhline(b - 0.5, color="#ef4444", linewidth=0.9, alpha=0.95)

    ax.set_yticks(centers)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_title("LastFM-Asia Interaction Matrix (Users Grouped by Country)")
    ax.set_xlabel("Item Index")
    ax.set_ylabel("Users (country-grouped blocks)")
    fig.tight_layout()
    fig.savefig(matrix_image_path, dpi=220)
    plt.close()

    # Per-country top-20 popular items (by interaction count).
    top_k_items = 20
    country_item_counts: dict[str, Counter[int]] = {label: Counter() for label in label_order}
    for uid in range(n_users):
        label = labels[uid]
        country_item_counts[label].update(interactions[uid])

    country_top_items: dict[str, list[int]] = {}
    country_top_items_with_counts: dict[str, list[dict[str, int]]] = {}
    for label in label_order:
        ranked = sorted(country_item_counts[label].items(), key=lambda x: (-x[1], x[0]))[:top_k_items]
        country_top_items[label] = [int(item) for item, _ in ranked]
        country_top_items_with_counts[label] = [
            {"item_idx": int(item), "count": int(cnt)} for item, cnt in ranked
        ]

    with country_top_items_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "top_k": top_k_items,
                "country_top_items": country_top_items,
                "country_top_items_with_counts": country_top_items_with_counts,
            },
            f,
            indent=2,
        )

    # Country-by-country overlap matrix on top-20 sets.
    n_countries = len(label_order)
    overlap = np.zeros((n_countries, n_countries), dtype=int)
    top_sets = {label: set(country_top_items[label]) for label in label_order}
    for i, a in enumerate(label_order):
        for j, b in enumerate(label_order):
            overlap[i, j] = len(top_sets[a] & top_sets[b])

    cmap = LinearSegmentedColormap.from_list("darkred_to_white", ["#7f0000", "#ffffff"])
    fig2, ax2 = plt.subplots(figsize=(max(6, 0.6 * n_countries + 4), max(5, 0.6 * n_countries + 2)))
    im = ax2.imshow(overlap, cmap=cmap, vmin=0, vmax=top_k_items, aspect="auto")
    ax2.set_xticks(np.arange(n_countries))
    ax2.set_yticks(np.arange(n_countries))
    ax2.set_xticklabels([f"{lbl}" for lbl in label_order], rotation=45, ha="right")
    ax2.set_yticklabels([f"{lbl}" for lbl in label_order])
    ax2.set_xlabel("Country Label")
    ax2.set_ylabel("Country Label")
    ax2.set_title("Top-20 Item Overlap Between Country Pairs")
    cbar = fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Top-20 Overlap Size")
    for i in range(n_countries):
        for j in range(n_countries):
            val = int(overlap[i, j])
            txt_color = "white" if val <= 8 else "black"
            ax2.text(j, i, f"{val}", ha="center", va="center", color=txt_color, fontsize=8)
    fig2.tight_layout()
    fig2.savefig(country_overlap_heatmap_path, dpi=220)
    plt.close(fig2)

    # Summary stats are computed on the filtered user/item universe only.
    # At this point, `interactions` and `labels` already reflect country filtering.
    filtered_per_user_counts = [len(interactions[uid]) for uid in range(n_users)]
    total_interactions = int(sum(filtered_per_user_counts))
    per_user_counts_np = np.asarray(filtered_per_user_counts, dtype=float)
    pct_points = list(range(10, 100, 10))
    pct_values = np.percentile(per_user_counts_np, pct_points)

    print("LastFM-Asia Split Summary")
    print(f"- data_dir: {data_dir}")
    print(f"- output_dir: {output_dir}")
    print(f"- top_country_labels: {', '.join(top_country_labels)}")
    print(f"- min_interactions_per_user_for_{num_folds}_fold_cv: {min_required_interactions}")
    print(f"- users_dropped_for_low_interactions: {before_min_filter_n_users - len(raw_user_ids)}")
    print(f"- users: {n_users}")
    print(f"- items: {n_items}")
    print(f"- interactions: {total_interactions}")
    print(f"- interactions_per_user_mean: {mean(filtered_per_user_counts):.6f}")
    print(f"- interactions_per_user_median: {median(filtered_per_user_counts):.6f}")
    print("- interactions_per_user_percentiles_10pct:")
    for p, val in zip(pct_points, pct_values):
        print(f"  - p{p}: {float(val):.6f}")
    print("- country_label_sizes:")
    for label in sorted(country_sizes.keys(), key=lambda x: (-country_sizes[x], x)):
        print(f"  - {label}: {country_sizes[label]}")
    print(f"- wrote: {labels_path.name}")
    print(f"- matrix_image: {matrix_image_path}")
    print(f"- country_top_items_json: {country_top_items_path}")
    print(f"- country_overlap_heatmap: {country_overlap_heatmap_path}")
    print(f"- folds: {num_folds}")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = _resolve_data_dir(repo_root, args.data_dir)
    output_dir = _resolve(repo_root, args.output_dir)
    countries_to_keep = (
        [x.strip() for x in str(args.countries_to_keep).split(",") if x.strip()]
        if str(args.countries_to_keep).strip()
        else None
    )
    prepare_lastfm_asia(
        data_dir,
        output_dir,
        matrix_image_path=(
            _resolve(repo_root, args.matrix_image_path)
            if args.matrix_image_path is not None
            else None
        ),
        country_top_items_path=(
            _resolve(repo_root, args.country_top_items_path)
            if args.country_top_items_path is not None
            else None
        ),
        country_overlap_heatmap_path=(
            _resolve(repo_root, args.country_overlap_heatmap_path)
            if args.country_overlap_heatmap_path is not None
            else None
        ),
        countries_to_keep=countries_to_keep,
        top_k_countries=args.top_k_countries,
        seed=args.seed,
        num_folds=args.num_folds,
    )


if __name__ == "__main__":
    main()
