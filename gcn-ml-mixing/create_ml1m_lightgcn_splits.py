#!/usr/bin/env python3
"""Create user-level k-fold validation + test LightGCN splits for MovieLens-1M."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ml-1m LightGCN k-fold validation split files."
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
        help="Output directory for train.txt/val.txt/test.txt.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random split seed.")
    parser.add_argument("--num-folds", type=int, default=5, help="Number of validation folds.")
    return parser.parse_args()


def _resolve(repo_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def split_counts_test_only(n_items: int, num_folds: int) -> tuple[int, int]:
    """Return non_test_count and test_count, with non_test_count >= num_folds."""
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


def write_lightgcn_file(interactions: np.ndarray, output_path: Path) -> None:
    n_users, _ = interactions.shape
    with output_path.open("w", encoding="utf-8") as f:
        for user_idx in range(n_users):
            items = np.flatnonzero(interactions[user_idx] > 0)
            if items.size == 0:
                raise AssertionError(f"User {user_idx} has zero entries in {output_path.name}.")
            f.write(f"{user_idx} {' '.join(str(i) for i in items.tolist())}\n")


def load_movielens(repo_root: Path, data_dir: Path):
    sys.path.insert(0, str(repo_root))
    from loaders.movielens import movielens  # pylint: disable=import-error

    return movielens(min_ratings=0, min_users=0, binary=True, data_dir=str(data_dir) + "/")


def ensure_labels_pickle(ml_obj, output_dir: Path) -> Path:
    labels = {"Gender": ml_obj.get_user_labels("Gender"), "Age": ml_obj.get_user_labels("Age")}
    labels_path = output_dir / "user_labels.pkl"
    with labels_path.open("wb") as f:
        pickle.dump(labels, f)
    print(f"Wrote labels pickle: {labels_path}")
    return labels_path


def _max_indices_from_file(path: Path) -> tuple[int, int]:
    max_user = -1
    max_item = -1
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            uid = int(parts[0])
            items = [int(x) for x in parts[1:]]
            if not items:
                raise AssertionError(f"{path.name} has zero-item row for user {uid}.")
            max_user = max(max_user, uid)
            max_item = max(max_item, max(items))
    if max_user < 0 or max_item < 0:
        raise AssertionError(f"{path.name} appears empty.")
    return max_user, max_item


def assert_max_indices_equal(paths: list[Path]) -> None:
    max_map = {p.name: _max_indices_from_file(p) for p in paths}
    user_maxes = {name: pair[0] for name, pair in max_map.items()}
    item_maxes = {name: pair[1] for name, pair in max_map.items()}
    assert len(set(user_maxes.values())) == 1, f"Max user mismatch: {user_maxes}"
    assert len(set(item_maxes.values())) == 1, f"Max item mismatch: {item_maxes}"


def _file_contains_item(path: Path, item_idx: int) -> bool:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            items = [int(x) for x in parts[1:]]
            if item_idx in items:
                return True
    return False


def assert_max_item_present_in_training_files(training_paths: list[Path]) -> None:
    if not training_paths:
        raise AssertionError("No training paths provided.")
    max_item_idx = _max_indices_from_file(training_paths[0])[1]
    for p in training_paths:
        if not _file_contains_item(p, max_item_idx):
            raise AssertionError(
                f"{p.name} does not contain any interaction with max item index {max_item_idx}."
            )


def ensure_max_item_in_all_training_folds(
    fold_trains: list[np.ndarray], fold_vals: list[np.ndarray], max_item_idx: int
) -> None:
    """
    Override fold assignments so each training fold contains max_item_idx at least once.
    Keeps fold partition valid by swapping one train item with max_item between val/train.
    """
    for fold_idx in range(len(fold_trains)):
        train_f = fold_trains[fold_idx]
        val_f = fold_vals[fold_idx]
        if np.any(train_f[:, max_item_idx] > 0):
            continue

        # Find a user whose max item is in val for this fold and has another train item to swap out.
        candidate_users = np.where(val_f[:, max_item_idx] > 0)[0]
        fixed = False
        for user_idx in candidate_users:
            train_items = np.flatnonzero(train_f[user_idx] > 0)
            alt_items = train_items[train_items != max_item_idx]
            if alt_items.size == 0:
                continue
            alt_item = int(alt_items[0])
            # Swap: move max item to train, move one train item to val.
            val_f[user_idx, max_item_idx] = 0
            train_f[user_idx, max_item_idx] = 1
            train_f[user_idx, alt_item] = 0
            val_f[user_idx, alt_item] = 1
            fixed = True
            break

        if not fixed or not np.any(train_f[:, max_item_idx] > 0):
            raise AssertionError(
                f"Could not enforce max-item presence in training fold {fold_idx} "
                f"for item index {max_item_idx} without breaking partition."
            )


def create_splits(
    repo_root: Path,
    data_dir: Path,
    output_dir: Path,
    seed: int,
    num_folds: int = 5,
    write_labels: bool = True,
) -> None:
    if num_folds < 2:
        raise ValueError("num_folds must be >= 2.")
    ml = load_movielens(repo_root, data_dir)
    X = ml.get_X().astype(np.int8)
    n_users, n_items = X.shape
    rng = np.random.default_rng(seed)

    test = np.zeros_like(X, dtype=np.int8)
    train_full = np.zeros_like(X, dtype=np.int8)
    fold_vals = [np.zeros_like(X, dtype=np.int8) for _ in range(num_folds)]
    fold_trains = [np.zeros_like(X, dtype=np.int8) for _ in range(num_folds)]

    for user_idx in range(n_users):
        interacted = np.flatnonzero(X[user_idx] > 0)
        non_test_n, test_n = split_counts_test_only(interacted.size, num_folds=num_folds)
        shuffled = rng.permutation(interacted)
        non_test_items = shuffled[:non_test_n]
        test_items = shuffled[non_test_n : non_test_n + test_n]
        test[user_idx, test_items] = 1
        train_full[user_idx, non_test_items] = 1

        chunks = np.array_split(non_test_items, num_folds)
        if any(len(c) == 0 for c in chunks):
            raise AssertionError(f"User {user_idx} has empty fold chunk; increase interactions.")
        for fold_idx in range(num_folds):
            val_items = chunks[fold_idx]
            train_items = np.concatenate(
                [chunks[i] for i in range(num_folds) if i != fold_idx]
            )
            fold_vals[fold_idx][user_idx, val_items] = 1
            fold_trains[fold_idx][user_idx, train_items] = 1

    max_item_idx = int(np.max(np.flatnonzero(np.sum(X, axis=0) > 0)))
    ensure_max_item_in_all_training_folds(
        fold_trains=fold_trains, fold_vals=fold_vals, max_item_idx=max_item_idx
    )

    assert train_full.shape == test.shape
    assert np.all(train_full.sum(axis=1) > 0)
    assert np.all(test.sum(axis=1) > 0)
    for fold_idx in range(num_folds):
        train_f = fold_trains[fold_idx]
        val_f = fold_vals[fold_idx]
        assert np.all(train_f.sum(axis=1) > 0), f"Fold {fold_idx}: empty train user."
        assert np.all(val_f.sum(axis=1) > 0), f"Fold {fold_idx}: empty val user."
        assert np.array_equal(train_f + val_f + test, X), f"Fold {fold_idx}: partition mismatch."
        assert np.any(train_f[:, max_item_idx] > 0), (
            f"Fold {fold_idx}: training is missing max item index {max_item_idx}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_lightgcn_file(train_full, output_dir / "train_full.txt")
    write_lightgcn_file(test, output_dir / "test.txt")
    # Backward-compatible aliases.
    write_lightgcn_file(fold_trains[0], output_dir / "train.txt")
    write_lightgcn_file(fold_vals[0], output_dir / "val.txt")
    training_split_paths = [output_dir / "train_full.txt", output_dir / "train.txt"]
    for fold_idx in range(num_folds):
        train_path = output_dir / f"train_fold_{fold_idx}.txt"
        val_path = output_dir / f"val_fold_{fold_idx}.txt"
        write_lightgcn_file(fold_trains[fold_idx], train_path)
        write_lightgcn_file(fold_vals[fold_idx], val_path)
        training_split_paths.append(train_path)

    assert_max_indices_equal(training_split_paths)
    assert_max_item_present_in_training_files(training_split_paths)
    if write_labels:
        ensure_labels_pickle(ml, output_dir)

    print(f"Wrote LightGCN splits to: {output_dir}")
    print(f"Users: {n_users}, Items: {n_items}, Seed: {seed}, Folds: {num_folds}")


def ensure_splits_and_labels(
    repo_root: Path, data_dir: Path, output_dir: Path, seed: int, num_folds: int = 5
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    training_split_paths = [output_dir / "train_full.txt"]
    split_paths = [output_dir / "train_full.txt", output_dir / "test.txt"]
    for fold_idx in range(num_folds):
        split_paths.append(output_dir / f"train_fold_{fold_idx}.txt")
        split_paths.append(output_dir / f"val_fold_{fold_idx}.txt")
        training_split_paths.append(output_dir / f"train_fold_{fold_idx}.txt")
    missing_splits = [p for p in split_paths if not p.is_file()]
    if missing_splits:
        print("Missing split files detected; creating splits:")
        for p in missing_splits:
            print(f"  - {p}")
        create_splits(
            repo_root, data_dir, output_dir, seed=seed, num_folds=num_folds, write_labels=False
        )
    else:
        print("Reusing existing split files (no rewrite).")
        assert_max_indices_equal(training_split_paths)
        assert_max_item_present_in_training_files(training_split_paths)

    labels_pkl = output_dir / "user_labels.pkl"
    if labels_pkl.is_file():
        print(f"Reusing existing labels pickle: {labels_pkl}")
    else:
        print("Labels pickle missing; creating user_labels.pkl")
        ml = load_movielens(repo_root, data_dir)
        labels_pkl = ensure_labels_pickle(ml, output_dir)
    return labels_pkl


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = _resolve(repo_root, args.data_dir)
    output_dir = _resolve(repo_root, args.output_dir)
    create_splits(
        repo_root, data_dir, output_dir, seed=args.seed, num_folds=args.num_folds
    )


if __name__ == "__main__":
    main()
