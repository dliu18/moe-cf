"""Create cleaned Community Alignment splits by first-turn prompt.

This script reproduces the preprocessing used in:
`notebooks/community_alignment_exploratory.ipynb` by:
1) loading the raw Community Alignment CSV
2) filtering to rows where `assigned_lang == 'en'`
3) filtering to rows where `is_pregenerated_first_prompt` is true
4) splitting by distinct `first_turn_prompt` at the group level into
   70/10/20 train/val/test splits

Rows are partitioned by prompt group, so every row associated with a prompt
ends up in the same split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Community Alignment splits")
    parser.add_argument(
        "--input_csv",
        default="data/community_alignment.csv",
        help="Path to raw Community Alignment CSV.",
    )
    parser.add_argument(
        "--output_dir",
        default="data",
        help="Directory where cleaned data and splits are written.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed used to shuffle prompt groups before splitting.",
    )
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns in input CSV: "
            + ", ".join(sorted(missing))
        )


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    _require_columns(df, ["assigned_lang", "is_pregenerated_first_prompt", "first_turn_prompt"])

    df_en = df[df["assigned_lang"].astype(str).str.lower() == "en"].copy()
    is_first_prompt_true = (
        df_en["is_pregenerated_first_prompt"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )
    df_filtered = df_en[is_first_prompt_true].copy()
    df_filtered = df_filtered[df_filtered["first_turn_prompt"].notna()].copy()

    prompt_values = df_filtered["first_turn_prompt"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed=args.random_seed)
    rng.shuffle(prompt_values)

    n_prompts = len(prompt_values)
    n_train = int(0.70 * n_prompts)
    n_val = int(0.10 * n_prompts)
    n_test = n_prompts - n_train - n_val

    train_prompts = set(prompt_values[:n_train].tolist())
    val_prompts = set(prompt_values[n_train : n_train + n_val].tolist())
    test_prompts = set(prompt_values[n_train + n_val :].tolist())

    split_map = {
        "train": df_filtered[df_filtered["first_turn_prompt"].isin(train_prompts)].copy(),
        "val": df_filtered[df_filtered["first_turn_prompt"].isin(val_prompts)].copy(),
        "test": df_filtered[df_filtered["first_turn_prompt"].isin(test_prompts)].copy(),
    }

    # keep the complete filtered dataset for convenience/reproducibility
    filtered_path = output_dir / "community_alignment_en.csv"
    df_filtered.to_csv(filtered_path, index=False)

    split_paths = {
        "train": output_dir / "community_alignment_en_train.csv",
        "val": output_dir / "community_alignment_en_val.csv",
        "test": output_dir / "community_alignment_en_test.csv",
    }
    for split_name, split_df in split_map.items():
        split_df.to_csv(split_paths[split_name], index=False)

    print(f"Input rows: {len(df):,}")
    print(f"English rows: {len(df_en):,}")
    print(f"Pregenerated English rows: {len(df_filtered):,}")
    print(f"Unique first_turn_prompt groups: {n_prompts:,}")
    print(f"Split prompt counts: train={n_train:,}, val={n_val:,}, test={n_test:,}")
    print(f"Saved filtered dataset to: {filtered_path}")
    for split_name, split_path in split_paths.items():
        print(f"Saved {split_name} rows to: {split_path} ({len(split_map[split_name]):,})")


if __name__ == "__main__":
    main()
