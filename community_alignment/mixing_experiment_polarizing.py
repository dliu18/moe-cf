"""Polarization-aware mixing experiment for Community Alignment first-turn prompts.

Differences from the standard mixing experiment:
1) Test prompts are fixed and shared across all trials.
2) The fixed test set is the top-k most polarizing prompts between source and augmentation groups.
3) Polarization is cosine similarity of preference vectors, ranked by a weighted score
   so that prompts with larger (1-cosine) * (source_rows * augmentation_rows) appear first.
4) In addition to standard outputs, an HTML report with true and predicted preference
   vector charts is generated.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESPONSE_TEXT_COLUMNS = {
    "a": "first_turn_response_a",
    "b": "first_turn_response_b",
    "c": "first_turn_response_c",
    "d": "first_turn_response_d",
}
PREFERRED_RESPONSE_COLUMN = "first_turn_preferred_response"
PROMPT_COLUMN = "first_turn_prompt"
RESPONSE_COLUMNS = [
    "first_turn_response_a",
    "first_turn_response_b",
    "first_turn_response_c",
    "first_turn_response_d",
]


PROTECTED_ATTRIBUTE_OPTIONS = [
    "annotator_age",
    "annotator_gender",
    "annotator_education_level",
    "annotator_political",
    "annotator_ethnicity",
    "annotator_country",
]

SYSTEM_PROMPT = (
    "You are a personal assistant whose goal is to personalize your responses "
    "to the preferences of a user, based on a history of their preferred choices."
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Polarization-aware in-context mixing experiment"
    )
    parser.add_argument(
        "--protected_attribute",
        required=True,
        choices=PROTECTED_ATTRIBUTE_OPTIONS,
        help="Protected attribute column to condition on.",
    )
    parser.add_argument(
        "--source_group",
        required=True,
        help="Source group label to align with.",
    )
    parser.add_argument(
        "--augmentation_group",
        required=True,
        help="Separate augmentation group label (must differ from source group).",
    )
    parser.add_argument(
        "--training_prompts",
        type=int,
        default=-1,
        help="Number of training prompts to sample into example set; -1 => max.",
    )
    parser.add_argument(
        "--test_prompts",
        type=int,
        default=-1,
        help="Number of polarizing test prompts to use; -1 => all polarizing prompts.",
    )
    parser.add_argument(
        "--test_set",
        choices=["val", "test", "both"],
        default="test",
        help="Which split to build the test pool from.",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=1,
        help="Number of trials for each alpha value.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Single alpha value (0.0..1.0); if provided, skip alpha sweep.",
    )
    parser.add_argument(
        "--alpha_step",
        type=float,
        default=0.05,
        help="Step size for alpha sweep when --alpha is omitted.",
    )
    parser.add_argument(
        "--train_path",
        default="data/community_alignment_en_train.csv",
        help="Path to pre-built training split.",
    )
    parser.add_argument(
        "--val_path",
        default="data/community_alignment_en_val.csv",
        help="Path to pre-built validation split.",
    )
    parser.add_argument(
        "--test_path",
        default="data/community_alignment_en_test.csv",
        help="Path to pre-built test split.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Base random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="If set, skip API calls and use synthetic predictions.",
    )
    parser.add_argument(
        "--simulate_prediction_mode",
        choices=["all_a", "random", "ground_truth"],
        default="all_a",
        help=(
            "'all_a' returns all a; 'random' samples a-d uniformly; "
            "'ground_truth' returns ground truth labels."
        ),
    )
    parser.add_argument(
        "--model_name",
        default="gpt-5-mini",
        help="OpenAI model for API calls.",
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=25000,
        help="OpenAI max completion token cap (uses max_completion_tokens for GPT-5-style calls).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write exact API payload + metadata to debug log.",
    )
    parser.add_argument(
        "--log_call_metrics",
        action="store_true",
        help="Print per-call response/token metrics.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/community_alignment_mixing_polarizing",
        help="Directory where outputs are written.",
    )
    parser.add_argument(
        "--debug_log_path",
        default=None,
        help="Path for debug JSONL logs.",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Output CSV path for trial-level results.",
    )
    parser.add_argument(
        "--plot_path",
        default=None,
        help="Output path for accuracy scatter plot.",
    )
    parser.add_argument(
        "--html_path",
        default=None,
        help="Output path for HTML report.",
    )
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(sorted(missing))}")


def _safe_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _canonical_preferred(response_value: object) -> str:
    value = _safe_text(response_value).lower()
    if value.startswith("response_") and len(value) == 10 and value[-1] in RESPONSE_TEXT_COLUMNS:
        return value[-1]
    if value in RESPONSE_TEXT_COLUMNS:
        return value
    return ""


def _majority_label_from_counts(counts: np.ndarray) -> str:
    if counts.sum() == 0:
        return ""
    max_val = counts.max()
    if max_val == 0:
        return ""
    candidates = [k for k, v in zip("abcd", counts) if v == max_val]
    return sorted(candidates)[0]


def _count_tokens_local(text: str) -> int:
    if text is None:
        return 0
    return len(re.findall(r"\S+", str(text)))


def _estimate_completion_tokens(expected_test_rows: int) -> int:
    return max(1, 2 * int(expected_test_rows) - 1)


def _serialize_usage(usage) -> dict[str, object]:
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "dict"):
        return usage.dict()
    if isinstance(usage, dict):
        return usage
    return {"raw": str(usage)}


def _run_api_call(payload: dict[str, object]) -> tuple[str, int, int, str | None, str | None, str | None, dict[str, object]]:
    from openai import OpenAI

    client = OpenAI()
    completion = client.chat.completions.create(**payload)
    message = completion.choices[0].message
    api_text = message.content or ""
    finish_reason = completion.choices[0].finish_reason
    refusal_text = getattr(message, "refusal", None)
    response_id = getattr(completion, "id", None)
    if not completion.usage:
        raise RuntimeError("OpenAI usage data unavailable for real API call.")
    usage_payload = _serialize_usage(completion.usage)
    return (
        api_text,
        int(completion.usage.prompt_tokens),
        int(completion.usage.completion_tokens),
        finish_reason,
        refusal_text,
        response_id,
        usage_payload,
    )


def _build_alpha_values(alpha_step: float) -> list[float]:
    if not (0 < alpha_step <= 1):
        raise ValueError("alpha_step must be in (0, 1].")
    if alpha_step == 1.0:
        return [0.0, 1.0]
    alpha_values = []
    current = 0.0
    safety_iter = 0
    while current < 1.0 - 1e-12 and safety_iter < 20000:
        alpha_values.append(round(current, 10))
        current += alpha_step
        safety_iter += 1
    alpha_values.append(1.0)
    return sorted({min(1.0, max(0.0, a)) for a in alpha_values})


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)


def _preference_counts(rows: pd.DataFrame) -> np.ndarray:
    counts = np.zeros(4, dtype=np.int64)
    labels = rows[PREFERRED_RESPONSE_COLUMN].map(_canonical_preferred)
    for label in labels:
        if label in RESPONSE_TEXT_COLUMNS:
            idx = ord(label) - ord("a")
            counts[idx] += 1
    return counts


def _sample_rows(df: pd.DataFrame, n: int, rng: np.random.Generator, seed_label: str) -> pd.DataFrame:
    if n == 0:
        return df.iloc[0:0].copy()
    if n > len(df):
        raise ValueError(
            f"Cannot sample {n} rows for {seed_label}; only {len(df)} rows available."
        )
    return df.sample(n=n, random_state=int(rng.integers(low=0, high=2**31 - 1)), replace=False)


def _prepare_train_rows(train_df: pd.DataFrame, protected_attribute: str, source_group: str, augmentation_group: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_rows = train_df[train_df[protected_attribute] == source_group].copy()
    aug_rows = train_df[train_df[protected_attribute] == augmentation_group].copy()

    def keep_valid_rows(rows: pd.DataFrame) -> pd.DataFrame:
        rows = rows.copy()
        rows = rows[rows[PREFERRED_RESPONSE_COLUMN].map(_canonical_preferred).astype(bool)]
        rows = rows[rows[PROMPT_COLUMN].notna()]
        return rows

    return keep_valid_rows(source_rows), keep_valid_rows(aug_rows)


def _resolve_test_pool(args: argparse.Namespace) -> pd.DataFrame:
    val_df = pd.read_csv(args.val_path) if Path(args.val_path).exists() else pd.DataFrame()
    test_df = pd.read_csv(args.test_path) if Path(args.test_path).exists() else pd.DataFrame()
    if args.test_set == "both":
        if val_df.empty and test_df.empty:
            raise FileNotFoundError("Both val and test split files are missing.")
        if val_df.empty:
            return test_df
        if test_df.empty:
            return val_df
        return pd.concat([val_df, test_df], axis=0, ignore_index=True)
    if args.test_set == "val":
        if val_df.empty:
            raise FileNotFoundError(f"Validation split not found: {args.val_path}")
        return val_df
    if args.test_set == "test":
        if test_df.empty:
            raise FileNotFoundError(f"Test split not found: {args.test_path}")
        return test_df
    raise ValueError(f"Unsupported test_set value: {args.test_set}")


def _select_polarizing_prompts(
    test_pool: pd.DataFrame,
    protected_attribute: str,
    source_group: str,
    augmentation_group: str,
    test_prompts: int,
) -> pd.DataFrame:
    required = [protected_attribute, PROMPT_COLUMN, PREFERRED_RESPONSE_COLUMN] + RESPONSE_COLUMNS
    _require_columns(test_pool, required, "Resolved test pool")

    pool = test_pool.copy()
    pool = pool[pool[PROMPT_COLUMN].notna()]
    pool = pool[pool[PREFERRED_RESPONSE_COLUMN].map(_canonical_preferred).astype(bool)]
    if pool.empty:
        raise ValueError("No valid rows available in test pool after filtering.")

    records = []
    for prompt, g in pool.groupby(PROMPT_COLUMN, sort=False):
        source_rows = g[g[protected_attribute] == source_group]
        aug_rows = g[g[protected_attribute] == augmentation_group]
        if source_rows.empty or aug_rows.empty:
            continue
        source_labels = [
            _canonical_preferred(v)
            for v in source_rows[PREFERRED_RESPONSE_COLUMN].to_numpy()
            if _canonical_preferred(v)
        ]
        aug_labels = [
            _canonical_preferred(v)
            for v in aug_rows[PREFERRED_RESPONSE_COLUMN].to_numpy()
            if _canonical_preferred(v)
        ]
        if not source_labels or not aug_labels:
            continue

        source_counts = np.zeros(4, dtype=np.int64)
        aug_counts = np.zeros(4, dtype=np.int64)
        for label in source_labels:
            source_counts[ord(label) - ord("a")] += 1
        for label in aug_labels:
            aug_counts[ord(label) - ord("a")] += 1

        source_total = float(source_counts.sum())
        aug_total = float(aug_counts.sum())
        source_vec = source_counts / source_total
        aug_vec = aug_counts / aug_total
        cos = _cosine_similarity(source_vec, aug_vec)
        cross_support = source_total * aug_total
        weighted_polarization = (1.0 - cos) * cross_support
        row_sample = g.iloc[0]
        records.append(
            {
                PROMPT_COLUMN: _safe_text(prompt),
                "response_a": _safe_text(row_sample["first_turn_response_a"]),
                "response_b": _safe_text(row_sample["first_turn_response_b"]),
                "response_c": _safe_text(row_sample["first_turn_response_c"]),
                "response_d": _safe_text(row_sample["first_turn_response_d"]),
                "source_preference_vector": source_vec.tolist(),
                "augmentation_preference_vector": aug_vec.tolist(),
                "source_preferred_labels": source_labels,
                "polarization_score": cos,
                "weighted_polarization_score": weighted_polarization,
                "source_count": int(source_counts.sum()),
                "augmentation_count": int(aug_counts.sum()),
                "n_rows": int(len(g)),
            }
        )

    if not records:
        raise ValueError(
            "No prompt has both source and augmentation rows with valid preferred labels."
        )

    records = sorted(
        records,
        key=lambda r: (
            -r["weighted_polarization_score"],
            -r["n_rows"],
            r[PROMPT_COLUMN],
        ),
    )
    if test_prompts != -1:
        if test_prompts <= 0:
            records = []
        else:
            records = records[:test_prompts]
    return pd.DataFrame(records).reset_index(drop=True)


def _build_example_block(examples: pd.DataFrame) -> str:
    lines = [
        "### Start Examples",
        "",
    ]
    for _, row in examples.reset_index(drop=True).iterrows():
        preferred = _safe_text(row[PREFERRED_RESPONSE_COLUMN])
        if preferred in RESPONSE_TEXT_COLUMNS and row.get(RESPONSE_TEXT_COLUMNS[preferred]):
            preferred = _safe_text(row[RESPONSE_TEXT_COLUMNS[preferred]])
        elif preferred.startswith("response_") and preferred[-1] in RESPONSE_TEXT_COLUMNS:
            preferred = _safe_text(row[RESPONSE_TEXT_COLUMNS[preferred[-1]]])
        lines.append(f"Prompt: {row[PROMPT_COLUMN]}")
        lines.append(f"Preferred Response: {preferred}")
        lines.append("")
    lines.append("### End Examples")
    return "\n".join(lines)


def _build_test_block(test_prompts: pd.DataFrame) -> str:
    lines = ["### Start Test Prompts", ""]
    for _, row in test_prompts.iterrows():
        lines.append(f"Prompt: {row[PROMPT_COLUMN]}")
        lines.append(f"Response a: {row['response_a']}")
        lines.append(f"Response b: {row['response_b']}")
        lines.append(f"Response c: {row['response_c']}")
        lines.append(f"Response d: {row['response_d']}")
        lines.append("")
    lines.append("### End Test Prompts")
    return "\n".join(lines)


def _build_call_payload(
    examples: pd.DataFrame,
    test_prompts: pd.DataFrame,
    include_preference_context: bool,
) -> list[dict[str, str]]:
    example_block = _build_example_block(examples)
    test_block = _build_test_block(test_prompts)
    prompt_count = len(test_prompts)
    prefix = (
        "Now, taking into account these above preferred choices, "
        if include_preference_context
        else "Now, "
    )
    instruction_block = (
        f"{prefix}select the most appropriate potential response from each of the "
        f"following {prompt_count} prompts. Answer with just a sequence of chosen "
        f"responses e.g. a,c,d,b,a,c,... The response should be {prompt_count} letters "
        "separated by commas."
    )
    user_prompt = (
        f"{example_block}\n\n"
        f"{instruction_block}\n\n"
        f"{test_block}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _build_openai_request(
    messages: list[dict[str, str]],
    model_name: str,
    temperature: float = 0.0,
    max_completion_tokens: int = 25000,
) -> dict[str, object]:
    request = {
        "model": model_name,
        "messages": messages,
    }
    if model_name.startswith("gpt-5"):
        request["max_completion_tokens"] = max_completion_tokens
    else:
        request["temperature"] = temperature
        request["max_tokens"] = max_completion_tokens
    return request


def _extract_predictions(response: str) -> list[str]:
    if not response:
        return []
    tokens = re.findall(r"[a-dA-D]", response)
    return [tok.lower() for tok in tokens if tok.lower() in RESPONSE_TEXT_COLUMNS]


def _parse_predictions(response: str, n_expected: int) -> list[str]:
    return _extract_predictions(response)[:n_expected]


def _simulate_predictions(
    n_expected: int,
    mode: str,
    expected: np.ndarray,
    rng: np.random.Generator,
) -> list[str]:
    if n_expected <= 0:
        return []
    if mode == "all_a":
        return ["a"] * n_expected
    if mode == "ground_truth":
        if n_expected == 0:
            return []
        if isinstance(expected[0], (list, tuple, np.ndarray)):
            predicted: list[str] = []
            for prompt_labels in expected[:n_expected]:
                if isinstance(prompt_labels, (list, tuple, np.ndarray)) and len(prompt_labels) > 0:
                    predicted.append(_safe_text(prompt_labels[0]))
                else:
                    predicted.append("")
            return predicted
        return [str(v) for v in expected[:n_expected]]
    return [str(chr(ord("a") + int(rng.integers(0, 4)))) for _ in range(n_expected)]


def _write_debug_entry(debug_fp, payload: dict[str, object], meta: dict[str, object] | None = None) -> None:
    entry = {"api_call": payload}
    if meta:
        entry["meta"] = meta
    debug_fp.write(json.dumps(entry, ensure_ascii=False, indent=2) + "\n\n")


def _run_single_trial(
    *,
    args: argparse.Namespace,
    alpha_label: float | str,
    trial: int,
    selected_source: pd.DataFrame,
    selected_aug: pd.DataFrame,
    test_prompt_df: pd.DataFrame,
    rng: np.random.Generator,
    training_budget: int,
    output_rows: list[dict[str, object]],
    total_input_tokens_ref: dict[str, int],
    total_completion_tokens_ref: dict[str, int],
    total_calls_ref: dict[str, int],
    debug_fp,
) -> tuple[dict[str, object], np.ndarray | None]:
    examples = pd.concat([selected_source, selected_aug], axis=0, ignore_index=True)
    if test_prompt_df.empty:
        raise ValueError("Fixed polarizing test prompt set is empty.")

    is_baseline = alpha_label == "baseline" or len(examples) == 0
    messages = _build_call_payload(examples, test_prompt_df, include_preference_context=not is_baseline)
    api_call_payload = _build_openai_request(
        messages=messages,
        model_name=args.model_name,
        temperature=0.0,
        max_completion_tokens=args.max_completion_tokens,
    )
    estimated_input_tokens = _count_tokens_local(messages[0]["content"]) + _count_tokens_local(messages[1]["content"])

    expected_labels = np.array(
        [
            [ _safe_text(label) for label in labels ]
            for labels in test_prompt_df["source_preferred_labels"].to_numpy()
        ],
        dtype=object,
    )
    expected_len = len(test_prompt_df)
    flat_expected_count = int(sum(len(labels) for labels in expected_labels))
    api_response = ""
    predictions: list[str] = []
    actual_input_tokens = 0
    completion_tokens = 0
    api_finish_reason: str | None = None
    api_refusal: str | None = None
    api_response_id: str | None = None
    usage_payload: dict[str, object] | None = None
    parsed_prediction_count = 0

    if args.simulate:
        predictions = _simulate_predictions(expected_len, args.simulate_prediction_mode, expected_labels, rng)
        parsed_prediction_count = len(predictions)
        completion_tokens = _estimate_completion_tokens(expected_len)
        actual_input_tokens = estimated_input_tokens
        call_tokens = estimated_input_tokens + completion_tokens
        api_finish_reason = "stop"
    else:
        (
            api_response,
            prompt_tokens,
            completion_tokens,
            api_finish_reason,
            api_refusal,
            api_response_id,
            usage_payload,
        ) = _run_api_call(api_call_payload)
        actual_input_tokens = int(prompt_tokens)
        parsed_predictions = _extract_predictions(api_response)
        parsed_prediction_count = len(parsed_predictions)
        predictions = _parse_predictions(api_response, expected_len)
        call_tokens = prompt_tokens + completion_tokens

    total_calls_ref["value"] += 1
    total_input_tokens_ref["value"] += actual_input_tokens
    total_completion_tokens_ref["value"] += completion_tokens

    is_success = api_finish_reason == "stop"
    is_sequence_length_valid = parsed_prediction_count == expected_len
    if expected_len == 0:
        num_correct = 0
        accuracy = float("nan")
    elif is_success and is_sequence_length_valid:
        expected_flat: list[str] = []
        pred_flat: list[str] = []
        for i, labels in enumerate(expected_labels):
            prompt_pred = predictions[i] if i < len(predictions) else ""
            expected_flat.extend(labels)
            pred_flat.extend([prompt_pred] * len(labels))
        expected_flat_arr = np.array(expected_flat, dtype=str)
        pred_flat_arr = np.array(pred_flat, dtype=str)
        n_expected_annotations = len(expected_flat_arr)
        if n_expected_annotations == 0:
            num_correct = 0
            accuracy = float("nan")
        else:
            num_correct = int(np.sum(pred_flat_arr == expected_flat_arr))
            accuracy = num_correct / n_expected_annotations
    else:
        num_correct = None
        accuracy = float("nan")

    predicted_one_hot = np.zeros((expected_len, 4), dtype=float)
    valid_vector = is_success and is_sequence_length_valid and len(predictions) == expected_len
    if valid_vector:
        for i, p in enumerate(predictions):
            predicted_one_hot[i, ord(p) - ord("a")] = 1.0

    record = {
        "alpha": alpha_label,
        "trial": trial + 1,
        "training_budget": training_budget,
        "n_examples_source": len(selected_source),
        "n_examples_augmentation": len(selected_aug),
        "n_test_prompts": expected_len,
        "num_correct": num_correct,
        "accuracy": accuracy,
        "estimated_call_tokens": call_tokens,
        "estimated_input_tokens": estimated_input_tokens,
        "actual_input_tokens": actual_input_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": api_finish_reason,
        "success": bool(is_success),
        "is_sequence_length_valid": bool(is_sequence_length_valid),
        "predicted_response_count": len(predictions),
        "expected_response_count": int(flat_expected_count),
        "predicted_labels": ",".join(predictions),
        "predicted_one_hot": json.dumps(predicted_one_hot.tolist()),
        "sequence_vector_valid_for_aggregation": bool(valid_vector),
    }
    output_rows.append(record)

    if args.debug:
        meta = {
            "alpha": alpha_label,
            "trial": trial + 1,
            "training_budget": training_budget,
            "token_count": call_tokens,
            "n_examples_source": len(selected_source),
            "n_examples_augmentation": len(selected_aug),
            "n_test_prompts": expected_len,
            "estimated_input_tokens": estimated_input_tokens,
            "actual_input_tokens": actual_input_tokens,
            "simulate": bool(args.simulate),
            "simulate_prediction_mode": args.simulate_prediction_mode,
            "api_mode": "simulated" if args.simulate else "live_api",
            "sequence_length_valid": bool(is_sequence_length_valid),
            "predicted_response_count": len(predictions),
            "expected_response_count": int(flat_expected_count),
            "predicted_one_hot": json.dumps(predicted_one_hot.tolist()),
            "vector_aggregation_valid": bool(valid_vector),
        }
        if api_response:
            meta["api_response"] = api_response
        if api_finish_reason is not None:
            meta["finish_reason"] = api_finish_reason
        if api_refusal is not None:
            meta["refusal"] = api_refusal
        if api_response_id is not None:
            meta["response_id"] = api_response_id
        if usage_payload is not None:
            meta["completion_tokens"] = completion_tokens
            meta["usage"] = usage_payload
        if args.simulate:
            meta["simulated_predictions"] = predictions
        _write_debug_entry(debug_fp, api_call_payload, meta)

    if args.log_call_metrics and not args.simulate:
        metric_accuracy = "n/a (non-stop or invalid length)" if not (is_success and is_sequence_length_valid) else f"{accuracy:.4f}"
        print(
            f"alpha={alpha_label} trial={trial+1} "
            f"train_ex={len(examples):,}, "
            f"test_prompts={expected_len:,}, "
            f"tokens={call_tokens:,}"
        )
        print(
            "Call metrics:"
            f"\n  response: {api_response!r}"
            f"\n  accuracy: {metric_accuracy}"
            f"\n  estimated_input_tokens: {estimated_input_tokens}"
            f"\n  actual_input_tokens: {actual_input_tokens}"
            f"\n  completion_tokens: {completion_tokens}"
            f"\n  predicted_response_count: {len(predictions)}"
            f"\n  expected_response_count: {expected_len}"
            f"\n  sequence_length_valid: {is_sequence_length_valid}"
            f"\n  finish_reason: {api_finish_reason}"
            f"\n  response_id: {api_response_id}"
            f"\n  refusal: {api_refusal}"
        )

    return record, (predicted_one_hot if valid_vector else None)


def _figure_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _build_vector_chart(true_source: list[float], true_aug: list[float], title: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    idx = np.arange(4)
    labels = ["a", "b", "c", "d"]
    width = 0.35
    ax.bar(idx - width / 2, true_source, width, label="Source")
    ax.bar(idx + width / 2, true_aug, width, label="Augmentation")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Preference probability")
    ax.set_title(title)
    ax.legend(loc="upper right")
    return _figure_to_base64(fig)


def _build_predicted_chart(pred_vec: list[float], title: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 2.8))
    idx = np.arange(4)
    labels = ["a", "b", "c", "d"]
    ax.bar(idx, pred_vec, width=0.65)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Predicted probability")
    ax.set_title(title)
    return _figure_to_base64(fig)


def _build_html_report(
    out_path: Path,
    args: argparse.Namespace,
    test_prompt_df: pd.DataFrame,
    alpha_values: list[float | str],
    pred_sum_by_alpha: dict[float | str, np.ndarray],
    pred_count_by_alpha: dict[float | str, np.ndarray],
    out_csv: Path,
) -> None:
    html_lines = [
        "<html>",
        "<head><meta charset='utf-8'><title>Community Alignment Polarizing Mixing Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 24px; background: #f6f7fb; }",
        ".prompt { border: 1px solid #c8cbd6; border-radius: 8px; padding: 14px; margin-bottom: 22px; background: white; }",
        ".prompt h3 { margin-top: 0; }",
        "table { border-collapse: collapse; width: 100%; }",
        "td, th { border: 1px solid #ddd; padding: 8px; vertical-align: top; }",
        "th { background-color: #f0f0f0; }",
        "img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }",
        ".meta { margin-bottom: 20px; }",
        "</style></head>",
        "<body>",
        "<h1>Community Alignment Polarizing Mixing Experiment</h1>",
        "<div class='meta'>",
        f"<p><b>Protected attribute:</b> {args.protected_attribute}</p>",
        f"<p><b>Source group:</b> {args.source_group}</p>",
        f"<p><b>Augmentation group:</b> {args.augmentation_group}</p>",
        f"<p><b>Model:</b> {args.model_name}</p>",
        f"<p><b>Test prompts:</b> {len(test_prompt_df)}</p>",
        f"<p><b>Output CSV:</b> {out_csv}</p>",
        f"<p><b>Output directory:</b> {args.output_dir}</p>",
        "</div>",
    ]

    report_alpha_values: list[float | str] = []
    for value in alpha_values:
        if value not in report_alpha_values:
            report_alpha_values.append(value)
    if "baseline" in pred_sum_by_alpha and "baseline" not in report_alpha_values:
        report_alpha_values.append("baseline")

    for idx, row in test_prompt_df.iterrows():
        prompt_text = _safe_text(row[PROMPT_COLUMN])
        true_source = row["source_preference_vector"]
        true_aug = row["augmentation_preference_vector"]
        html_lines.append("<div class='prompt'>")
        html_lines.append(f"<h3>Test Prompt {idx + 1}</h3>")
        html_lines.append(f"<p><b>Prompt:</b> {prompt_text}</p>")
        html_lines.append("<table>")
        html_lines.append("<tr><th>Option</th><th>Response</th></tr>")
        html_lines.append(f"<tr><td>a</td><td>{row['response_a']}</td></tr>")
        html_lines.append(f"<tr><td>b</td><td>{row['response_b']}</td></tr>")
        html_lines.append(f"<tr><td>c</td><td>{row['response_c']}</td></tr>")
        html_lines.append(f"<tr><td>d</td><td>{row['response_d']}</td></tr>")
        html_lines.append("</table>")
        true_img = _build_vector_chart(
            true_source,
            true_aug,
            "True preference vectors (Source vs Augmentation)",
        )
        html_lines.append("<p><b>True preference vectors (source vs augmentation):</b></p>")
        html_lines.append(f"<img src='data:image/png;base64,{true_img}' alt='true vectors'/>")

        for alpha_value in report_alpha_values:
            if alpha_value == "baseline":
                key = "baseline"
            else:
                key = float(alpha_value)
            if key not in pred_count_by_alpha:
                continue
            denom = float(pred_count_by_alpha[key][idx])
            avg = np.zeros(4, dtype=float)
            if denom > 0:
                avg = pred_sum_by_alpha[key][idx] / denom
            pred_img = _build_predicted_chart(
                [float(x) for x in avg],
                f"Predicted mean response vector (alpha={key})",
            )
            html_lines.append(
                f"<p><b>Predicted mean response vector (alpha={key}):</b></p>"
            )
            html_lines.append(f"<img src='data:image/png;base64,{pred_img}' alt='predicted alpha {key}'/>")
        html_lines.append("</div>")

    html_lines.append("</body></html>")
    out_path.write_text("\n".join(html_lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()

    if args.source_group == args.augmentation_group:
        raise ValueError("augmentation_group must be different from source_group.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_csv is None:
        args.output_csv = str(output_dir / "community_alignment_polarizing_results.csv")
    if args.plot_path is None:
        args.plot_path = str(output_dir / "community_alignment_polarizing_scatter.png")
    if args.html_path is None:
        args.html_path = str(output_dir / "community_alignment_polarizing_report.html")
    if args.debug and args.debug_log_path is None:
        args.debug_log_path = str(output_dir / "community_alignment_polarizing_api_calls.jsonl")

    train_df = pd.read_csv(args.train_path)
    _require_columns(
        train_df,
        [args.protected_attribute, PROMPT_COLUMN, PREFERRED_RESPONSE_COLUMN]
        + RESPONSE_COLUMNS,
        "Train split",
    )
    required_test_cols = [
        args.protected_attribute,
        PROMPT_COLUMN,
        PREFERRED_RESPONSE_COLUMN,
    ] + RESPONSE_COLUMNS
    val_df = pd.read_csv(args.val_path) if Path(args.val_path).exists() else pd.DataFrame()
    test_df = pd.read_csv(args.test_path) if Path(args.test_path).exists() else pd.DataFrame()
    _require_columns(val_df, required_test_cols, "Validation split") if not val_df.empty else None
    _require_columns(test_df, required_test_cols, "Test split") if not test_df.empty else None

    test_pool = _resolve_test_pool(args)
    _require_columns(
        test_pool,
        required_test_cols,
        f"Resolved test pool ({args.test_set})",
    )

    source_train, aug_train = _prepare_train_rows(
        train_df,
        args.protected_attribute,
        args.source_group,
        args.augmentation_group,
    )
    if source_train.empty or aug_train.empty:
        raise ValueError("Source or augmentation group has no valid training rows.")

    polarizing_prompts_df = _select_polarizing_prompts(
        test_pool,
        args.protected_attribute,
        args.source_group,
        args.augmentation_group,
        args.test_prompts,
    )
    if polarizing_prompts_df.empty:
        raise ValueError("No polarizing prompts available for selection.")
    test_prompt_count = len(polarizing_prompts_df)

    if args.training_prompts == -1:
        training_budget = min(len(source_train), len(aug_train))
    else:
        training_budget = args.training_prompts
    if training_budget <= 0:
        raise ValueError("training_prompts resolves to 0 or less.")

    if args.alpha is not None:
        if not (0.0 <= args.alpha <= 1.0):
            raise ValueError("--alpha must be between 0.0 and 1.0.")
        alpha_values = [round(float(args.alpha), 10)]
    else:
        alpha_values = _build_alpha_values(args.alpha_step)

    if args.debug:
        debug_log_path = Path(args.debug_log_path)
        debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        debug_fp = debug_log_path.open("w", encoding="utf-8")
    else:
        debug_fp = None

    # allocate aggregation for predicted vectors across trials
    pred_sum_by_alpha: dict[float | str, np.ndarray] = {}
    pred_count_by_alpha: dict[float | str, np.ndarray] = {}
    if args.alpha is None:
        for alpha_value in alpha_values:
            pred_sum_by_alpha[float(alpha_value)] = np.zeros((test_prompt_count, 4), dtype=float)
            pred_count_by_alpha[float(alpha_value)] = np.zeros(test_prompt_count, dtype=int)
        pred_sum_by_alpha["baseline"] = np.zeros((test_prompt_count, 4), dtype=float)
        pred_count_by_alpha["baseline"] = np.zeros(test_prompt_count, dtype=int)
    else:
        pred_sum_by_alpha[float(alpha_values[0])] = np.zeros((test_prompt_count, 4), dtype=float)
        pred_count_by_alpha[float(alpha_values[0])] = np.zeros(test_prompt_count, dtype=int)

    total_input_tokens_ref = {"value": 0}
    total_completion_tokens_ref = {"value": 0}
    total_calls_ref = {"value": 0}
    all_records: list[dict[str, object]] = []

    if args.alpha is None:
        for trial in range(args.num_trials):
            rng = np.random.default_rng(args.random_seed + 10000 + trial)
            record, pred_one_hot = _run_single_trial(
                args=args,
                alpha_label="baseline",
                trial=trial,
                selected_source=source_train.iloc[0:0].copy(),
                selected_aug=aug_train.iloc[0:0].copy(),
                test_prompt_df=polarizing_prompts_df,
                rng=rng,
                training_budget=training_budget,
                output_rows=all_records,
                total_input_tokens_ref=total_input_tokens_ref,
                total_completion_tokens_ref=total_completion_tokens_ref,
                total_calls_ref=total_calls_ref,
                debug_fp=debug_fp,
            )
            if pred_one_hot is not None:
                pred_sum_by_alpha["baseline"] += pred_one_hot
                pred_count_by_alpha["baseline"] += (pred_one_hot.sum(axis=1) > 0).astype(int)
            print(
                f"Baseline trial={trial+1} accuracy={record['accuracy']} "
                f"success={record['success']} "
                f"valid_len={record['is_sequence_length_valid']}"
            )

    for alpha in alpha_values:
        for trial in range(args.num_trials):
            rng = np.random.default_rng(args.random_seed + trial + int(round(float(alpha) * 1000)))
            desired_source = int(training_budget * float(alpha))
            desired_aug = int(training_budget * (1.0 - float(alpha)))
            desired_total = desired_source + desired_aug
            if desired_total < training_budget:
                if len(source_train) >= desired_source + (training_budget - desired_total):
                    desired_source += training_budget - desired_total
                else:
                    desired_aug += training_budget - desired_total
            desired_source = min(desired_source, len(source_train))
            desired_aug = min(desired_aug, len(aug_train))
            selected_source = _sample_rows(source_train, desired_source, rng, "source group")
            selected_aug = _sample_rows(aug_train, desired_aug, rng, "augmentation group")

            record, pred_one_hot = _run_single_trial(
                args=args,
                alpha_label=alpha,
                trial=trial,
                selected_source=selected_source,
                selected_aug=selected_aug,
                test_prompt_df=polarizing_prompts_df,
                rng=rng,
                training_budget=training_budget,
                output_rows=all_records,
                total_input_tokens_ref=total_input_tokens_ref,
                total_completion_tokens_ref=total_completion_tokens_ref,
                total_calls_ref=total_calls_ref,
                debug_fp=debug_fp,
            )
            if pred_one_hot is not None:
                pred_sum_by_alpha[float(alpha)] += pred_one_hot
                pred_count_by_alpha[float(alpha)] += (pred_one_hot.sum(axis=1) > 0).astype(int)

    total_input_tokens = total_input_tokens_ref["value"]
    total_completion_tokens = total_completion_tokens_ref["value"]
    total_calls = total_calls_ref["value"]
    if args.debug and debug_fp is not None:
        debug_fp.close()

    out_df = pd.DataFrame(all_records)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    plot_path = Path(args.plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    valid_plot_df = out_df.dropna(subset=["accuracy"])
    sweep_plot_df = valid_plot_df[
        pd.Series(valid_plot_df["alpha"]).apply(lambda x: isinstance(x, (int, float)))
    ]
    plt.figure(figsize=(8, 5))
    if not sweep_plot_df.empty:
        sweep_plot_df = sweep_plot_df.copy()
        sweep_plot_df["alpha_numeric"] = sweep_plot_df["alpha"].astype(float)
        plt.scatter(sweep_plot_df["alpha"], sweep_plot_df["accuracy"], alpha=0.8)
        if args.alpha is None:
            sweep_mean_df = (
                sweep_plot_df.groupby("alpha_numeric", as_index=False)["accuracy"].mean()
                .sort_values("alpha_numeric")
            )
            if not sweep_mean_df.empty:
                plt.plot(
                    sweep_mean_df["alpha_numeric"],
                    sweep_mean_df["accuracy"],
                    linewidth=2.0,
                    marker="o",
                    label="Mean accuracy",
                )
    if args.alpha is None and "baseline" in out_df["alpha"].astype(str).tolist():
        baseline_records = out_df[out_df["alpha"].astype(str) == "baseline"]
        baseline_records = baseline_records.copy()
        baseline_records = baseline_records.dropna(subset=["accuracy"])
        if not baseline_records.empty:
            baseline_mean = baseline_records["accuracy"].mean()
            plt.axhline(
                y=float(baseline_mean),
                color="red",
                linestyle="--",
                linewidth=1.5,
                label="No-example baseline (mean)",
            )
    plt.title("Polarization Experiment: Accuracy by alpha")
    plt.xlabel("alpha")
    plt.ylabel("accuracy")
    plt.grid(alpha=0.2, linestyle="--")
    if not valid_plot_df.empty:
        plt.ylim(0.0, 1.0)
    if args.alpha is None:
        plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    if not out_df.empty and "success" in out_df.columns:
        print("Success rate by alpha:")
        out_df_numeric = out_df.copy()
        out_df_numeric["alpha_numeric"] = pd.to_numeric(out_df_numeric["alpha"], errors="coerce")
        alpha_success_rates = (
            out_df_numeric.dropna(subset=["alpha_numeric"])
            .groupby("alpha_numeric")["success"]
            .mean()
            .sort_index()
        )
        for alpha_value, rate in alpha_success_rates.items():
            print(f"  alpha={float(alpha_value):.3f}: {float(rate):.3f}")
        if "baseline" in out_df["alpha"].astype(str).tolist():
            baseline_rate = (
                out_df[out_df["alpha"].astype(str) == "baseline"]["success"].mean()
            )
            if pd.notna(baseline_rate):
                print(f"  baseline: {float(baseline_rate):.3f}")

        print("Sequence-length validity rate by alpha:")
        alpha_length_rates = (
            out_df_numeric.dropna(subset=["alpha_numeric"])
            .groupby("alpha_numeric")["is_sequence_length_valid"]
            .mean()
            .sort_index()
        )
        for alpha_value, rate in alpha_length_rates.items():
            print(f"  alpha={float(alpha_value):.3f}: {float(rate):.3f}")
        if "baseline" in out_df["alpha"].astype(str).tolist():
            baseline_rate = (
                out_df[out_df["alpha"].astype(str) == "baseline"]["is_sequence_length_valid"].mean()
            )
            if pd.notna(baseline_rate):
                print(f"  baseline: {float(baseline_rate):.3f}")

    avg_input_tokens = total_input_tokens / total_calls if total_calls else 0.0
    avg_completion_tokens = total_completion_tokens / total_calls if total_calls else 0.0
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total completion tokens: {total_completion_tokens}")
    print(f"Avg input tokens per call: {avg_input_tokens:.2f}")
    print(f"Avg completion tokens per call: {avg_completion_tokens:.2f}")

    html_path = Path(args.html_path)
    alpha_for_html = list(alpha_values)
    if args.alpha is None and "baseline" in pred_sum_by_alpha:
        alpha_for_html.append("baseline")
    _build_html_report(
        out_path=html_path,
        args=args,
        test_prompt_df=polarizing_prompts_df,
        alpha_values=alpha_for_html,
        pred_sum_by_alpha=pred_sum_by_alpha,
        pred_count_by_alpha=pred_count_by_alpha,
        out_csv=output_csv,
    )
    print(f"Saved HTML report to: {html_path}")
    print(f"Saved trial data to: {output_csv}")
    print(f"Saved scatter plot to: {plot_path}")


if __name__ == "__main__":
    main()
