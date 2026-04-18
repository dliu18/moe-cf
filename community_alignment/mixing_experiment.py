"""Run alpha-sweep data-mixing experiments for Community Alignment first-turn prompts."""

from __future__ import annotations

import argparse
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
        description="Data mixing experiment scaffold for Community Alignment"
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
        help="Number of training prompts to sample into the example set; -1 => max.",
    )
    parser.add_argument(
        "--test_prompts",
        type=int,
        default=-1,
        help="Number of test prompts to sample; -1 => all test prompts.",
    )
    parser.add_argument(
        "--test_set",
        choices=["val", "test", "both"],
        default="test",
        help="Which split to use as the test pool.",
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
        help=(
            "If provided, run only this single alpha (0.0 <= alpha <= 1.0) "
            "and skip alpha sweep."
        ),
    )
    parser.add_argument(
        "--alpha_step",
        type=float,
        default=0.05,
        help=(
            "Step size for alpha sweep (inclusive range [0.0, 1.0]). "
            "Example: 0.1 => [0.0, 0.1, ..., 1.0]."
        ),
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
        help=(
            "If set, do not call the OpenAI API and simulate model predictions. "
            "This enables full end-to-end outputs (accuracy, csv, plot) for sanity checks."
        ),
    )
    parser.add_argument(
        "--simulate_prediction_mode",
        choices=["all_a", "random", "ground_truth"],
        default="all_a",
        help=(
            "Prediction mode used when --simulate is enabled. "
            "'all_a' returns all 'a'; 'random' samples uniformly; "
            "'ground_truth' returns the true label (upper-bound sanity check)."
        ),
    )
    parser.add_argument(
        "--model_name",
        default="gpt-5-mini",
        help="OpenAI model name for API calls.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Enable detailed API-call logs (system + user prompts and metadata) to a log file."
        ),
    )
    parser.add_argument(
        "--log_call_metrics",
        action="store_true",
        help="Print per-call response/accuracy/token metrics to stdout.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/community_alignment_mixing",
        help="Directory where experiment outputs are written.",
    )
    parser.add_argument(
        "--debug_log_path",
        default=None,
        help="Path for API call debug logs. Defaults into --output_dir.",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Output CSV containing the per-alpha/trial data. Defaults into --output_dir.",
    )
    parser.add_argument(
        "--plot_path",
        default=None,
        help="Output path for alpha-vs-accuracy scatter plot. Defaults into --output_dir.",
    )
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(sorted(missing))}")


def _safe_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    return text.strip()

def _canonical_preferred(response_value: object) -> str:
    value = _safe_text(response_value).lower()
    if value.startswith("response_") and len(value) == 10 and value[-1] in RESPONSE_TEXT_COLUMNS:
        return value[-1]
    if value in RESPONSE_TEXT_COLUMNS:
        return value
    return ""

def _preferred_response_text(row: pd.Series) -> str:
    letter = _canonical_preferred(row[PREFERRED_RESPONSE_COLUMN])
    if not letter:
        return ""
    return _safe_text(row[RESPONSE_TEXT_COLUMNS[letter]])


def _count_tokens_local(text: str) -> int:
    if text is None:
        return 0
    return len(re.findall(r"\S+", str(text)))


def _estimate_completion_tokens(expected_test_rows: int) -> int:
    # Expected response is a comma-separated list of letters.
    # Conservative local estimate: one token per letter and one for separator.
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


def _build_alpha_values(training_budget: int, alpha_step: float) -> list[float]:
    if not (0 < alpha_step <= 1):
        raise ValueError("alpha_step must be in (0, 1].")
    if alpha_step == 1.0:
        return [0.0, 1.0]

    alpha_values = []
    current = 0.0
    safety_iter = 0
    max_iter = 10000
    while current < 1.0 - 1e-12 and safety_iter < max_iter:
        alpha_values.append(round(current, 10))
        current += alpha_step
        safety_iter += 1
    alpha_values.append(1.0)
    # Ensure bounds and uniqueness.
    return sorted({min(1.0, max(0.0, a)) for a in alpha_values})


def _build_example_block(examples: pd.DataFrame) -> str:
    lines = [
        "You are a personal assistant whose goal is to personalize your responses "
        "to the preferences of a user, based on a history of their preferred choices.",
        "### Start Examples",
        "",
    ]
    for idx, row in examples.reset_index(drop=True).iterrows():
        preferred = _preferred_response_text(row)
        lines.append(f"Prompt: {row[PROMPT_COLUMN]}")
        lines.append(f"Preferred Response: {preferred}")
        lines.append("")
    lines.append("### End Examples")
    return "\n".join(lines)


def _build_test_block(test_rows: pd.DataFrame) -> str:
    lines = ["### Start Test Prompts", ""]
    for idx, row in test_rows.reset_index(drop=True).iterrows():
        lines.append(f"Prompt: {row[PROMPT_COLUMN]}")
        for letter, col in RESPONSE_TEXT_COLUMNS.items():
            lines.append(f"Response {letter}: {_safe_text(row[col])}")
        lines.append("")
    lines.append("### End Test Prompts")
    return "\n".join(lines)


def _build_call_payload(
    examples: pd.DataFrame,
    test_rows: pd.DataFrame,
    test_prompt_count: int,
    include_preference_context: bool = True,
) -> tuple[list[dict[str, str]], str]:
    example_block = _build_example_block(examples)
    test_block = _build_test_block(test_rows)
    if include_preference_context:
        instruction_prefix = "Now, taking into account these above preferred choices, "
    else:
        instruction_prefix = "Now, "
    instruction_block = (
        f"{instruction_prefix}select the most "
        f"appropriate potential response from each of the following {test_prompt_count} prompts. "
        f"Answer with just a sequence of chosen responses e.g. a,c,d,b,a,c,... "
        f"The response should be {test_prompt_count} letters separated by commas."
    )
    user_prompt = (
        f"{example_block}\n\n"
        f"{instruction_block}\n\n"
        f"{test_block}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return messages, user_prompt


def _build_openai_request(
    messages: list[dict[str, str]],
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 25000,
) -> dict[str, object]:
    request = {
        "model": model_name,
        "messages": messages,
    }
    if model_name.startswith("gpt-5"):
        # GPT-5 models currently require default temperature behavior only.
        request["max_completion_tokens"] = max_tokens
    else:
        request["temperature"] = temperature
        request["max_tokens"] = max_tokens
    return request


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
        if n_expected <= len(expected) and isinstance(expected[0], (list, tuple, np.ndarray)):
            return [str(x[0]) if len(x) > 0 else "" for x in expected[:n_expected]]
        return expected.tolist()
    return [str(chr(ord("a") + int(rng.integers(0, 4)))) for _ in range(n_expected)]


def _write_debug_entry(
    debug_fp,
    payload: dict[str, object],
    meta: dict[str, object] | None = None,
) -> None:
    entry = {
        "api_call": payload,
    }
    if meta:
        entry["meta"] = meta
    debug_fp.write(json.dumps(entry, ensure_ascii=False, indent=2) + "\n\n")


def _extract_predictions(response: str) -> list[str]:
    if not response:
        return []
    tokens = re.findall(r"[a-dA-D]", response)
    return [tok.lower() for tok in tokens if tok.lower() in RESPONSE_TEXT_COLUMNS]


def _parse_predictions(response: str, n_expected: int) -> list[str]:
    return _extract_predictions(response)[:n_expected]


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


def _prepare_test_rows(
    test_pool: pd.DataFrame,
    protected_attribute: str,
    source_group: str,
    sample_n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = test_pool[test_pool[protected_attribute] == source_group].copy()
    rows = rows[rows[PROMPT_COLUMN].notna()]
    rows = rows.copy()
    # Keep rows where the preferred response is an explicit first-turn choice.
    rows = rows[rows[PREFERRED_RESPONSE_COLUMN].map(_canonical_preferred).astype(bool)]
    if rows.empty:
        return rows.iloc[0:0]

    prompt_groups = rows.groupby(PROMPT_COLUMN, sort=False)
    valid_prompts = list(prompt_groups.groups.keys())
    if not valid_prompts:
        return rows.iloc[0:0]

    if sample_n == -1:
        selected_prompts = valid_prompts
    elif sample_n <= 0:
        return rows.iloc[0:0]
    elif sample_n > len(valid_prompts):
        raise ValueError(
            f"Requested test_prompts={sample_n} but only {len(valid_prompts)} source-group test prompts are available."
        )
    else:
        selected_prompts = pd.Index(valid_prompts).to_series().sample(
            n=sample_n,
            random_state=int(rng.integers(low=0, high=2**31 - 1)),
            replace=False,
        ).tolist()

    sampled_prompts = []
    for prompt in selected_prompts:
        prompt_rows = prompt_groups.get_group(prompt)
        prompt_rows = prompt_rows.copy()
        prompt_rows = prompt_rows[prompt_rows[PREFERRED_RESPONSE_COLUMN].map(_canonical_preferred).astype(bool)]
        label_list = [
            _canonical_preferred(v)
            for v in prompt_rows[PREFERRED_RESPONSE_COLUMN].to_numpy()
            if _canonical_preferred(v)
        ]
        if not label_list:
            continue
        row_sample = prompt_rows.iloc[0]
        sampled_prompt = row_sample.to_dict()
        sampled_prompt["source_preferred_labels"] = label_list
        sampled_prompt[PREFERRED_RESPONSE_COLUMN] = label_list[0]
        sampled_prompts.append(sampled_prompt)

    if not sampled_prompts:
        return rows.iloc[0:0]

    return pd.DataFrame(sampled_prompts).reset_index(drop=True)


def _run_single_trial(
    *,
    args: argparse.Namespace,
    alpha_label: float | str,
    trial: int,
    selected_source: pd.DataFrame,
    selected_aug: pd.DataFrame,
    test_pool: pd.DataFrame,
    rng: np.random.Generator,
    training_budget: int,
    output_rows: list[dict[str, object]],
    total_input_tokens_ref: dict[str, int],
    total_completion_tokens_ref: dict[str, int],
    total_calls_ref: dict[str, int],
    debug_fp,
) -> dict[str, object]:
    examples = pd.concat([selected_source, selected_aug], axis=0, ignore_index=True)

    test_rows = _prepare_test_rows(
        test_pool,
        args.protected_attribute,
        args.source_group,
        args.test_prompts,
        rng,
    )
    if test_rows.empty:
        raise ValueError(f"No source-group rows in chosen test set for trial {trial} and alpha={alpha_label}.")

    n_test_prompts = len(test_rows)
    is_baseline_call = alpha_label == "baseline" or len(examples) == 0
    messages, _ = _build_call_payload(
        examples,
        test_rows,
        n_test_prompts,
        include_preference_context=not is_baseline_call,
    )
    api_call_payload = _build_openai_request(
        messages=messages,
        model_name=args.model_name,
        temperature=0.0,
        max_tokens=25000,
    )
    estimated_input_tokens = (
        _count_tokens_local(messages[0]["content"])
        + _count_tokens_local(messages[1]["content"])
    )

    expected = np.array(test_rows["source_preferred_labels"].to_numpy(), dtype=object)
    api_response = ""
    predictions: list[str] = []
    actual_input_tokens = 0
    completion_tokens = 0
    api_finish_reason: str | None = None
    api_refusal: str | None = None
    api_response_id: str | None = None
    usage_payload: dict[str, object] | None = None

    if args.simulate:
        predictions = _simulate_predictions(len(test_rows), args.simulate_prediction_mode, expected, rng)
        completion_tokens = _estimate_completion_tokens(len(test_rows))
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
        predicted_letters = _extract_predictions(api_response)
        predictions = predicted_letters[:len(test_rows)]
        call_tokens = prompt_tokens + completion_tokens

    total_calls_ref["value"] += 1
    total_input_tokens_ref["value"] += actual_input_tokens
    total_completion_tokens_ref["value"] += completion_tokens

    print(
        f"alpha={alpha_label} trial={trial+1} "
        f"train_ex={len(examples):,}, test_rows={len(test_rows):,}, "
        f"tokens={call_tokens:,}"
    )

    is_success = api_finish_reason == "stop"
    expected_labels = np.array(test_rows["source_preferred_labels"].to_numpy(), dtype=object)
    expected_len = len(expected_labels)
    flat_expected_count = int(sum(len(labels) for labels in expected_labels))
    if expected_len == 0:
        num_correct = 0
        accuracy = float("nan")
    elif is_success:
        flat_expected: list[str] = []
        flat_pred: list[str] = []
        for i, labels in enumerate(expected_labels):
            prompt_pred = predictions[i] if i < len(predictions) else ""
            flat_expected.extend(labels)
            flat_pred.extend([prompt_pred] * len(labels))
        flat_expected_arr = np.array(flat_expected, dtype=str)
        flat_pred_arr = np.array(flat_pred, dtype=str)
        n_expected_annotations = len(flat_expected_arr)
        if n_expected_annotations == 0:
            num_correct = 0
            accuracy = float("nan")
        else:
            num_correct = int(np.sum(flat_pred_arr == flat_expected_arr))
            accuracy = num_correct / n_expected_annotations
    else:
        num_correct = None
        accuracy = float("nan")

    if args.simulate:
        predicted_len = len(predictions)
    else:
        predicted_len = len(predicted_letters)
    is_sequence_length_valid = predicted_len == expected_len

    record = {
        "alpha": alpha_label,
        "trial": trial + 1,
        "training_budget": training_budget,
        "n_examples_source": len(selected_source),
        "n_examples_augmentation": len(selected_aug),
        "n_test_prompts": len(test_rows),
        "n_test_annotations": int(flat_expected_count),
        "num_correct": num_correct,
        "accuracy": accuracy,
        "estimated_call_tokens": call_tokens,
        "estimated_input_tokens": estimated_input_tokens,
        "actual_input_tokens": actual_input_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": api_finish_reason,
        "success": bool(is_success),
        "is_sequence_length_valid": bool(is_sequence_length_valid),
        "predicted_response_count": int(predicted_len),
        "expected_response_count": int(expected_len),
        "expected_annotation_count": int(flat_expected_count),
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
            "n_test_prompts": len(test_rows),
            "n_test_annotations": int(flat_expected_count),
            "estimated_input_tokens": estimated_input_tokens,
            "actual_input_tokens": actual_input_tokens,
            "simulate": bool(args.simulate),
            "simulate_prediction_mode": args.simulate_prediction_mode,
            "api_mode": "simulated" if args.simulate else "live_api",
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
        meta["sequence_length_valid"] = bool(is_sequence_length_valid)
        meta["predicted_response_count"] = int(predicted_len)
        meta["expected_response_count"] = int(expected_len)
        meta["expected_annotation_count"] = int(flat_expected_count)
        if args.simulate:
            meta["simulated_predictions"] = predictions
        _write_debug_entry(debug_fp, api_call_payload, meta)

    if args.log_call_metrics and not args.simulate:
        metric_accuracy = "n/a (non-stop)" if not is_success else f"{accuracy:.4f}"
        print(
            "Call metrics:"
            f"\n  response: {api_response!r}"
            f"\n  accuracy: {metric_accuracy}"
            f"\n  estimated_input_tokens: {estimated_input_tokens}"
            f"\n  actual_input_tokens: {actual_input_tokens}"
            f"\n  completion_tokens: {completion_tokens}"
            f"\n  usage: {json.dumps(usage_payload, ensure_ascii=False) if usage_payload else None}"
            f"\n  predicted_response_count: {predicted_len}"
            f"\n  expected_response_count: {expected_len}"
            f"\n  sequence_length_valid: {is_sequence_length_valid}"
            f"\n  finish_reason: {api_finish_reason}"
            f"\n  response_id: {api_response_id}"
            f"\n  refusal: {api_refusal}"
        )

    return record


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


def main() -> None:
    args = _parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_csv is None:
        args.output_csv = str(output_dir / "community_alignment_mixing_results.csv")
    if args.plot_path is None:
        args.plot_path = str(output_dir / "community_alignment_mixing_scatter.png")
    if args.debug and args.debug_log_path is None:
        args.debug_log_path = str(output_dir / "community_alignment_api_calls.jsonl")

    if args.source_group == args.augmentation_group:
        raise ValueError("augmentation_group must be different from source_group.")

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

    if args.training_prompts == -1:
        training_budget = min(len(source_train), len(aug_train))
    else:
        training_budget = args.training_prompts
    if training_budget <= 0:
        raise ValueError("training_prompts resolves to 0 or less; check source/augmentation group sizes.")

    if args.alpha is not None:
        if not (0.0 <= args.alpha <= 1.0):
            raise ValueError("--alpha must be between 0.0 and 1.0.")
        alpha_values = [round(float(args.alpha), 10)]
    else:
        alpha_values = _build_alpha_values(training_budget, args.alpha_step)

    if args.debug:
        debug_log_path = Path(args.debug_log_path)
        debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        debug_fp = debug_log_path.open("w", encoding="utf-8")
    else:
        debug_fp = None

    all_records: list[dict[str, object]] = []
    baseline_records: list[dict[str, object]] = []
    total_input_tokens_ref = {"value": 0}
    total_completion_tokens_ref = {"value": 0}
    total_calls_ref = {"value": 0}

    if args.alpha is None:
        for trial in range(args.num_trials):
            trial_seed = args.random_seed + 10000 + trial
            rng = np.random.default_rng(trial_seed)
            baseline_record = _run_single_trial(
                args=args,
                alpha_label="baseline",
                trial=trial,
                selected_source=source_train.iloc[0:0].copy(),
                selected_aug=aug_train.iloc[0:0].copy(),
                test_pool=test_pool,
                rng=rng,
                training_budget=training_budget,
                output_rows=baseline_records,
                total_input_tokens_ref=total_input_tokens_ref,
                total_completion_tokens_ref=total_completion_tokens_ref,
                total_calls_ref=total_calls_ref,
                debug_fp=debug_fp,
            )
            all_records.append(baseline_record)
            print(
                f"Baseline trial={trial+1} accuracy={baseline_record['accuracy']}"
            )

    for alpha in alpha_values:
        for trial in range(args.num_trials):
            trial_seed = args.random_seed + trial + int(round(alpha * 1000))
            rng = np.random.default_rng(trial_seed)

            desired_source = int(training_budget * alpha)
            desired_aug = int(training_budget * (1.0 - alpha))
            desired_total = desired_source + desired_aug
            if desired_total < training_budget:
                # compensate rounding loss deterministically without changing intent too much
                if len(source_train) >= desired_source + (training_budget - desired_total):
                    desired_source += training_budget - desired_total
                else:
                    desired_aug += training_budget - desired_total

            desired_source = min(desired_source, len(source_train))
            desired_aug = min(desired_aug, len(aug_train))
            selected_source = _sample_rows(source_train, desired_source, rng, "source group")
            selected_aug = _sample_rows(aug_train, desired_aug, rng, "augmentation group")
            _run_single_trial(
                args=args,
                alpha_label=alpha,
                trial=trial,
                selected_source=selected_source,
                selected_aug=selected_aug,
                test_pool=test_pool,
                rng=rng,
                training_budget=training_budget,
                output_rows=all_records,
                total_input_tokens_ref=total_input_tokens_ref,
                total_completion_tokens_ref=total_completion_tokens_ref,
                total_calls_ref=total_calls_ref,
                debug_fp=debug_fp,
            )

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
    if args.alpha is None and baseline_records:
        baseline_accuracies = [
            record["accuracy"]
            for record in baseline_records
            if isinstance(record.get("accuracy"), (int, float)) and not pd.isna(record.get("accuracy"))
        ]
        if baseline_accuracies:
            baseline_mean = float(np.nanmean(baseline_accuracies))
            plt.axhline(
                y=baseline_mean,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label="No-example baseline (mean)",
            )
    plt.title("Community Alignment Mixing Experiment (accuracy by alpha)")
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

    print(f"Saved trial data to: {output_csv}")
    print(f"Saved scatter plot to: {plot_path}")

    if not out_df.empty and "success" in out_df.columns:
        print("Success rate by alpha:")
        numeric_out_df = out_df.copy()
        numeric_out_df["alpha_numeric"] = pd.to_numeric(
            numeric_out_df["alpha"], errors="coerce"
        )
        alpha_success_rates = (
            numeric_out_df.dropna(subset=["alpha_numeric"])
            .groupby("alpha_numeric")["success"]
            .mean()
            .sort_index()
        )
        for alpha_value, success_rate in alpha_success_rates.items():
            print(f"  alpha={float(alpha_value):.3f}: {float(success_rate):.3f}")
        if "baseline" in out_df["alpha"].astype(str).tolist():
            baseline_mask = out_df["alpha"].astype(str) == "baseline"
            baseline_success = out_df.loc[baseline_mask, "success"].mean()
            if pd.notna(baseline_success):
                print(f"  baseline: {float(baseline_success):.3f}")

        print("Sequence-length validity rate by alpha:")
        alpha_length_rates = (
            numeric_out_df.dropna(subset=["alpha_numeric"])
            .groupby("alpha_numeric")["is_sequence_length_valid"]
            .mean()
            .sort_index()
        )
        for alpha_value, validity_rate in alpha_length_rates.items():
            print(f"  alpha={float(alpha_value):.3f}: {float(validity_rate):.3f}")
        if "baseline" in out_df["alpha"].astype(str).tolist():
            baseline_length = out_df.loc[
                out_df["alpha"].astype(str) == "baseline", "is_sequence_length_valid"
            ].mean()
            if pd.notna(baseline_length):
                print(f"  baseline: {float(baseline_length):.3f}")

    avg_input_tokens = total_input_tokens / total_calls if total_calls else 0.0
    avg_completion_tokens = total_completion_tokens / total_calls if total_calls else 0.0
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total completion tokens: {total_completion_tokens}")
    print(f"Avg input tokens per call: {avg_input_tokens:.2f}")
    print(f"Avg completion tokens per call: {avg_completion_tokens:.2f}")


if __name__ == "__main__":
    main()
