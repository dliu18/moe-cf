#!/usr/bin/env python3
"""Run a single LightGCN training job on the full training split and save metrics."""

from __future__ import annotations

import argparse
import json
import pickle
import re
import subprocess
import time
from pathlib import Path


LOSS_RE = re.compile(r"EPOCH\[(\d+)/(\d+)\]\s+loss([0-9]*\.?[0-9]+)-")
ARRAY_RE = r"array\(\[([^\]]+)\]\)"
METRICS_RE = re.compile(
    rf"\{{'precision':\s*{ARRAY_RE},\s*'recall':\s*{ARRAY_RE},\s*'ndcg':\s*{ARRAY_RE}\}}"
)
GPU_PEAK_ALLOC_RE = re.compile(r"peak_allocated=([0-9]*\.?[0-9]+)\s+GiB")
GPU_PEAK_RESERVED_RE = re.compile(r"peak_reserved=([0-9]*\.?[0-9]+)\s+GiB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one LightGCN training process using eval_split=test "
            "(so train_full.txt is used when present)."
        )
    )
    parser.add_argument("--decay", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument(
        "--dataset",
        type=str,
        default="ml-1m",
        help="Dataset name passed to LightGCN (e.g. ml-1m, lastfm-asia).",
    )
    parser.add_argument("--topks", type=str, default="[20]")
    parser.add_argument("--recdim", type=int, default=64)
    parser.add_argument(
        "--bpr_batch",
        type=int,
        default=1_000_000,
        help="BPR training mini-batch size passed to LightGCN as --bpr_batch.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--group-labels-pkl",
        type=Path,
        default=None,
        help=(
            "Optional labels pickle for group-disaggregated eval. "
            "Default: LightGCN/data/<dataset>/user_labels.pkl"
        ),
    )
    parser.add_argument(
        "--feature-name",
        type=str,
        default="",
        help=(
            "Feature key for disaggregated eval (e.g. Age, Gender, Country). "
            "If empty, only overall metrics are reported."
        ),
    )
    parser.add_argument(
        "--lightgcn-code-dir",
        type=Path,
        default=Path("LightGCN/code"),
        help="Directory containing main.py",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Where to write metrics JSON. "
            "Default: mixing/<dataset>_lgn_metrics.json"
        ),
    )
    return parser.parse_args()


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_training_output(stdout_text: str, topks: list[int]) -> dict:
    loss_matches = list(LOSS_RE.finditer(stdout_text))
    if not loss_matches:
        raise RuntimeError("Could not parse training loss from LightGCN output.")

    last_loss = loss_matches[-1]
    final_epoch = int(last_loss.group(1))
    total_epochs = int(last_loss.group(2))
    final_loss = float(last_loss.group(3))

    metric_matches = list(METRICS_RE.finditer(stdout_text))
    if not metric_matches:
        raise RuntimeError(
            "Could not parse metrics from LightGCN output. "
            "Ensure training emitted at least one [TEST] block."
        )

    last_metrics = metric_matches[-1]
    precision_vals = _parse_float_list(last_metrics.group(1))
    recall_vals = _parse_float_list(last_metrics.group(2))
    ndcg_vals = _parse_float_list(last_metrics.group(3))

    if not (len(precision_vals) == len(recall_vals) == len(ndcg_vals) == len(topks)):
        raise RuntimeError(
            "Parsed metric lengths do not match number of k values in --topks. "
            f"topks={topks}, lens={(len(precision_vals), len(recall_vals), len(ndcg_vals))}"
        )

    precision_at_k = {f"precision@{k}": v for k, v in zip(topks, precision_vals)}
    recall_at_k = {f"recall@{k}": v for k, v in zip(topks, recall_vals)}
    ndcg_at_k = {f"ndcg@{k}": v for k, v in zip(topks, ndcg_vals)}

    return {
        "final_epoch": final_epoch,
        "total_epochs": total_epochs,
        "final_loss": final_loss,
        "final_precision": precision_at_k,
        "final_recall": recall_at_k,
        "final_ndcg": ndcg_at_k,
    }


def parse_gpu_memory_from_stdout(stdout_text: str) -> dict[str, float | None]:
    alloc_matches = GPU_PEAK_ALLOC_RE.findall(stdout_text)
    reserved_matches = GPU_PEAK_RESERVED_RE.findall(stdout_text)
    peak_alloc_gib = float(alloc_matches[-1]) if alloc_matches else None
    peak_reserved_gib = float(reserved_matches[-1]) if reserved_matches else None
    return {
        "peak_allocated_gib": peak_alloc_gib,
        "peak_reserved_gib": peak_reserved_gib,
    }


def parse_metrics_only_output(stdout_text: str, topks: list[int]) -> dict:
    metric_matches = list(METRICS_RE.finditer(stdout_text))
    if not metric_matches:
        raise RuntimeError("Could not parse ranking metrics from LightGCN output.")
    m = metric_matches[-1]
    precision_vals = _parse_float_list(m.group(1))
    recall_vals = _parse_float_list(m.group(2))
    ndcg_vals = _parse_float_list(m.group(3))
    if not (len(precision_vals) == len(recall_vals) == len(ndcg_vals) == len(topks)):
        raise RuntimeError(
            "Parsed metric lengths do not match number of k values in --topks. "
            f"topks={topks}, lens={(len(precision_vals), len(recall_vals), len(ndcg_vals))}"
        )
    return {
        "final_precision": {f"precision@{k}": v for k, v in zip(topks, precision_vals)},
        "final_recall": {f"recall@{k}": v for k, v in zip(topks, recall_vals)},
        "final_ndcg": {f"ndcg@{k}": v for k, v in zip(topks, ndcg_vals)},
    }


def _run_cmd(code_dir: Path, cmd: list[str]) -> tuple[subprocess.CompletedProcess[str], float]:
    print("Running command:")
    print(" ".join(cmd))
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(code_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    return proc, elapsed


def _build_main_cmd(
    *,
    dataset: str,
    lr: float,
    decay: float,
    layer: int,
    seed: int,
    topks: str,
    recdim: int,
    bpr_batch: int,
    epochs: int,
    eval_split: str = "test",
    load: int = 0,
    group_mixing: int = 0,
    group_labels_pkl: Path | None = None,
    feature_name: str = "",
    source_group: str = "",
    alpha_aug: float = 0.0,
    alpha_mix: str = "",
) -> list[str]:
    cmd = [
        "python",
        "main.py",
        f"--decay={decay}",
        f"--lr={lr}",
        f"--layer={layer}",
        f"--seed={seed}",
        f"--dataset={dataset}",
        f"--topks={topks}",
        f"--recdim={recdim}",
        f"--bpr_batch={bpr_batch}",
        f"--eval_split={eval_split}",
        "--tensorboard=0",
        f"--load={load}",
        "--epochs",
        str(epochs),
    ]
    if group_mixing:
        if group_labels_pkl is None:
            raise ValueError("group_labels_pkl must be set when group_mixing=1.")
        cmd.extend(
            [
                "--group_mixing=1",
                f"--group_labels_pkl={group_labels_pkl}",
                f"--feature_name={feature_name}",
                f"--source_group={source_group}",
                f"--alpha_aug={alpha_aug}",
                f"--alpha_mix={alpha_mix}",
            ]
        )
    return cmd


def run_single_model(
    *,
    code_dir: Path,
    dataset: str,
    lr: float,
    decay: float,
    layer: int,
    seed: int,
    topks: str,
    recdim: int,
    bpr_batch: int,
    epochs: int,
    topks_list: list[int],
) -> dict:
    cmd = _build_main_cmd(
        dataset=dataset,
        lr=lr,
        decay=decay,
        layer=layer,
        seed=seed,
        topks=topks,
        recdim=recdim,
        bpr_batch=bpr_batch,
        epochs=epochs,
        eval_split="test",
        load=0,
    )
    proc, elapsed = _run_cmd(code_dir, cmd)

    if proc.returncode != 0:
        short_out = (proc.stdout or "")[-4000:]
        short_err = (proc.stderr or "")[-4000:]
        raise RuntimeError(
            "LightGCN run failed "
            f"(exit={proc.returncode}).\n"
            "--- stdout (tail) ---\n"
            f"{short_out}\n"
            "--- stderr (tail) ---\n"
            f"{short_err}"
        )

    metrics = parse_training_output(proc.stdout, topks=topks_list)
    gpu_mem = parse_gpu_memory_from_stdout(proc.stdout)

    return {
        "runtime_seconds": elapsed,
        "gpu_memory": gpu_mem,
        "metrics": metrics,
    }


def run_group_disaggregated_eval(
    *,
    code_dir: Path,
    dataset: str,
    lr: float,
    decay: float,
    layer: int,
    seed: int,
    topks: str,
    recdim: int,
    bpr_batch: int,
    topks_list: list[int],
    labels_path: Path,
    feature_name: str,
) -> dict:
    with labels_path.open("rb") as f:
        labels_all = pickle.load(f)
    if feature_name not in labels_all:
        raise KeyError(
            f"Feature '{feature_name}' not found in labels pickle {labels_path}. "
            f"Available: {list(labels_all.keys())}"
        )
    label_to_users_raw = labels_all[feature_name]
    label_to_users = {str(k): [int(u) for u in v] for k, v in label_to_users_raw.items()}
    groups = sorted(label_to_users.keys(), key=lambda x: str(x))
    if len(groups) < 2:
        raise ValueError(f"Need at least 2 groups for disaggregated eval; got {groups}")

    output: dict[str, dict] = {}
    for i, source_group in enumerate(groups):
        aug_groups = [g for g in groups if g != source_group]
        uniform_mix = {g: 1.0 / len(aug_groups) for g in aug_groups}
        cmd = _build_main_cmd(
            dataset=dataset,
            lr=lr,
            decay=decay,
            layer=layer,
            seed=seed + 10_000 + i,
            topks=topks,
            recdim=recdim,
            bpr_batch=bpr_batch,
            epochs=0,
            eval_split="test",
            load=1,
            group_mixing=1,
            group_labels_pkl=labels_path,
            feature_name=feature_name,
            source_group=source_group,
            alpha_aug=0.0,
            alpha_mix=json.dumps(uniform_mix, sort_keys=True),
        )
        proc, elapsed = _run_cmd(code_dir, cmd)
        if proc.returncode != 0:
            short_out = (proc.stdout or "")[-4000:]
            short_err = (proc.stderr or "")[-4000:]
            raise RuntimeError(
                f"Group eval run failed (group={source_group}, exit={proc.returncode}).\n"
                "--- stdout (tail) ---\n"
                f"{short_out}\n"
                "--- stderr (tail) ---\n"
                f"{short_err}"
            )
        metrics = parse_metrics_only_output(proc.stdout, topks_list)
        output[source_group] = {
            "num_users": int(len(label_to_users[source_group])),
            "runtime_seconds": float(elapsed),
            "final_precision": metrics["final_precision"],
            "final_recall": metrics["final_recall"],
            "final_ndcg": metrics["final_ndcg"],
        }
    return output


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    code_dir = args.lightgcn_code_dir
    if not code_dir.is_absolute():
        code_dir = (repo_root / code_dir).resolve()

    output_json = args.output_json
    if output_json is None:
        dataset_slug = args.dataset.replace("/", "_")
        output_json = Path(f"mixing/{dataset_slug}_lgn_metrics.json")
    if not output_json.is_absolute():
        output_json = (repo_root / output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    labels_path = args.group_labels_pkl
    if labels_path is None:
        labels_path = Path("LightGCN") / "data" / args.dataset / "user_labels.pkl"
    if not labels_path.is_absolute():
        labels_path = (repo_root / labels_path).resolve()

    try:
        topks = [int(x.strip()) for x in args.topks.strip("[]").split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid --topks value: {args.topks}") from exc

    print("LightGCN single-run configuration:")
    print(f"- code_dir: {code_dir}")
    print(f"- dataset: {args.dataset}")
    print("- eval_split: test (uses train_full.txt when available)")
    print(f"- epochs: {args.epochs}")
    print(f"- feature_name (disaggregated eval): {args.feature_name or 'disabled'}")
    print(f"- output_json: {output_json}")

    result = run_single_model(
        code_dir=code_dir,
        dataset=args.dataset,
        lr=args.lr,
        decay=args.decay,
        layer=args.layer,
        seed=args.seed,
        topks=args.topks,
        recdim=args.recdim,
        bpr_batch=args.bpr_batch,
        epochs=args.epochs,
        topks_list=topks,
    )

    summary = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "seed": args.seed,
        "topks": topks,
        "runtime_seconds": float(result["runtime_seconds"]),
        "gpu_memory": result["gpu_memory"],
        "final_loss": float(result["metrics"]["final_loss"]),
        "final_precision": result["metrics"]["final_precision"],
        "final_recall": result["metrics"]["final_recall"],
        "final_ndcg": result["metrics"]["final_ndcg"],
    }

    output = {
        "summary": summary,
        "run": result,
    }
    if args.feature_name:
        if not labels_path.is_file():
            raise FileNotFoundError(
                f"Requested disaggregated metrics, but labels pickle was not found: {labels_path}"
            )
        group_metrics = run_group_disaggregated_eval(
            code_dir=code_dir,
            dataset=args.dataset,
            lr=args.lr,
            decay=args.decay,
            layer=args.layer,
            seed=args.seed,
            topks=args.topks,
            recdim=args.recdim,
            bpr_batch=args.bpr_batch,
            topks_list=topks,
            labels_path=labels_path,
            feature_name=args.feature_name,
        )
        output["group_metrics"] = {
            "feature_name": args.feature_name,
            "labels_path": str(labels_path),
            "groups": group_metrics,
        }
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")

    peak_alloc = result["gpu_memory"]["peak_allocated_gib"]
    print("\nRun complete:")
    print(f"- Runtime: {result['runtime_seconds']:.2f} s")
    print(f"- Peak GPU memory (allocated): {peak_alloc if peak_alloc is not None else 'n/a'} GiB")
    print(f"- Saved metrics JSON: {output_json}")


if __name__ == "__main__":
    main()
