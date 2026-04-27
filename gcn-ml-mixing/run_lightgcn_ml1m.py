#!/usr/bin/env python3
"""Run LightGCN CV folds (optionally in parallel) and collect metrics in one parent writer."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
            "Run one LightGCN process per CV fold, optionally in parallel, "
            "and write aggregated metrics once from the parent process."
        )
    )
    parser.add_argument("--decay", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--dataset", type=str, default="ml-1m")
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
        "--num-folds",
        type=int,
        default=5,
        help="Number of CV folds to run as separate main.py calls.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of concurrent fold processes to launch.",
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
        default=Path("gcn-ml-mixing/ml1m_lgn_metrics.json"),
        help="Where to write aggregated per-fold and mean metrics as JSON.",
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


def run_fold(
    fold_idx: int,
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
    cmd = [
        "python",
        "main.py",
        f"--decay={decay}",
        f"--lr={lr}",
        f"--layer={layer}",
        f"--seed={seed + fold_idx}",
        f"--dataset={dataset}",
        f"--topks={topks}",
        f"--recdim={recdim}",
        f"--bpr_batch={bpr_batch}",
        "--eval_split=val",
        f"--val_split_idx={fold_idx}",
        "--tensorboard=0",
        "--epochs",
        str(epochs),
    ]

    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(code_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start

    if proc.returncode != 0:
        short_out = (proc.stdout or "")[-4000:]
        short_err = (proc.stderr or "")[-4000:]
        raise RuntimeError(
            "LightGCN fold run failed "
            f"(fold={fold_idx}, exit={proc.returncode}).\n"
            "--- stdout (tail) ---\n"
            f"{short_out}\n"
            "--- stderr (tail) ---\n"
            f"{short_err}"
        )

    metrics = parse_training_output(proc.stdout, topks=topks_list)
    gpu_mem = parse_gpu_memory_from_stdout(proc.stdout)

    return {
        "fold_index": fold_idx,
        "runtime_seconds": elapsed,
        "gpu_memory": gpu_mem,
        "metrics": metrics,
    }


def _mean_dict(dicts: list[dict[str, float]]) -> dict[str, float]:
    if not dicts:
        return {}
    keys = sorted(dicts[0].keys())
    n = float(len(dicts))
    return {k: float(sum(d[k] for d in dicts) / n) for k in keys}


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if args.num_folds <= 0:
        raise ValueError("--num-folds must be >= 1")
    if args.parallel_workers <= 0:
        raise ValueError("--parallel-workers must be >= 1")

    code_dir = args.lightgcn_code_dir
    if not code_dir.is_absolute():
        code_dir = (repo_root / code_dir).resolve()

    output_json = args.output_json
    if not output_json.is_absolute():
        output_json = (repo_root / output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    try:
        topks = [int(x.strip()) for x in args.topks.strip("[]").split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid --topks value: {args.topks}") from exc

    parallel_workers = min(args.parallel_workers, args.num_folds)

    print("Parent orchestrator configuration:")
    print(f"- code_dir: {code_dir}")
    print(f"- dataset: {args.dataset}")
    print(f"- folds: {args.num_folds}")
    print(f"- parallel_workers: {parallel_workers}")
    print(f"- epochs: {args.epochs}")
    print(f"- output_json: {output_json}")

    start_wall = time.perf_counter()
    fold_results: list[dict] = []

    with ThreadPoolExecutor(max_workers=parallel_workers) as ex:
        fut_to_fold = {
            ex.submit(
                run_fold,
                fold_idx,
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
            ): fold_idx
            for fold_idx in range(args.num_folds)
        }

        for fut in as_completed(fut_to_fold):
            fold_idx = fut_to_fold[fut]
            result = fut.result()
            fold_results.append(result)
            fold_runtime = result["runtime_seconds"]
            peak_alloc = result["gpu_memory"]["peak_allocated_gib"]
            if peak_alloc is None:
                print(f"Fold {fold_idx}: completed in {fold_runtime:.2f}s | peak GPU mem: n/a")
            else:
                print(
                    f"Fold {fold_idx}: completed in {fold_runtime:.2f}s | "
                    f"peak GPU mem: {peak_alloc:.2f} GiB"
                )

    total_wall_seconds = time.perf_counter() - start_wall
    fold_results.sort(key=lambda x: int(x["fold_index"]))

    sum_child_runtimes = float(sum(float(r["runtime_seconds"]) for r in fold_results))
    peak_alloc_values = [
        float(r["gpu_memory"]["peak_allocated_gib"])
        for r in fold_results
        if r["gpu_memory"]["peak_allocated_gib"] is not None
    ]
    peak_reserved_values = [
        float(r["gpu_memory"]["peak_reserved_gib"])
        for r in fold_results
        if r["gpu_memory"]["peak_reserved_gib"] is not None
    ]

    max_peak_alloc_gib = max(peak_alloc_values) if peak_alloc_values else None
    max_peak_reserved_gib = max(peak_reserved_values) if peak_reserved_values else None

    per_fold_losses = [float(r["metrics"]["final_loss"]) for r in fold_results]
    per_fold_precision = [r["metrics"]["final_precision"] for r in fold_results]
    per_fold_recall = [r["metrics"]["final_recall"] for r in fold_results]
    per_fold_ndcg = [r["metrics"]["final_ndcg"] for r in fold_results]

    summary = {
        "num_folds": args.num_folds,
        "parallel_workers": parallel_workers,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "seed": args.seed,
        "topks": topks,
        "total_runtime_seconds": total_wall_seconds,
        "sum_fold_runtime_seconds": sum_child_runtimes,
        "max_gpu_peak_allocated_gib": max_peak_alloc_gib,
        "max_gpu_peak_reserved_gib": max_peak_reserved_gib,
        "mean_final_loss": float(sum(per_fold_losses) / len(per_fold_losses)),
        "mean_precision": _mean_dict(per_fold_precision),
        "mean_recall": _mean_dict(per_fold_recall),
        "mean_ndcg": _mean_dict(per_fold_ndcg),
    }

    output = {
        "summary": summary,
        "folds": fold_results,
    }
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\nAggregate results:")
    print(f"- Total runtime (wall): {total_wall_seconds:.2f} s")
    print(f"- Max GPU memory (peak allocated): {max_peak_alloc_gib if max_peak_alloc_gib is not None else 'n/a'} GiB")
    print(f"- Saved metrics JSON: {output_json}")


if __name__ == "__main__":
    main()
