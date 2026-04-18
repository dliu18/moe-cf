#!/usr/bin/env python3
"""Run LightGCN on ml-1m and collect final loss/precision/recall/ndcg."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


LOSS_RE = re.compile(r"EPOCH\[(\d+)/(\d+)\]\s+loss([0-9]*\.?[0-9]+)-")
ARRAY_RE = r"array\(\[([^\]]+)\]\)"
METRICS_RE = re.compile(
    rf"\{{'precision':\s*{ARRAY_RE},\s*'recall':\s*{ARRAY_RE},\s*'ndcg':\s*{ARRAY_RE}\}}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LightGCN training on ml-1m and parse final metrics."
    )
    parser.add_argument("--decay", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--dataset", type=str, default="ml-1m")
    parser.add_argument("--topks", type=str, default="[20]")
    parser.add_argument("--recdim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
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
        help="Where to write parsed final metrics as JSON.",
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
            "Could not parse test metrics from LightGCN output. "
            "Ensure training emitted at least one [TEST] block."
        )

    last_metrics = metric_matches[-1]
    precision_vals = _parse_float_list(last_metrics.group(1))
    recall_vals = _parse_float_list(last_metrics.group(2))
    ndcg_vals = _parse_float_list(last_metrics.group(3))

    if not (
        len(precision_vals) == len(recall_vals) == len(ndcg_vals) == len(topks)
    ):
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


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

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

    cmd = [
        "python",
        "main.py",
        f"--decay={args.decay}",
        f"--lr={args.lr}",
        f"--layer={args.layer}",
        f"--seed={args.seed}",
        f"--dataset={args.dataset}",
        f"--topks={args.topks}",
        f"--recdim={args.recdim}",
        "--epochs",
        str(args.epochs),
    ]

    print("Running command:")
    print(" ".join(cmd))
    print(f"Working directory: {code_dir}")
    print()

    result = subprocess.run(
        cmd,
        cwd=str(code_dir),
        capture_output=True,
        text=True,
        check=False,
    )

    # Echo training logs for visibility.
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"LightGCN training failed with exit code {result.returncode}.")

    metrics = parse_training_output(result.stdout, topks=topks)
    output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nParsed final metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics JSON: {output_json}")


if __name__ == "__main__":
    main()
