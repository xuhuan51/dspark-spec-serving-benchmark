#!/usr/bin/env python3
"""Fit adaptive speculative-decoding policy config from benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def fit_from_ladder(rows: list[dict[str, str]], min_speedup: float) -> dict[str, dict[str, Any]]:
    by_config: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_config[row["configuration"]].append(row)

    out: dict[str, dict[str, Any]] = {}
    for config, config_rows in by_config.items():
        points = sorted(
            (int(row["concurrency"]), float(row["speedup"])) for row in config_rows
        )
        profitable = [c for c, speedup in points if speedup >= min_speedup]
        if profitable:
            max_spec_concurrency = max(profitable)
            source = "ladder"
        else:
            max_spec_concurrency = 0
            source = "ladder_no_profitable_region"

        out[config] = {
            "max_spec_concurrency": max_spec_concurrency,
            "min_speedup": min_speedup,
            "min_output_tokens": 64,
            "min_accept_length": 2.0,
            "min_tpot_gain": round(min_speedup - 1.0, 4),
            "min_gpu_memory_headroom": 0.05,
            "source": source,
            "fit_points": [
                {"concurrency": c, "speedup": speedup} for c, speedup in points
            ],
        }
    return out


def add_summary_fallbacks(
    policies: dict[str, dict[str, Any]],
    summary_rows: list[dict[str, str]],
    min_speedup: float,
) -> None:
    for row in summary_rows:
        config = row["configuration"]
        if config in policies:
            continue
        policies[config] = {
            "max_spec_concurrency": int(float(row["breakpoint"])),
            "min_speedup": min_speedup,
            "min_output_tokens": 64,
            "min_accept_length": 2.0,
            "min_tpot_gain": round(min_speedup - 1.0, 4),
            "min_gpu_memory_headroom": 0.05,
            "source": "summary_breakpoint",
            "fit_points": [
                {
                    "concurrency": 1,
                    "speedup": float(row["c1_speedup"]),
                }
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ladder-csv", default="results/serving_concurrency_ladder.csv")
    parser.add_argument("--summary-csv", default="results/serving_speedup_summary.csv")
    parser.add_argument("--out", default="configs/adaptive_spec_policy.json")
    parser.add_argument("--min-speedup", type=float, default=1.05)
    args = parser.parse_args()

    ladder_rows = read_csv(Path(args.ladder_csv))
    summary_rows = read_csv(Path(args.summary_csv))
    policies = fit_from_ladder(ladder_rows, args.min_speedup)
    add_summary_fallbacks(policies, summary_rows, args.min_speedup)

    payload = {
        "description": "Benchmark-derived adaptive speculative decoding policy.",
        "min_speedup_definition": "A concurrency point is profitable when speculative speedup is >= min_speedup.",
        "policies": policies,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
