#!/usr/bin/env python3
"""Replay benchmark ladder results with the adaptive speculative policy."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from adaptive_policy import RuntimeSignals, decide, load_policy_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-config", default="configs/adaptive_spec_policy.json")
    parser.add_argument("--ladder-csv", default="results/serving_concurrency_ladder.csv")
    parser.add_argument("--out", default="results/adaptive_policy_decisions.csv")
    parser.add_argument("--expected-output-tokens", type=int, default=256)
    args = parser.parse_args()

    policies = load_policy_config(args.policy_config)
    out_rows: list[dict[str, str]] = []

    with Path(args.ladder_csv).open(newline="") as fh:
        for row in csv.DictReader(fh):
            config = row["configuration"]
            speedup = float(row["speedup"])
            policy = policies[config]
            decision = decide(
                RuntimeSignals(
                    configuration=config,
                    concurrency=int(row["concurrency"]),
                    expected_output_tokens=args.expected_output_tokens,
                    accepted_length=float(policy["min_accept_length"]) + 0.5,
                ),
                policies,
            )
            adaptive_speedup = speedup if decision.enable_speculative else 1.0
            out_rows.append(
                {
                    "configuration": config,
                    "concurrency": row["concurrency"],
                    "measured_spec_speedup": f"{speedup:.2f}",
                    "adaptive_backend": decision.backend,
                    "adaptive_speedup": f"{adaptive_speedup:.2f}",
                    "avoided_regression": str(
                        (not decision.enable_speculative)
                        and speedup < float(policy["min_speedup"])
                    ).lower(),
                    "reason": decision.reason,
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)

    for row in out_rows:
        print(row)


if __name__ == "__main__":
    main()
