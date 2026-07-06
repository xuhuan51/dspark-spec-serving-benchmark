#!/usr/bin/env python3
"""Adaptive routing policy for speculative decoding backends.

The policy is intentionally small: it consumes benchmark-derived config and
runtime signals, then returns either the baseline or speculative backend.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeSignals:
    configuration: str
    concurrency: int
    expected_output_tokens: int
    accepted_length: float | None = None
    recent_spec_tpot_ms: float | None = None
    recent_baseline_tpot_ms: float | None = None
    gpu_memory_headroom: float | None = None


@dataclass(frozen=True)
class Decision:
    backend: str
    enable_speculative: bool
    reason: str


def load_policy_config(path: str | Path) -> dict[str, dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    return data["policies"]


def decide(signals: RuntimeSignals, policies: dict[str, dict[str, Any]]) -> Decision:
    if signals.configuration not in policies:
        return Decision("baseline", False, "unknown_configuration")

    policy = policies[signals.configuration]
    max_concurrency = int(policy["max_spec_concurrency"])
    min_output_tokens = int(policy["min_output_tokens"])
    min_accept_length = float(policy["min_accept_length"])
    min_gpu_headroom = float(policy.get("min_gpu_memory_headroom", 0.0))
    min_tpot_gain = float(policy["min_tpot_gain"])

    if signals.concurrency > max_concurrency:
        return Decision("baseline", False, "above_measured_concurrency_region")

    if signals.expected_output_tokens < min_output_tokens:
        return Decision("baseline", False, "short_output_decode_not_dominant")

    if signals.accepted_length is not None and signals.accepted_length < min_accept_length:
        return Decision("baseline", False, "low_accepted_length")

    if (
        signals.gpu_memory_headroom is not None
        and signals.gpu_memory_headroom < min_gpu_headroom
    ):
        return Decision("baseline", False, "insufficient_gpu_memory_headroom")

    if (
        signals.recent_spec_tpot_ms is not None
        and signals.recent_baseline_tpot_ms is not None
    ):
        required_tpot = signals.recent_baseline_tpot_ms * (1.0 - min_tpot_gain)
        if signals.recent_spec_tpot_ms > required_tpot:
            return Decision("baseline", False, "insufficient_recent_tpot_gain")

    return Decision("speculative", True, "within_benchmark_validated_region")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-config", default="configs/adaptive_spec_policy.json")
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--expected-output-tokens", type=int, default=256)
    parser.add_argument("--accepted-length", type=float)
    parser.add_argument("--recent-spec-tpot-ms", type=float)
    parser.add_argument("--recent-baseline-tpot-ms", type=float)
    parser.add_argument("--gpu-memory-headroom", type=float)
    args = parser.parse_args()

    policies = load_policy_config(args.policy_config)
    decision = decide(
        RuntimeSignals(
            configuration=args.configuration,
            concurrency=args.concurrency,
            expected_output_tokens=args.expected_output_tokens,
            accepted_length=args.accepted_length,
            recent_spec_tpot_ms=args.recent_spec_tpot_ms,
            recent_baseline_tpot_ms=args.recent_baseline_tpot_ms,
            gpu_memory_headroom=args.gpu_memory_headroom,
        ),
        policies,
    )
    print(json.dumps(decision.__dict__, indent=2))


if __name__ == "__main__":
    main()
