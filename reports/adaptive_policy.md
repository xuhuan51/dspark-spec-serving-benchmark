# Adaptive Speculative Decoding Policy

## Goal

Add a policy-controlled routing layer for speculative decoding.

The policy decides whether a request should use:

```text
baseline backend:    target model only
speculative backend: target model + draft model + verification
```

This turns speculative decoding from a global serving switch into a conditional
optimization.

## Why This Is Not Hardcoded

The policy does not encode model names or fixed concurrency thresholds inside
the routing logic. It uses a generated config:

```text
configs/adaptive_spec_policy.json
```

The config is produced from benchmark artifacts:

```bash
python3 benchmark/fit_adaptive_policy.py \
  --ladder-csv results/serving_concurrency_ladder.csv \
  --summary-csv results/serving_speedup_summary.csv \
  --out configs/adaptive_spec_policy.json \
  --min-speedup 1.05
```

For ladder data, a concurrency point is considered profitable when measured
speculative speedup is at least `1.05x`. The policy selects the largest measured
profitable concurrency as the safe speculative region.

For configurations without full ladder data, the fitter falls back to the
reported benchmark summary breakpoint and marks the source as
`summary_breakpoint`.

## Runtime Signals

The runtime decision consumes:

- model / hardware / serving configuration id,
- current request concurrency,
- expected output length,
- speculative accepted length when available,
- recent baseline/speculative TPOT window when available,
- GPU memory headroom when available.

Decision rule:

```text
enable speculative decoding if:
  concurrency <= benchmark-derived safe concurrency
  and expected output tokens >= minimum decode length
  and accepted length is not below threshold
  and recent TPOT gain is not below threshold
  and GPU memory headroom is sufficient
otherwise use baseline decoding
```

## Fitted Policy

| Configuration | Policy Source | Safe Spec Concurrency | Minimum Speedup |
| --- | --- | ---: | ---: |
| Qwen3-8B BF16 single A30 | ladder | 16 | 1.05x |
| Qwen3-32B BF16 TP8 | ladder | 4 | 1.05x |
| Qwen3-32B INT4 TP4 | summary breakpoint | 5 | 1.05x |

The safe concurrency can be stricter than the approximate breakpoint in the
serving report. That is intentional: the adaptive policy uses the largest
measured profitable ladder point to avoid routing traffic into uncertain or
saturated regions.

## Simulation Result

Replay command:

```bash
python3 benchmark/simulate_adaptive_policy.py \
  --policy-config configs/adaptive_spec_policy.json \
  --ladder-csv results/serving_concurrency_ladder.csv \
  --out results/adaptive_policy_decisions.csv
```

On the available ladder results:

| Metric | Result |
| --- | ---: |
| Ladder points replayed | 13 |
| Routed to speculative backend | 8 |
| Routed to baseline backend | 5 |
| Regression / non-profitable points avoided | 5 |

The policy preserves the profitable low-concurrency speculative region and
falls back to baseline in saturated regions such as Qwen3-32B BF16 TP8 at
`c>=8`.

## Example Decision

```bash
python3 benchmark/adaptive_policy.py \
  --configuration qwen3_32b_bf16_tp8 \
  --concurrency 16 \
  --expected-output-tokens 256 \
  --accepted-length 2.5
```

Output:

```json
{
  "backend": "baseline",
  "enable_speculative": false,
  "reason": "above_measured_concurrency_region"
}
```

## Engineering Interpretation

This policy layer is the practical serving component of the project:

- benchmark scripts measure where speculative decoding is useful,
- the fitter converts benchmark results into routing config,
- the runtime policy uses current service signals to choose the backend,
- fallback to baseline avoids treating speculative decoding as an always-on
  optimization.

