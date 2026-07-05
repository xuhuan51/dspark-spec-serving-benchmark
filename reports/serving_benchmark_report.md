# OpenAI-Compatible Serving Benchmark Report

## Goal

Measure whether speculative decoding produces real end-to-end serving speedups
after being connected to an OpenAI-compatible LLM serving backend.

The benchmark compares baseline decoding and draft-model speculative decoding
under matched request shapes.

## Metrics

- TTFT: time to first token
- TPOT: time per output token
- P95 latency
- client-side tokens/s
- engine-side generation throughput
- speculative accepted length
- GPU memory usage
- KV-cache pressure

## Main Result

| Configuration | c=1 Speedup | Observed Breakpoint | Interpretation |
| --- | ---: | ---: | --- |
| Qwen3-8B BF16, single A30 | 1.76x | ~c=26 | best low-concurrency serving fit |
| Qwen3-32B BF16, TP8 | 1.57x | ~c=8 | TP communication reduces verification benefit |
| Qwen3-32B INT4, TP4 | 1.43x | ~c=5 | quantization already reduces the memory-bound decode bottleneck |

## Qwen3-8B BF16 Concurrency Ladder

Single A30, Qwen3-8B BF16 target, EAGLE3 draft model, 256-token decode budget.

| Concurrency | Speedup |
| ---: | ---: |
| 1 | 1.76x |
| 2 | 1.72x |
| 4 | 1.62x |
| 8 | 1.56x |
| 16 | 1.30x |
| 32 | 0.93x |

The useful region is low to moderate concurrency. Once the baseline backend is
already saturated, the draft pass and verification overhead can consume the
available headroom.

## Qwen3-32B BF16 TP8

| Concurrency | Speedup |
| ---: | ---: |
| 1 | 1.57x |
| 2 | 1.52x |
| 4 | 1.33x |
| 8 | 0.99x |
| 16 | 0.67x |
| 24 | 0.74x |
| 32 | 0.70x |

The 32B TP8 result is the key systems lesson. A larger target model does not
automatically make speculative decoding better. Tensor-parallel verification
adds communication cost, and the target backend saturates earlier.

## Qwen3-32B INT4 TP4

| Metric | BF16 | INT4 |
| --- | ---: | ---: |
| Mean accepted length | 2.11 | 1.89 |
| Draft acceptance rate | 27.7% | 22.2% |
| c=1 end-to-end speedup | 1.57x | 1.43x |

Quantization and speculative decoding both try to reduce decode cost. Their
benefits are not additive by default. INT4 improves the baseline decode path,
leaving less memory-bound slack for speculative decoding to exploit.

## Practical Conclusion

Speculative decoding should be treated as a policy-controlled serving feature,
not as a global switch.

It is useful when:

- output length is large enough for decode to dominate,
- concurrency is low or moderate,
- the draft model matches the workload distribution,
- target verification still has unused compute/communication headroom.

It is less useful when:

- the backend is already saturated,
- tensor-parallel communication dominates verification,
- quantization already removes most of the memory-bound decode bottleneck,
- the draft model has low accepted length on the workload.

The benchmark scripts in this repository are intended to make those tradeoffs
measurable before deployment.
