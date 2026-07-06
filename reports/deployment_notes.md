# Deployment Notes

## Serving Shape

The project keeps the business-facing API unchanged:

```text
client -> OpenAI-compatible /v1/chat/completions -> vLLM/SGLang backend
```

Speculative decoding is enabled by changing the serving backend configuration,
not the client contract.

## Suggested Policy

Use speculative decoding as a conditional optimization. The repository includes
an executable policy implementation:

```bash
python3 benchmark/fit_adaptive_policy.py
python3 benchmark/simulate_adaptive_policy.py
```

The runtime decision shape is:

```text
if concurrency is inside the benchmark-derived safe region
   and expected_output_tokens is large enough
   and accepted_length is healthy
   and recent TPOT gain is sufficient
   and GPU memory headroom is sufficient:
       route to speculative backend
else:
       route to baseline backend
```

The fitted config is stored in `configs/adaptive_spec_policy.json`; the
simulation output is stored in `results/adaptive_policy_decisions.csv`.

## Signals To Monitor

- accepted length
- TPOT
- TTFT
- request P95 / P99
- output tokens/s
- GPU memory
- KV-cache usage
- preemption / allocation failures
- tensor-parallel communication overhead

## Operational Rules

1. Always keep a baseline decoding backend for fallback.
2. Benchmark baseline and speculative decoding with the same request shape.
3. Treat accepted length as a leading indicator, not the final KPI.
4. Re-evaluate after changing quantization, tensor parallelism, max sequence
   length, or draft model.
5. Do not assume a draft model trained on one domain works on every workload.

## Why DSpark Is Included

DSpark is the DeepSpec algorithm baseline and project title hook. It establishes
the paper-level accepted-length improvement. The end-to-end serving benchmark
then measures how that kind of algorithmic gain translates into OpenAI-compatible
serving speedups under real system constraints.
