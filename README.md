# DeepSeek DSpark Speculative Decoding Serving Benchmark

<p align="center">
  <img src="assets/overview.png" alt="DSpark speculative decoding serving benchmark overview" width="920">
</p>

<p align="center">
  <b>基于 DeepSeek DSpark 的 LLM 投机解码推理加速优化</b>
</p>

<p align="center">
  <a href="reports/dspark_reproduction.md">DSpark Reproduction</a> ·
  <a href="reports/vllm026_dspark_serving_validation.md">Native DSpark Serving</a> ·
  <a href="reports/vllm026_dspark_32b_cross_target.md">32B Stress Test</a> ·
  <a href="reports/serving_benchmark_report.md">Serving Benchmark</a> ·
  <a href="reports/adaptive_policy.md">Adaptive Policy</a> ·
  <a href="reports/deployment_notes.md">Deployment Notes</a> ·
  <a href="results/serving_speedup_summary.csv">Result CSV</a>
</p>

This repository evaluates DeepSeek DSpark-style speculative decoding as an
OpenAI-compatible LLM serving optimization. It includes three connected parts:
DeepSpec/DSpark reproduction, end-to-end serving benchmark, and an adaptive
policy that decides when traffic should use speculative decoding.

The project answers a practical systems question:

> When does speculative decoding actually reduce wall-clock latency for an LLM
> service?

The project does **not** train a new draft model. It uses existing DSpark /
DeepSpec and supported draft-model serving paths to build a reproducible
benchmark, compare baseline vs speculative decoding, and derive a
policy-controlled routing strategy.

## What This Project Implements

```text
paper-level DSpark signal
  -> DeepSpec accepted-length reproduction
  -> OpenAI-compatible baseline/spec serving benchmark
  -> concurrency breakpoint analysis
  -> adaptive baseline/spec routing policy
```

| Component | Implementation |
| --- | --- |
| Algorithm reproduction | Reproduces DSpark / EAGLE3 / DFlash accepted-length results on Qwen3-8B |
| Serving benchmark | Uses OpenAI-compatible endpoints to compare target-only vs speculative decoding |
| Benchmark driver | Collects TTFT, TPOT, P95, tokens/s, accepted length, and backend metrics |
| Policy fitter | Converts measured concurrency ladder results into `configs/adaptive_spec_policy.json` |
| Runtime policy | Routes requests to speculative backend only inside benchmark-validated regions |
| Simulator | Replays measured ladder points to compare always-speculative vs adaptive routing |

## Highlights

| Area | What This Repository Provides |
| --- | --- |
| Algorithm validation | Reproduces DeepSpec DSpark / EAGLE3 / DFlash accepted length on Qwen3-8B |
| Serving benchmark | Runs OpenAI-compatible baseline/speculative endpoints with matched request shapes |
| Metrics | TTFT, TPOT, P95 latency, client tokens/s, engine throughput, accepted length |
| Systems analysis | Studies concurrency, tensor parallelism, quantization, draft overhead, and breakpoints |
| Adaptive policy | Fits benchmark-derived thresholds and routes traffic to baseline or speculative backend |
| Outputs | Scripts, policy config, CSV results, reproduction reports, and deployment notes |

## Key Results

<p align="center">
  <img src="assets/speedup_summary.png" alt="End-to-end speculative decoding serving speedup summary" width="920">
</p>

### DeepSpec / DSpark Reproduction

On Qwen3-8B with official DeepSpec-style checkpoints:

| Metric | Result |
| --- | ---: |
| DSpark macro accepted length | 5.021 |
| DSpark vs EAGLE3 | +26.4% |
| DSpark vs DFlash | +18.6% |

The reproduced result is close to the paper-level claim:

| Comparison | Reproduced | Paper |
| --- | ---: | ---: |
| DSpark vs EAGLE3 | +26.4% | +26.7% |
| DSpark vs DFlash | +18.6% | +18.4% |

### OpenAI-Compatible Serving Speedup

The serving experiments compare target-only decoding with speculative decoding
under the same request shape.

| Configuration | Serving draft | c=1 Speedup | Breakpoint | Main Bottleneck |
| --- | --- | ---: | ---: | --- |
| Qwen3-8B BF16, single A30 | AngelSlim EAGLE3 | 1.76x | ~c=26 | draft overhead and batching budget |
| Qwen3-32B BF16, TP8 | AngelSlim EAGLE3 | 1.57x | ~c=8 | tensor-parallel communication |
| Qwen3-32B INT4, TP4 | AngelSlim EAGLE3 | 1.43x | ~c=5 | quantization reduces decode bottleneck |

Main conclusion: speculative decoding should be treated as a
policy-controlled serving optimization. It is most useful for low-to-moderate
concurrency, long-output, domain-matched workloads where decode remains the
bottleneck.

### Native DSpark Serving on vLLM 0.26.0

The original DeepSpec `Qwen3DSparkModel` checkpoint now runs directly through
vLLM's native `dspark` serving method. A matched Qwen3-8B BF16 decode-only
validation on one A30 per endpoint produced:

| Concurrency | Throughput speedup | TPOT speedup | Accepted length |
| ---: | ---: | ---: | ---: |
| 1 | 2.05x | 2.08x | 2.63 |
| 4 | 1.89x | 1.91x | 2.57 |
| 16 | 1.61x | 1.63x | 2.67 |

See `reports/vllm026_dspark_serving_validation.md` for the full five-point
ladder and correctness notes.

### Qwen3-32B Cross-Target Stress Test

The DeepSpec checkpoint set available for this experiment did not include a
matched Qwen3-32B DSpark drafter. Qwen3-14B and Qwen3-32B share the hidden
width and vocabulary needed by the runtime, so the 14B DSpark checkpoint was
connected to the 32B target as an intentional mismatch stress test:

| Target configuration | c=1 | c=2 | c=4 | c=8 | c=16 | Accepted length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-32B BF16, TP8 | 1.17x | 1.12x | 0.85x | 0.58x | 0.47x | 1.48-1.53 |
| Qwen3-32B INT4, TP4 | 1.00x | 0.90x | 0.63x | 0.45x | 0.37x | 1.41-1.52 |

The BF16 target retained a small low-concurrency region because its baseline
decode was expensive. INT4 removed most of that headroom, and the mismatched
draft plus verification overhead became negative almost immediately.

These rows are **not** matched Qwen3-32B DSpark results. They are evidence for
the routing policy's model-match, quantization, and concurrency guards. See
`reports/vllm026_dspark_32b_cross_target.md` for the full results.

### Adaptive Routing Policy

The project includes a small policy layer that converts benchmark results into
runtime routing decisions:

```text
request + runtime signals
  -> adaptive policy
  -> speculative backend if inside benchmark-validated region
  -> baseline backend otherwise
```

The policy config is generated from benchmark CSVs, not hardcoded in the router:

```bash
python3 benchmark/fit_adaptive_policy.py
python3 benchmark/simulate_adaptive_policy.py
```

On the available concurrency ladder, the adaptive policy routes 8 profitable
points to speculative decoding and falls back to baseline on 5 saturated or
non-profitable points.

Fitted safe regions:

| Configuration | Policy Source | Safe Spec Concurrency |
| --- | --- | ---: |
| Qwen3-8B BF16, single A30 | measured ladder | 16 |
| Qwen3-32B BF16, TP8 | measured ladder | 4 |
| Qwen3-32B INT4, TP4 | summary breakpoint | 5 |

The safe region is intentionally more conservative than the approximate
breakpoint reported in the benchmark summary. The policy uses the largest
measured profitable concurrency point, then falls back to baseline outside that
validated region.

## Methodology

The project is split into two stages.

| Stage | Purpose | Output |
| --- | --- | --- |
| A. DeepSpec reproduction | Validate DSpark-side accepted-length improvement against EAGLE3 and DFlash | `results/deepspec_qwen3_8b_acceptance.csv` |
| B. Serving A/B benchmark | Measure whether the algorithmic signal becomes wall-clock serving speedup | `results/serving_speedup_summary.csv` |
| C. Adaptive policy | Fit safe speculative regions and simulate baseline/spec routing | `configs/adaptive_spec_policy.json` |

Serving benchmark shape:

```text
client
  -> /v1/chat/completions
  -> baseline endpoint: target model only
  -> spec endpoint: target model + draft model + speculative verification
  -> benchmark driver: latency, throughput, accepted length, backend metrics
```

This separates the algorithm metric, `accepted length`, from the serving metric,
`wall-clock latency and throughput`.

## Quick Start

### 1. Install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Reproduce DeepSpec accepted length

```bash
bash scripts/run_deepspec_eval_qwen3_8b.sh
```

Expected logs are written under:

```text
outputs/deepspec-runs/
```

### 3. Start a baseline serving endpoint

```bash
MODEL_PATH=/home/liuguangli/models/Qwen3-8B \
PORT=8550 \
GPUS=0 \
TP=1 \
bash scripts/run_vllm_local_baseline.sh
```

### 4. Start a native DSpark endpoint

```bash
MODEL_PATH=/home/liuguangli/models/Qwen3-8B \
DRAFT_MODEL_PATH=/home/liuguangli/models/dspark_qwen3_8b_block7 \
PORT=8551 \
GPUS=1 \
TP=1 \
bash scripts/run_vllm_dspark.sh
```

### 5. Run the concurrency ladder

```bash
python3 benchmark/spec_decode_microbench.py \
  --tag qwen3_8b_spec \
  --base-url http://127.0.0.1:8550/v1 \
  --out-dir outputs/qwen3_8b_spec \
  --concurrencies 1,2,4,8,16,24,32 \
  --max-tokens 256 \
  --ignore-eos
```

The driver writes per-request JSONL, sampled backend metrics, and a compact
`ladder.csv` summary under the selected output directory.

### 6. Fit and simulate adaptive policy

Generate the policy config from benchmark artifacts:

```bash
python3 benchmark/fit_adaptive_policy.py \
  --ladder-csv results/serving_concurrency_ladder.csv \
  --summary-csv results/serving_speedup_summary.csv \
  --out configs/adaptive_spec_policy.json \
  --min-speedup 1.05
```

Replay measured concurrency points with adaptive routing:

```bash
python3 benchmark/simulate_adaptive_policy.py \
  --policy-config configs/adaptive_spec_policy.json \
  --ladder-csv results/serving_concurrency_ladder.csv \
  --out results/adaptive_policy_decisions.csv
```

Example runtime decision:

```bash
python3 benchmark/adaptive_policy.py \
  --configuration qwen3_32b_bf16_tp8 \
  --concurrency 16 \
  --expected-output-tokens 256 \
  --accepted-length 2.5
```

Expected output:

```json
{
  "backend": "baseline",
  "enable_speculative": false,
  "reason": "above_measured_concurrency_region"
}
```

## Repository Layout

```text
.
├── assets/
│   ├── generate_readme_figures.py
│   ├── overview.png
│   └── speedup_summary.png
├── benchmark/
│   ├── adaptive_policy.py
│   ├── fit_adaptive_policy.py
│   ├── simulate_adaptive_policy.py
│   └── spec_decode_microbench.py
├── configs/
│   ├── adaptive_spec_policy.json
│   └── qwen3_spec_benchmark.env.example
├── reports/
│   ├── adaptive_policy.md
│   ├── deployment_notes.md
│   ├── dspark_reproduction.md
│   ├── serving_benchmark_report.md
│   ├── vllm026_dspark_32b_cross_target.md
│   └── vllm026_dspark_serving_validation.md
├── results/
│   ├── adaptive_policy_decisions.csv
│   ├── deepspec_qwen3_8b_acceptance.csv
│   ├── serving_concurrency_ladder.csv
│   ├── serving_speedup_summary.csv
│   ├── vllm026_qwen3_32b_dspark_cross_target.csv
│   └── vllm026_qwen3_8b_dspark_serving.csv
└── scripts/
    ├── run_decode_ladder.sh
    ├── run_deepspec_eval_qwen3_8b.sh
    ├── run_vllm_baseline.sh
    ├── run_vllm_dspark.sh
    ├── run_vllm_local_baseline.sh
    └── run_vllm_spec.sh
```

## Hardware and Runtime

Original experiments were run on:

- 8x NVIDIA A30 24GB
- Qwen3-8B and Qwen3-32B target models
- DSpark / DFlash / EAGLE3 official or open checkpoints
- vLLM / SGLang OpenAI-compatible serving interfaces

The scripts are parameterized through environment variables so the benchmark
can be rerun on different GPU topologies, model paths, and serving backends.

## Reports

- [DSpark reproduction report](reports/dspark_reproduction.md)
- [Native DSpark serving validation](reports/vllm026_dspark_serving_validation.md)
- [Qwen3-32B cross-target stress test](reports/vllm026_dspark_32b_cross_target.md)
- [OpenAI-compatible serving benchmark report](reports/serving_benchmark_report.md)
- [Adaptive speculative decoding policy](reports/adaptive_policy.md)
- [Deployment notes](reports/deployment_notes.md)

## Scope and Limitations

The original full serving matrix predates native DSpark support and uses
AngelSlim EAGLE3 checkpoints. The newer vLLM 0.26.0 validation adds a matched
Qwen3-8B + DSpark path. Its Qwen3-32B extension uses the compatible Qwen3-14B
DSpark checkpoint as an explicit cross-target mismatch test because no matched
Qwen3-32B DSpark checkpoint was available in the evaluated release set.

This distinction keeps the algorithm question (`accepted length`) separate
from the serving question (`wall-clock latency and throughput`), while making
the runtime used by each result explicit.
