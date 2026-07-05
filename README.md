# DeepSeek DSpark Speculative Decoding Serving Benchmark

**基于 DeepSeek DSpark 的 LLM 投机解码推理加速优化**

This repository is a reproducible AI infrastructure project for evaluating
DeepSeek DSpark-style speculative decoding as an OpenAI-compatible LLM serving
optimization.

The project connects two layers that are often discussed separately:

- algorithm-side acceptance length, reproduced with the DeepSpec / DSpark
  evaluator;
- serving-side wall-clock speedup, measured through OpenAI-compatible vLLM /
  SGLang endpoints under matched workloads.

The goal is not to train a new draft model. The goal is to measure when a
speculative-decoding algorithm improvement actually becomes end-to-end serving
speedup.

## Why This Project Exists

Speculative decoding can reduce autoregressive decode latency by using a small
draft model to propose tokens and a target model to verify them in batches.
However, higher accepted length does not automatically mean faster production
serving. Real speedup also depends on draft overhead, tensor parallelism,
quantization, KV-cache pressure, batching, and request concurrency.

This project builds a complete benchmark path:

```text
DeepSpec / DSpark evaluator
        |
        v
accepted-length reproduction
        |
        v
OpenAI-compatible baseline/spec serving endpoints
        |
        v
TTFT / TPOT / P95 / tokens/s / accepted-length measurement
        |
        v
deployment policy and breakpoint analysis
```

## What Is Included

- DeepSpec reproduction scripts for DSpark, EAGLE3, and DFlash on Qwen3-8B.
- OpenAI-compatible serving benchmark scripts for baseline vs speculative
  decoding.
- Concurrency ladder benchmark driver for TTFT, TPOT, P95 latency, tokens/s,
  accepted length, and engine metrics.
- Result CSVs and reports for Qwen3-8B and Qwen3-32B serving experiments.
- Deployment notes describing when speculative decoding should be enabled.

## Key Results

### 1. DeepSpec / DSpark Reproduction

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

### 2. End-to-End Serving Speedup

OpenAI-compatible serving benchmark with Qwen3 target models and supported
EAGLE3 draft-model serving paths:

| Configuration | c=1 Speedup | Breakpoint | Main Bottleneck |
| --- | ---: | ---: | --- |
| Qwen3-8B BF16, single A30 | 1.76x | ~c=26 | draft overhead and batching budget |
| Qwen3-32B BF16, TP8 | 1.57x | ~c=8 | tensor-parallel communication |
| Qwen3-32B INT4, TP4 | 1.43x | ~c=5 | quantization reduces decode bottleneck |

Main conclusion: speculative decoding is useful as a policy-controlled serving
optimization, especially for low-to-moderate concurrency, long-output,
domain-matched workloads where decode remains the bottleneck.

## Methodology

The benchmark is split into two stages.

### Stage A: Algorithm Reproduction

DeepSpec's evaluator is used to reproduce accepted length for DSpark, EAGLE3,
and DFlash on Qwen3-8B. This validates that the DSpark-side algorithmic signal
is present before testing serving behavior.

Output:

```text
results/deepspec_qwen3_8b_acceptance.csv
reports/dspark_reproduction.md
```

### Stage B: Serving A/B Benchmark

The serving benchmark starts two OpenAI-compatible endpoints with the same
target model and request shape:

```text
baseline endpoint: target model only
spec endpoint:     target model + draft model + speculative decoding
```

The benchmark driver sends streaming `/v1/chat/completions` requests and records
client-side latency plus engine-side metrics exposed by the backend.

Output:

```text
results/serving_speedup_summary.csv
reports/serving_benchmark_report.md
reports/deployment_notes.md
```

## Repository Layout

```text
.
├── benchmark/
│   └── spec_decode_microbench.py
├── configs/
│   └── qwen3_spec_benchmark.env.example
├── reports/
│   ├── deployment_notes.md
│   ├── dspark_reproduction.md
│   └── serving_benchmark_report.md
├── results/
│   ├── deepspec_qwen3_8b_acceptance.csv
│   └── serving_speedup_summary.csv
└── scripts/
    ├── run_decode_ladder.sh
    ├── run_deepspec_eval_qwen3_8b.sh
    ├── run_vllm_baseline.sh
    └── run_vllm_spec.sh
```

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
MODEL_PATH=/models/Qwen3-8B \
PORT=8550 \
TP=1 \
GPUS=0 \
bash scripts/run_vllm_baseline.sh
```

### 4. Start a speculative-decoding endpoint

```bash
MODEL_PATH=/models/Qwen3-8B \
DRAFT_MODEL_PATH=/models/Qwen3-8B_eagle3_angelslim \
PORT=8550 \
TP=1 \
GPUS=0 \
SPEC_TOKENS=4 \
bash scripts/run_vllm_spec.sh
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

## Hardware and Runtime

Original experiments were run on:

- 8x NVIDIA A30 24GB
- Qwen3-8B and Qwen3-32B target models
- DSpark / DFlash / EAGLE3 official or open checkpoints
- vLLM / SGLang OpenAI-compatible serving interfaces

The scripts are intentionally parameterized through environment variables so
the same benchmark can be rerun on different GPU topologies and model paths.

## Reports

- [DSpark reproduction report](reports/dspark_reproduction.md)
- [OpenAI-compatible serving benchmark report](reports/serving_benchmark_report.md)
- [Deployment notes](reports/deployment_notes.md)

## Scope and Limitations

This repository does not claim that DSpark itself is fully integrated into vLLM
serving. DSpark is used as the DeepSpec algorithm baseline and reproduction
target. The end-to-end serving experiments use draft-model paths supported by
the tested OpenAI-compatible serving stack.

This distinction is intentional: it separates the algorithm question
(`accepted length`) from the serving question (`wall-clock latency and
throughput`), which is the core systems problem the project evaluates.

