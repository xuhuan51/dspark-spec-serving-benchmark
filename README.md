# DeepSeek DSpark Speculative Decoding Serving Benchmark

An open, reproducible AI infrastructure project for evaluating speculative
decoding from paper-level acceptance metrics to end-to-end OpenAI-compatible LLM
serving speedups.

The project is built around DeepSeek DeepSpec / DSpark, vLLM/SGLang serving, and
Qwen3 target models on 8x NVIDIA A30 GPUs.

## What This Project Does

- Reproduces DeepSpec DSpark / EAGLE3 / DFlash acceptance-length results on
  Qwen3-8B.
- Builds an OpenAI-compatible serving benchmark for baseline vs speculative
  decoding.
- Measures end-to-end serving metrics: TTFT, TPOT, P95 latency, tokens/s,
  accepted length, GPU memory, and KV cache pressure.
- Studies when accepted length turns into real wall-clock speedup, and how
  hardware topology, quantization, and concurrency change the benefit.
- Provides scripts, result tables, and reports that can be used as a clean
  benchmark baseline for future draft-model or serving-engine work.

## Key Results

### DeepSpec Reproduction

On Qwen3-8B with official DeepSpec checkpoints:

| Metric | Result |
| --- | ---: |
| DSpark macro accepted length | 5.021 |
| DSpark vs EAGLE3 | +26.4% |
| DSpark vs DFlash | +18.6% |

The reproduced DSpark gain matches the paper-level result closely
(paper: +26.7% vs EAGLE3, +18.4% vs DFlash).

### End-to-End Serving Speedup

OpenAI-compatible serving benchmark with Qwen3 / AngelSlim EAGLE3 draft models:

| Configuration | c=1 Speedup | Observed Breakpoint | Note |
| --- | ---: | ---: | --- |
| Qwen3-8B BF16, single A30 | 1.76x | ~c=26 | widest profitable region |
| Qwen3-32B BF16, TP8 | 1.57x | ~c=8 | tensor-parallel communication cost |
| Qwen3-32B INT4, TP4 | 1.43x | ~c=5 | quantization reduces decode bottleneck |

The main engineering conclusion is that speculative decoding is a conditional
serving optimization: it works best for low-concurrency, long-output,
domain-matched workloads where decode is still the bottleneck.

## Repository Layout

```text
.
├── benchmark/
│   └── spec_decode_microbench.py
├── configs/
│   └── qwen3_spec_benchmark.env.example
├── reports/
│   ├── dspark_reproduction.md
│   ├── serving_benchmark_report.md
│   └── deployment_notes.md
├── results/
│   ├── deepspec_qwen3_8b_acceptance.csv
│   └── serving_speedup_summary.csv
└── scripts/
    ├── run_deepspec_eval_qwen3_8b.sh
    ├── run_vllm_baseline.sh
    ├── run_vllm_spec.sh
    └── run_decode_ladder.sh
```

## Quick Start

### 1. DeepSpec acceptance reproduction

```bash
bash scripts/run_deepspec_eval_qwen3_8b.sh
```

Expected output logs are written under `outputs/deepspec-runs/`.

### 2. Start a baseline vLLM service

```bash
MODEL_PATH=/models/Qwen3-8B \
PORT=8550 \
TP=1 \
GPUS=0 \
bash scripts/run_vllm_baseline.sh
```

### 3. Start a speculative-decoding vLLM service

```bash
MODEL_PATH=/models/Qwen3-8B \
DRAFT_MODEL_PATH=/models/Qwen3-8B_eagle3_angelslim \
PORT=8550 \
TP=1 \
GPUS=0 \
SPEC_TOKENS=4 \
bash scripts/run_vllm_spec.sh
```

### 4. Run the decode ladder benchmark

```bash
python3 benchmark/spec_decode_microbench.py \
  --tag qwen3_8b_spec \
  --base-url http://127.0.0.1:8550/v1 \
  --out-dir outputs/qwen3_8b_spec \
  --concurrencies 1,2,4,8,16,24,32 \
  --max-tokens 256 \
  --ignore-eos
```

## Hardware Used

- 8x NVIDIA A30 24GB
- vLLM 0.24 development build for serving experiments
- DeepSpec official evaluator for acceptance-length reproduction
- Qwen3-8B / Qwen3-32B target models
- DSpark / DFlash / EAGLE3 official or open checkpoints

## Reports

- [DSpark reproduction report](reports/dspark_reproduction.md)
- [Serving benchmark report](reports/serving_benchmark_report.md)
- [Deployment notes](reports/deployment_notes.md)

## Scope

This project does not claim that DSpark itself is fully integrated into vLLM
serving. DSpark is used as the DeepSpec algorithm baseline and title hook.
End-to-end OpenAI-compatible serving tests use draft-model implementations that
are supported by the tested serving stack.

The value of the project is the full path from DSpark/DeepSpec reproduction to
serving-level A/B benchmark, speedup analysis, and deployment policy.
