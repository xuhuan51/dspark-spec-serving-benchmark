# DSpark Multi-GPU Serving

<p align="center">
  <b>Workload-aware verification scheduling and CUDA Graph bucketing for DSpark inference in vLLM.</b>
</p>

<p align="center">
  <b>English</b> · <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="#key-features">Features</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#performance">Performance</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="benchmark/qwen38_multigpu/README.md">Reproduction Guide</a>
</p>

## Overview

A vLLM-based project for deploying and optimizing DSpark speculative decoding on multiple GPUs. It implements **verification-budget scheduling** and **shape-aware CUDA Graph dispatch**, with paired benchmarks to measure their effect on serving throughput and token latency.

The primary workload is **Qwen3.8-27B BF16 + a matched ~1.99B DSpark drafter on 4×NVIDIA A30, TP4**. The target uses hybrid attention with Gated DeltaNet (GDN) layers.

## Key Features

- **Workload-aware verification.** Select a uniform verification prefix from the actual active batch using a frozen calibration policy. The drafter keeps its full 7-token block. [Scheduler adapter](benchmark/qwen38_multigpu/uniform_budget.py)
- **Shape-aware CUDA Graphs.** Bucket verification by query width and request count to reduce graph padding while preserving the draft capture path. [Graph adapter](benchmark/qwen38_multigpu/uniform_graphs.py)
- **Controlled multi-GPU evaluation.** Compare AR, original DSpark, graph-only and combined policies with randomized paired runs; collect capacity and target-decision diagnostics separately from timing. [Experiment driver](benchmark/qwen38_multigpu/threeway.py)

## Architecture

<p align="center">
  <img src="assets/overview.png" alt="DSpark decode loop: full draft proposals, local verification-budget scheduler, local graph dispatcher, upstream target verification and state update" width="920">
</p>

The two local adapters control **how much of a proposal is verified** and **which captured shape executes it**. Model weights, rejection sampling and GDN state management use upstream implementations.

## Performance

**Setup:** vLLM 0.29.0 · 4×A30 TP4 · BF16 · ~2K input / 256 output tokens · 4.5 GiB cache budget per GPU · five paired rounds.

### Native DSpark vs. autoregressive decoding

<p align="center">
  <img src="assets/qwen38_speedup.png" alt="Native DSpark at concurrency 1/4/16: throughput 2.433/1.688/0.898 times AR, client TPOT reduced by 63.9/55.9/27.3 percent" width="920">
</p>

### Verification budget and graph bucketing

<p align="center">
  <img src="assets/qwen38_optimization.png" alt="Graph padding 1.41 to 1.09; combined policy throughput gain 6.3/7.8 percent at concurrency 4/16 over original DSpark policy; c=16 direct AR throughput ratio 0.995" width="920">
</p>

The framework gains compare against the **original DSpark policy**. At concurrency 16, the optimized version reaches **near parity with AR**, rather than a stable throughput speedup. [Ablations and direct comparisons](reports/qwen38_dspark_closure_20261003.md)

<details>
<summary>Numerical results and comparison scopes</summary>

| Experiment | Concurrency | Comparison | Throughput | Client TPOT reduction |
| --- | ---: | --- | ---: | ---: |
| Independent deployments | 1 | Native DSpark / AR | **2.433×** | **63.9%** |
| Independent deployments | 4 | Native DSpark / AR | **1.688×** | **55.9%** |
| Independent deployments | 16 | Native DSpark / AR | 0.898× | 27.3% |
| Held-out ablation | 4 | Budget + graphs / original policy | **+6.3%** | — |
| Shared-pool direct comparison | 16 | Budget + graphs / original policy | **+7.8%** | — |
| Shared-pool direct comparison | 16 | Budget + graphs / AR | **0.995×** | **40.6%** |

- Ratios are medians of five **within-round ratios**. Independent deployments and shared-pool ablations have different controls; their numbers should be interpreted separately.
- The c=16 AR ratio ranges from **0.974× to 1.016×**. Graph-only padding is **1.09**; the combined policy is **1.10**. Padding counts captured / effective tokens, not GPU time.
- Throughput uses actual output token IDs over a finite closed-loop request group. Client TPOT includes scheduling and streaming delivery. Loading, warm-up, capture and diagnostic instrumentation are excluded from timing.
- Greedy diagnostics checked 13,056 emitted decisions against each path's target argmax. Long sequences can still diverge because of BF16 target-logit ordering changes; universal token-for-token equality has not been established.

Frozen inputs, original records and summaries: [results](results/qwen38_multigpu_20261003) · [SHA-256 manifest](results/qwen38_multigpu_20261003/bundle-manifest.json).

</details>

## Quick Start

### Run the multi-GPU comparison

Prepare four available A30 GPUs and the pinned target / draft model files using the [model setup guide](benchmark/qwen38_multigpu/README.md#gpu-environment). Install the experiment environment:

```bash
python3 -m venv benchmark/qwen38_multigpu/.venv
source benchmark/qwen38_multigpu/.venv/bin/activate
python -m pip install -r benchmark/qwen38_multigpu/requirements.txt
```

Set the model paths, verify the draft weights, then compare AR, original DSpark, fine graph buckets and the frozen budget policy:

```bash
export DSPARK_MODEL=/absolute/path/Qwen3.8-27B
export DSPARK_DRAFT_MODEL=/absolute/path/Qwen3.8-27B-speculator.dspark
export DSPARK_GPUS=0,1,2,3  # Choose four available GPUs

python benchmark/qwen38_multigpu/download_draft.py
python benchmark/qwen38_multigpu/threeway.py \
  --gpus "$DSPARK_GPUS" --rounds 5 \
  --profile configs/qwen38_uniform_budget.json \
  --output outputs/qwen38-threeway.json
python benchmark/qwen38_multigpu/analyze_closure.py outputs/qwen38-threeway.json
```

Use a fresh output filename for each run. The launcher checks GPU processes and memory, locks the experiment and cleans up only its own servers. The [full guide](benchmark/qwen38_multigpu/README.md#reproduction) covers the c=1/4/16 integration benchmark, calibration, ablations and profiling.

<details>
<summary>Verify the bundled results without a GPU</summary>

Only the Python standard library is needed:

```bash
python3 benchmark/qwen38_multigpu/reproduce_results.py \
  --output outputs/qwen38-evidence
```

The script verifies 27 compressed archives and their decompressed SHA-256 hashes, recomputes four summaries, and checks target argmax / TP-rank decisions. Choose a new, empty output directory.

</details>

## Documentation

| Resource | Contents |
| --- | --- |
| [Reproduction guide](benchmark/qwen38_multigpu/README.md) · [中文指南](benchmark/qwen38_multigpu/README.zh-CN.md) | Pinned models and runtime, commands, metrics and implementation boundaries |
| [Integration and ablations](reports/qwen38_dspark_tp4_20261003.md) | Calibration, held-out validation and component comparisons; Chinese |
| [Capacity and numerical diagnostics](reports/qwen38_dspark_closure_20261003.md) | Direct AR comparison, actual GDN allocations and greedy divergence analysis; Chinese |
| [Raw results](results/qwen38_multigpu_20261003) | Frozen workload, summaries, request / trace archives and provenance |

## Additional Benchmarks

The repository also evaluates **Qwen3 / EAGLE3** across single-GPU, TP8 BF16 and TP4 INT4 deployments. This is a separate model / drafter study of concurrency and quantization boundaries.

<p align="center">
  <img src="assets/speedup_summary.png" alt="Separate Qwen3 EAGLE3 study: c=1 throughput ratios 1.76x for 8B single A30, 1.57x for 32B BF16 TP8 and 1.43x for 32B INT4 TP4" width="920">
</p>

<details>
<summary>Supporting experiments and figure generation</summary>

- [DSpark / DeepSpec acceptance-length reproduction](reports/dspark_reproduction.md): Qwen3-8B, eight datasets, DSpark / EAGLE3 / DFlash comparison.
- [Native DSpark serving](reports/vllm026_dspark_serving_validation.md): matched Qwen3-8B on vLLM 0.26.0, single A30.
- [Serving benchmark](reports/serving_benchmark_report.md): EAGLE3 concurrency ladders across target sizes, TP and precision.
- [Cross-target draft stress test](reports/vllm026_dspark_32b_cross_target.md): 32B target with a 14B DSpark drafter.
- [Backend routing policy](reports/adaptive_policy.md): thresholds fitted from the corresponding CSVs and offline routing replay. This is separate from the Qwen3.8 verification-prefix adapter; automatic Qwen3.8 AR fallback is not integrated.
- [Deployment notes](reports/deployment_notes.md): API shape and routing considerations.

Performance figures read the committed JSON / CSV records and share a common palette:

```bash
python3 -m pip install -r assets/requirements.txt
python3 assets/generate_readme_figures.py
```

</details>
