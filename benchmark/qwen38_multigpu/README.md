# Qwen3.8-27B: DSpark Multi-GPU Optimization

English · [简体中文](README.zh-CN.md) · [Project home](../../README.md)

This directory provides vLLM scheduling / CUDA Graph adapters, experiment drivers and analysis tools for **Qwen3.8-27B BF16 + a matched DSpark drafter on 4×A30 TP4**. The drafter has approximately **1.99B parameters**; `27B` in its repository name identifies the target it matches.

## Implementation

- [`uniform_budget.py`](uniform_budget.py): extend V1 AsyncScheduler to choose a uniform verification prefix K from the actual active batch. The drafter still generates a full 7-token block; upstream rejection sampling and state management remain in use.
- [`uniform_graphs.py`](uniform_graphs.py): capture target query widths 2 / 3 / 5 / 8 and draft width 7, with additional request buckets 6 / 12 / 24. The c=16 graph-only ablation reduces captured / effective target-token padding from 1.41 to 1.09.
- [`threeway.py`](threeway.py): randomized AR / original DSpark / fine graph buckets / budget + graphs comparison, with a frozen policy and a shared 45-graph pool for the DSpark variants.
- [`admission_audit.py`](admission_audit.py), [`decision_audit.py`](decision_audit.py) and [`capacity_probe.py`](capacity_probe.py): observe real cache allocations, target argmax and emitted token IDs; diagnostics run outside the timed performance comparisons.

Results and component comparisons are summarized on the [project home](../../README.md#performance). Detailed reports are available in Chinese: [integration and ablations](../../reports/qwen38_dspark_tp4_20261003.md), [direct comparisons and diagnostics](../../reports/qwen38_dspark_closure_20261003.md).

## GPU Environment

The measured environment uses **vLLM 0.29.0**, **Torch 2.13.0** and **FlashInfer 0.6.18**, with a compatible CUDA driver. The adapters depend on the pinned vLLM internal interfaces. Install the dedicated [requirements](requirements.txt) from the repository root:

```bash
python3 -m venv benchmark/qwen38_multigpu/.venv
source benchmark/qwen38_multigpu/.venv/bin/activate
python -m pip install -r benchmark/qwen38_multigpu/requirements.txt
```

Prepare target weights and target / draft configuration and tokenizer files from these revisions:

| Model | Revision |
| --- | --- |
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| `RedHatAI/Qwen3.8-27B-speculator.dspark` | `87ca2fdc67f316f6c1f9ebc7ef7bbb0ad299a4ce` |

```bash
export DSPARK_MODEL=/absolute/path/Qwen3.8-27B
export DSPARK_DRAFT_MODEL=/absolute/path/Qwen3.8-27B-speculator.dspark
export DSPARK_GPUS=0,1,2,3  # Select four available cards
```

Once the draft configuration and tokenizer are prepared, use the resumable downloader to fetch the pinned weight file, verify its upstream LFS SHA-256 and create the launcher's required `download-manifest.json`:

```bash
python benchmark/qwen38_multigpu/download_draft.py
```

The downloader fetches the weight file only. Its expected size is **3,976,869,890 bytes**, SHA-256 `6cf6f33fbb2dfd7e74f8009132f0a2a2b4c71cd6cce9360a7d4145c9218f1781`.

| Variable | Purpose | Default |
| --- | --- | --- |
| `DSPARK_PYTHON` | Server interpreter | Current Python |
| `DSPARK_MODEL` | Target model directory | `~/models/Qwen3.8-27B` |
| `DSPARK_DRAFT_MODEL` | Draft directory | `~/models/Qwen3.8-27B-speculator.dspark` |
| `DSPARK_WORKLOAD` | Frozen performance workload | Committed result directory's `workload.json` |
| `DSPARK_QUALITY_CASES` | EOS / stop check cases | Committed result directory's `quality-cases.json` |
| `DSPARK_RUNTIME` | Optional separately installed package directory | Packages in the current interpreter |
| `DSPARK_NSYS` | Nsight Systems executable | `nsys` on PATH |

The measured physical GPUs were 1 / 5 / 6 / 7, with PCIe PHB / PIX topology and no NVLink. Choose four cards available to your experiment. Launchers check free memory and actual compute processes, lock the experiment and stop only server process groups they created. The draft has 20 attention heads; this model pair was measured at TP4, with no TP8 results. Record changed topologies and cache budgets as separate experiments.

## Reproduction

Run these commands from the repository root with a new output filename for each experiment. Servers bind locally; diagnostic RPC is used for local experiments.

### 1. Native DSpark / AR integration benchmark

Five paired rounds at concurrency 1 / 4 / 16, a 4.5 GiB cache budget per GPU, ~2K input and 256 actual output tokens; thinking and prefix caching are disabled.

```bash
python benchmark/qwen38_multigpu/run.py \
  --gpus "$DSPARK_GPUS" --output outputs/qwen38-native.json \
  --modes ar,fixed --rounds 5 --concurrency 1,4,16 \
  --requests 4 --tokens 256 --kv-gib 4.5 --sample-offset 16
python benchmark/qwen38_multigpu/summarize.py outputs/qwen38-native.json
```

### 2. Calibrate and freeze the verification policy

Calibrate K=1/2/4/7 on the first 16 prompts, then evaluate five held-out ablation rounds on the remaining 16. The committed [policy](../../configs/qwen38_uniform_budget.json) is already frozen; do not fit it on validation responses.

```bash
python benchmark/qwen38_multigpu/check_graphs.py
python benchmark/qwen38_multigpu/calibrate.py \
  --gpus "$DSPARK_GPUS" --output outputs/qwen38-budget.json
python benchmark/qwen38_multigpu/summarize_budget.py outputs/qwen38-budget.json
```

### 3. Direct comparison at concurrency 16

Each round starts fresh AR and DSpark engines. The latter randomly compares original verification / original buckets, fine graph buckets, and budget + fine buckets in a common 45-graph pool. Loading, warm-up, capture and additional diagnostics are excluded from timing. `--diagnostics` also collects separate greedy target-decision traces.

```bash
python benchmark/qwen38_multigpu/threeway.py \
  --gpus "$DSPARK_GPUS" --rounds 5 --diagnostics \
  --profile configs/qwen38_uniform_budget.json \
  --output outputs/qwen38-threeway.json
python benchmark/qwen38_multigpu/analyze_closure.py outputs/qwen38-threeway.json
```

### 4. Independent capacity and numerical diagnostics

Record actual allocation returns and GDN / attention / draft state requirements, with extra AR serial / concurrent controls. Instrumented throughput is excluded from the performance conclusions.

```bash
python benchmark/qwen38_multigpu/capacity_probe.py \
  --gpus "$DSPARK_GPUS" --output outputs/qwen38-capacity.json
python benchmark/qwen38_multigpu/analyze_capacity.py outputs/qwen38-capacity.json
```

### 5. Separate Nsight capture

`capture_profile.py` and `phases.py` mark target / draft phases; `analyze_profile.py` reads exported SQLite records. Profiling runs are separate from the timed comparisons.

```bash
python benchmark/qwen38_multigpu/capture_profile.py \
  --gpus "$DSPARK_GPUS" --output outputs/qwen38-profiles.json
python benchmark/qwen38_multigpu/analyze_profile.py --help
```

## Verify Recorded Results Without a GPU

This path needs only the Python standard library:

```bash
python3 benchmark/qwen38_multigpu/reproduce_results.py \
  --output outputs/qwen38-evidence
```

The script verifies 27 gzip archives and their decompressed SHA-256 hashes, relocates bundled file references, recomputes four summaries and requires exact agreement with the committed summaries except for source paths. JSONL bytes stay unchanged, preserving capacity-window offsets. It also checks 13,056 emitted target decisions and four-TP-rank agreement. Use a new, empty output directory.

The [result bundle](../../results/qwen38_multigpu_20261003) includes raw requests / traces, summaries, a frozen policy and the complete public ShareGPT tokenized workload. Its [manifest](../../results/qwen38_multigpu_20261003/bundle-manifest.json) records original experiment source hashes separately from published portability adaptations. Model weights, installed runtimes, service logs and large Nsight databases are not bundled.

## Metrics and Boundaries

- **Throughput:** actual output tokens / finite request-group completion time. These are closed-loop request groups, not steady-state production QPS. SSE chunks are not counted as tokens.
- **Client TPOT:** first-to-last nonempty SSE interval / (actual output tokens − 1). It includes scheduling and streamed delivery, rather than pure GPU step time.
- **Statistics:** median of five within-round ratios. Different experiments and medians of component ratios cannot be multiplied. The native integration comparison uses independent deployments; DSpark ablations use a common graph pool.
- **High concurrency:** the combined c=16 policy reaches 0.995× AR throughput, ranging from 0.974× to 1.016×. It reaches near parity, without a stable throughput speedup. At c=4, the held-out increment is +6.3%; at c=16, the direct increment over the original policy is +7.8%.
- **Ablation scope:** c=16 graph-only / original policy is +0.9%; budget + graphs / graph-only is +8.2%. The latter is conditional on the graph adaptation. Graph-only padding is 1.09; combined padding is 1.10.
- **State capacity:** maximum K=7 still reserves 8 GDN state blocks. A shorter submitted verification prefix does not reduce this preallocated state capacity. Real allocation records confirm admission queueing, without assigning all throughput loss to this cause.
- **Greedy validation:** 13,056 checked decisions match each path's target argmax. Long sequences still differ; checked first divergences correspond to BF16 target-logit ordering changes. Universal token-for-token equality and losslessness over all inputs have not been established.
- **Routing scope:** the prefix adapter chooses a verification budget inside a DSpark engine. The separate AR / speculative routing study is a different module; automatic Qwen3.8 AR fallback is not integrated or validated.
