# Native DSpark Serving Validation with vLLM 0.26.0

Date: 2026-08-05

## Scope

This validation closes the earlier serving gap in which the DeepSpec acceptance
study used DSpark, but the available OpenAI-compatible serving path used an
EAGLE3 checkpoint. vLLM 0.26.0 now directly recognizes the original
`Qwen3DSparkModel` checkpoint and serves it with the native `dspark` method.

## Environment

- GPU: NVIDIA A30 24 GiB
- Target: Qwen3-8B, BF16
- Draft: `dspark_qwen3_8b_block7`, BF16
- Runtime: vLLM 0.26.0, torch 2.11.0+cu130
- Context limit: 2048
- Output shape: fixed 256 tokens, `temperature=0`, `ignore_eos=true`
- Concurrency: 1, 2, 4, 8, 16
- Server `max_num_seqs`: 4 for c=1-4, 16 for c=8-16

The baseline and DSpark endpoints used separate A30 GPUs with the same vLLM
version, model dtype, request payload, and request-level generation parameters.
Each concurrency point used matched baseline and DSpark server budgets.

## Results

| Concurrency | Baseline tok/s | DSpark tok/s | Throughput speedup | Baseline TPOT | DSpark TPOT | Accepted length |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 47.349 | 97.006 | 2.049x | 21.090 ms | 10.147 ms | 2.633 |
| 2 | 94.848 | 187.196 | 1.974x | 20.987 ms | 10.521 ms | 2.576 |
| 4 | 187.186 | 353.056 | 1.886x | 21.203 ms | 11.091 ms | 2.566 |
| 8 | 366.852 | 655.033 | 1.786x | 21.627 ms | 12.017 ms | 2.591 |
| 16 | 716.766 | 1152.974 | 1.609x | 22.076 ms | 13.550 ms | 2.667 |

Native DSpark remained profitable throughout the tested range. The relative
gain narrowed as concurrency increased, while the measured accepted length
stayed near 2.6. This is direct evidence that accepted length and wall-clock
speedup are related but not interchangeable metrics.

Raw compact results are stored in
`results/vllm026_qwen3_8b_dspark_serving.csv`. Per-request and metric snapshots
are stored under the ignored local `outputs/vllm026_*` directories.

## Correctness Checks

- Service smoke tests returned valid Chinese, arithmetic, and Python outputs.
- vLLM counters confirmed that DSpark produced and accepted draft tokens.
- With Qwen3 thinking disabled, six short deterministic prompts matched the
  baseline output exactly.
- With thinking enabled, one of four 128-token traces diverged in wording while
  both endpoints remained individually deterministic and coherent.

The last observation is recorded as a numerical reproducibility caveat rather
than a semantic failure. On A30 (compute capability 8.0), vLLM's
batch-invariant mode is unavailable; baseline single-token decode and
multi-token verification may therefore take different BF16 numerical paths.
Production rollout should still include task-level accuracy tests in addition
to text equality checks.

## Reproduction

Start matched endpoints:

```bash
GPU=0 PORT=8550 MAX_NUM_SEQS=16 \
  bash scripts/run_vllm_local_baseline.sh

GPU=1 PORT=8551 MAX_NUM_SEQS=16 \
  bash scripts/run_vllm_dspark.sh
```

Run the decode-only ladder against each endpoint:

```bash
python3 benchmark/spec_decode_microbench.py \
  --tag qwen3_8b_baseline_vllm026 \
  --base-url http://127.0.0.1:8550/v1 \
  --model qwen \
  --out-dir outputs/vllm026_baseline \
  --concurrencies 1,2,4,8,16 \
  --max-tokens 256 \
  --ignore-eos

python3 benchmark/spec_decode_microbench.py \
  --tag qwen3_8b_dspark_vllm026 \
  --base-url http://127.0.0.1:8551/v1 \
  --model qwen \
  --out-dir outputs/vllm026_dspark \
  --concurrencies 1,2,4,8,16 \
  --max-tokens 256 \
  --ignore-eos
```

The separate Qwen3-32B experiment is documented as a cross-target mismatch
stress test in `reports/vllm026_dspark_32b_cross_target.md`; it is not a matched
32B DSpark result.
