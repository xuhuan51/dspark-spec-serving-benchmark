# DSpark Cross-Target Stress Test on Qwen3-32B

Date: 2026-08-05

## Scope

This experiment extends the native vLLM 0.26.0 DSpark serving validation to
Qwen3-32B BF16/TP8 and Qwen3-32B INT4/TP4.

The evaluated DeepSpec release set did not provide a matched Qwen3-32B DSpark
checkpoint. Qwen3-14B and Qwen3-32B share hidden size 5120 and vocabulary size
151936, so vLLM can technically connect the Qwen3-14B DSpark checkpoint to the
32B target. This setup is intentionally a cross-target mismatch stress test.
It must not be interpreted as the performance of a trained Qwen3-32B DSpark
drafter.

## Environment

- GPU: 8x NVIDIA A30 24 GiB
- Runtime: vLLM 0.26.0, torch 2.11.0+cu130
- Target A: Qwen3-32B BF16, TP8
- Target B: Qwen3-32B-W4A16-AWQ, TP4
- Draft: `dspark_qwen3_14b_block7`, BF16
- Draft TP: matched to target TP
- Speculative tokens: 7
- Context limit: 2048
- Output shape: fixed 256 tokens, `temperature=0`, `ignore_eos=true`
- Concurrency: 1, 2, 4, 8, 16
- Server `max_num_seqs`: 16

Baseline and DSpark measurements used the same target configuration, request
shape, vLLM version, and GPU budget at each precision/parallelism point.

## Qwen3-32B BF16, TP8

| Concurrency | Baseline tok/s | DSpark tok/s | Speedup | Baseline TPOT | DSpark TPOT | Accepted length | Accept rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 49.379 | 57.629 | 1.167x | 20.203 ms | 17.272 ms | 1.483 | 6.89% |
| 2 | 95.114 | 106.328 | 1.118x | 20.864 ms | 18.624 ms | 1.529 | 7.56% |
| 4 | 181.424 | 155.045 | 0.855x | 21.859 ms | 25.568 ms | 1.519 | 7.41% |
| 8 | 350.866 | 203.513 | 0.580x | 22.582 ms | 38.816 ms | 1.525 | 7.51% |
| 16 | 650.999 | 304.995 | 0.469x | 24.263 ms | 51.609 ms | 1.522 | 7.45% |

The expensive BF16 target leaves enough decode headroom for modest gains at
`c=1-2`. At `c=4`, the draft and verification overhead exceeds the target
decode work saved, and the gap grows as batching and TP communication increase.

## Qwen3-32B INT4, TP4

| Concurrency | Baseline tok/s | DSpark tok/s | Speedup | Baseline TPOT | DSpark TPOT | Accepted length | Accept rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 74.603 | 74.292 | 0.996x | 13.357 ms | 13.374 ms | 1.409 | 5.84% |
| 2 | 140.308 | 126.634 | 0.903x | 14.142 ms | 15.425 ms | 1.519 | 7.41% |
| 4 | 273.463 | 172.297 | 0.630x | 14.497 ms | 22.713 ms | 1.465 | 6.65% |
| 8 | 515.813 | 230.487 | 0.447x | 15.303 ms | 33.701 ms | 1.483 | 6.91% |
| 16 | 811.917 | 301.453 | 0.371x | 19.425 ms | 51.920 ms | 1.475 | 6.79% |

INT4 accelerates the normal target decode path, so there is less cost left for
speculative decoding to remove. With accepted length near 1.5, the cross-target
draft is already at parity at `c=1` and becomes strongly negative under
concurrency.

## Interpretation

This stress test isolates three practical routing signals:

- **Draft/target match:** runtime-compatible tensor shapes do not imply useful
  token predictions.
- **Quantization:** a faster baseline decode path raises the accepted-length
  threshold needed to amortize draft and verification work.
- **Concurrency:** batching and collective communication can move the
  profitable boundary even when accepted length remains stable.

The result reinforces the project's main systems conclusion: speculative
decoding should be enabled only inside a measured positive region. For this
cross-target pair, a conservative policy would permit BF16 only below
concurrency 4 and disable the INT4 path entirely.

## Correctness and Limitations

- A deterministic 96-token no-thinking check matched exactly between the INT4
  baseline and speculative endpoints.
- vLLM metrics confirmed that draft tokens were generated and verified.
- Accepted length is computed as `1 + accepted_draft_tokens / draft_rounds`;
  accept rate is `accepted_draft_tokens / proposed_draft_tokens`.
- This experiment evaluates a deliberately mismatched 14B drafter and does not
  replace training and testing a matched Qwen3-32B DSpark checkpoint.

Compact results are stored in
`results/vllm026_qwen3_32b_dspark_cross_target.csv`. Per-request traces and
metric snapshots remain under ignored local `outputs/qwen3_32b_*` directories.

## Reproduction

Run the BF16/TP8 baseline and DSpark endpoints sequentially on the same eight
GPUs:

```bash
GPUS=0,1,2,3,4,5,6,7 TP=8 DTYPE=bfloat16 GPU_MEM=0.75 \
MAX_NUM_SEQS=16 MAX_NUM_BATCHED_TOKENS=8192 PORT=8580 \
MODEL_PATH=/home/liuguangli/models/Qwen3-32B \
bash scripts/run_vllm_local_baseline.sh

GPUS=0,1,2,3,4,5,6,7 TP=8 DRAFT_TP=8 DTYPE=bfloat16 GPU_MEM=0.75 \
MAX_NUM_SEQS=16 MAX_NUM_BATCHED_TOKENS=8192 PORT=8581 \
MODEL_PATH=/home/liuguangli/models/Qwen3-32B \
DRAFT_MODEL_PATH=/home/liuguangli/models/dspark_qwen3_14b_block7 \
bash scripts/run_vllm_dspark.sh
```

For INT4/TP4, set the target path and use four GPUs per endpoint:

```bash
GPUS=0,1,2,3 TP=4 DTYPE=auto GPU_MEM=0.88 \
MAX_NUM_SEQS=16 MAX_NUM_BATCHED_TOKENS=8192 PORT=8570 \
MODEL_PATH=/home/liuguangli/models/Qwen3-32B-W4A16-AWQ \
bash scripts/run_vllm_local_baseline.sh

GPUS=4,5,6,7 TP=4 DRAFT_TP=4 DTYPE=auto GPU_MEM=0.88 \
MAX_NUM_SEQS=16 MAX_NUM_BATCHED_TOKENS=8192 PORT=8571 \
MODEL_PATH=/home/liuguangli/models/Qwen3-32B-W4A16-AWQ \
DRAFT_MODEL_PATH=/home/liuguangli/models/dspark_qwen3_14b_block7 \
bash scripts/run_vllm_dspark.sh
```
