# DSpark / DeepSpec Reproduction Report

## Goal

Reproduce the DeepSpec acceptance-length benchmark for DSpark, EAGLE3, and
DFlash on Qwen3-8B, then use the result as the algorithm-side baseline for
serving experiments.

Acceptance length is the average number of tokens accepted by the target model
per verification round. It is an algorithm metric, not an end-to-end serving
speedup metric.

## Setup

- Target model: `Qwen3-8B`
- Draft models:
  - `dspark_qwen3_8b_block7`
  - `eagle3_qwen3_8b_ttt7`
  - `dflash_qwen3_8b_block7`
- Evaluator: official DeepSpec `eval.py`
- Hardware: 8x NVIDIA A30 24GB
- Runtime note: ultra-long prompts were filtered to avoid A30 24GB OOM. The
  filtered set did not affect the main reproduced DSpark-vs-baseline conclusion.

## Results

| Dataset | DSpark | EAGLE3 | DFlash |
| --- | ---: | ---: | ---: |
| gsm8k | 6.15 | 5.27 | 5.34 |
| math500 | 5.80 | 4.77 | 4.91 |
| aime25 | 5.02 | 4.10 | 4.03 |
| humaneval | 5.53 | 4.35 | 4.65 |
| mbpp | 5.17 | 3.93 | 4.38 |
| livecodebench | 5.18 | 4.17 | 4.43 |
| mt-bench | 3.72 | 2.67 | 3.11 |
| alpaca | 3.60 | 2.53 | 3.01 |

Macro average over the eight completed datasets:

| Comparison | Reproduced Result | Paper Result |
| --- | ---: | ---: |
| DSpark vs EAGLE3 | +26.4% | +26.7% |
| DSpark vs DFlash | +18.6% | +18.4% |

## Interpretation

The reproduced numbers validate the algorithm-side claim: DSpark improves
accepted length over EAGLE3 and DFlash on the Qwen3-8B benchmark suite.

However, accepted length alone does not determine production serving speedup.
Serving speedup also depends on:

- draft-model cost,
- target verification cost,
- tensor-parallel communication,
- batch scheduling,
- KV-cache pressure,
- quantization,
- workload-domain match,
- concurrency.

This is why the project continues with OpenAI-compatible serving experiments
instead of stopping at the DeepSpec reproduction.

## Reproduction Command

```bash
bash scripts/run_deepspec_eval_qwen3_8b.sh
```

The cleaned table is stored in:

```text
results/deepspec_qwen3_8b_acceptance.csv
```
