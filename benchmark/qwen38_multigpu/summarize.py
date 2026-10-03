"""Keep paired ratios, per-round variation and token equivalence explicit."""
import argparse
import json
import re
import statistics
from pathlib import Path


def counter(text, name):
    return sum(float(m.group(1)) for m in re.finditer(
        r"^" + re.escape(name) + r"\{[^\n]*\}\s+([-+0-9.eE]+)$", text, re.M))


def acceptance(row):
    before, after = row["metrics_before"], row["metrics_after"]
    names = {"rounds": "vllm:spec_decode_num_drafts_total",
             "accepted": "vllm:spec_decode_num_accepted_tokens_total",
             "proposed": "vllm:spec_decode_num_draft_tokens_total",
             "preemptions": "vllm:num_preemptions_total"}
    delta = {k: counter(after, v) - counter(before, v) for k, v in names.items()}
    delta["accepted_length_including_bonus"] = 1 + delta["accepted"] / delta["rounds"] if delta["rounds"] else None
    delta["draft_acceptance_fraction"] = delta["accepted"] / delta["proposed"] if delta["proposed"] else None
    return delta


def summarize(path):
    x = json.loads(Path(path).read_text())
    pairs = []
    lookup = {(r["round"], r["mode"], r["concurrency"]): r for r in x["rows"]}
    for (rnd, mode, c), candidate in lookup.items():
        if mode != "fixed" or (rnd, "ar", c) not in lookup:
            continue
        base = lookup[rnd, "ar", c]
        if base["errors"] or candidate["errors"]:
            raise RuntimeError("Cannot summarize failed measurements")
        b, s = base["summary"], candidate["summary"]
        by_idx = {r["index"]: r for r in base["rows"]}
        same = 0
        divergent = []
        for r in candidate["rows"]:
            reference = by_idx[r["index"]]
            if r["prompt_sha256"] != reference["prompt_sha256"] or r["output_tokens"] != reference["output_tokens"]:
                raise RuntimeError("Paired input/output budgets differ")
            if r["token_ids"] == reference["token_ids"]:
                same += 1
            else:
                divergent.append(r["index"])
        pairs.append(dict(round=rnd, concurrency=c, requests=base["requests"],
                          ar_tokens_s=b["tokens_per_second"], spec_tokens_s=s["tokens_per_second"],
                          throughput_ratio=s["tokens_per_second"] / b["tokens_per_second"],
                          tpot_reduction_fraction=1-s["tpot_ms_median"] / b["tpot_ms_median"],
                          ar_tpot_ms=b["tpot_ms_median"], spec_tpot_ms=s["tpot_ms_median"],
                          ar_ttft_ms=b["ttft_ms_median"], spec_ttft_ms=s["ttft_ms_median"],
                          identical_sequences=same, total_sequences=len(candidate["rows"]), divergent_indices=divergent,
                          acceptance=acceptance(candidate)))
    summary = []
    for c in sorted({r["concurrency"] for r in pairs}):
        group = [r for r in pairs if r["concurrency"] == c]
        row = dict(concurrency=c, rounds=len(group),
                   throughput_ratio=statistics.median(r["throughput_ratio"] for r in group),
                   throughput_ratio_range=[min(r["throughput_ratio"] for r in group), max(r["throughput_ratio"] for r in group)],
                   tpot_reduction_fraction=statistics.median(r["tpot_reduction_fraction"] for r in group),
                   identical_sequences=sum(r["identical_sequences"] for r in group),
                   total_sequences=sum(r["total_sequences"] for r in group),
                   ar_tokens_s=statistics.median(r["ar_tokens_s"] for r in group),
                   spec_tokens_s=statistics.median(r["spec_tokens_s"] for r in group),
                   ar_tpot_ms=statistics.median(r["ar_tpot_ms"] for r in group),
                   spec_tpot_ms=statistics.median(r["spec_tpot_ms"] for r in group),
                   accepted_length=statistics.median(r["acceptance"]["accepted_length_including_bonus"] for r in group),
                   preemptions=sum(r["acceptance"]["preemptions"] for r in group))
        summary.append(row)
    output = dict(source=str(Path(path).resolve()), complete=x["complete"], pairs=pairs, summary=summary,
                  metric="median of within-round ratios, not ratio of marginal medians",
                  correctness_scope="Observed greedy sequence equality; repeated requests/rounds are not independent tasks")
    target = Path(path).with_suffix(".summary.json")
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("result", type=Path)
    summarize(p.parse_args().result)
