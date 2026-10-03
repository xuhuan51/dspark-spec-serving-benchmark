"""Separate graph-bucket and verification-prefix gains on held-out inputs."""
import argparse
import json
from pathlib import Path
import statistics


def graphs(row):
    workers = row["graph_metrics"]
    if isinstance(workers, dict) and "results" in workers:
        workers = workers["results"]
    if isinstance(workers, dict):
        workers = [workers]
    first = workers[0]
    if any(worker["dispatch_counts"] != first["dispatch_counts"] for worker in workers[1:]):
        raise RuntimeError("TP ranks dispatched different shapes")
    result = {}
    for role in ["target", "draft"]:
        counts = [r for r in first["dispatch_counts"] if r[0] == role and r[1] == "FULL"]
        # Padding ratio is a shape count, not a measurement of FLOPs or GPU time.
        logical = sum(r[3] * r[2] * r[6] for r in counts)
        captured = sum(r[5] * r[6] for r in counts)
        result[role] = dict(calls=sum(r[6] for r in counts), logical_tokens=logical,
                            captured_tokens=captured,
                            padding_ratio=captured / logical if logical else None)
    return result


def summary(path):
    path = Path(path)
    data = json.loads(path.read_text())
    lookup = {(r["round"], r["concurrency"], r["mode"]): r for r in data["validation"]}
    pairs = []
    for (rnd, c, mode), candidate in lookup.items():
        for baseline in ["native", "graphs"]:
            if mode == baseline or mode == "native" or (mode == "graphs" and baseline != "native"):
                continue
            if (rnd, c, baseline) not in lookup:
                continue
            base = lookup[rnd, c, baseline]
            if base["errors"] or candidate["errors"]:
                raise RuntimeError("Failed measurement cannot enter performance summary")
            same = 0
            base_requests = {r["index"]: r for r in base["rows"]}
            for r in candidate["rows"]:
                ref = base_requests[r["index"]]
                if (r["prompt_sha256"], r["output_tokens"]) != (ref["prompt_sha256"], ref["output_tokens"]):
                    raise RuntimeError("Paired workloads differ")
                same += r["token_ids"] == ref["token_ids"]
            b, s = base["summary"], candidate["summary"]
            pairs.append(dict(round=rnd, concurrency=c, comparison=f"{mode}/{baseline}",
                              throughput_ratio=s["tokens_per_second"] / b["tokens_per_second"],
                              tpot_reduction_fraction=1-s["tpot_ms_median"] / b["tpot_ms_median"],
                              identical_sequences=same, total_sequences=len(candidate["rows"])))
    comparisons = []
    for c, name in sorted({(r["concurrency"], r["comparison"]) for r in pairs}):
        group = [r for r in pairs if (r["concurrency"], r["comparison"]) == (c, name)]
        ratios = [r["throughput_ratio"] for r in group]
        comparisons.append(dict(concurrency=c, comparison=name, rounds=len(group),
                                throughput_ratio=statistics.median(ratios),
                                throughput_ratio_range=[min(ratios), max(ratios)],
                                tpot_reduction_fraction=statistics.median(r["tpot_reduction_fraction"] for r in group),
                                identical_sequences=sum(r["identical_sequences"] for r in group),
                                total_sequences=sum(r["total_sequences"] for r in group)))
    conditions = []
    for c, mode in sorted({(r["concurrency"], r["mode"]) for r in data["validation"]}):
        rows = [r for r in data["validation"] if (r["concurrency"], r["mode"]) == (c, mode)]
        measures = [graphs(r) for r in rows]
        conditions.append(dict(concurrency=c, mode=mode, rounds=len(rows),
                               tokens_per_second=statistics.median(r["summary"]["tokens_per_second"] for r in rows),
                               tpot_ms=statistics.median(r["summary"]["tpot_ms_median"] for r in rows),
                               accepted_length=statistics.median(r["acceptance"]["accepted_length_including_bonus"] for r in rows),
                               dominant_active_batch=[r["dominant_active_batch"] for r in rows],
                               graphs={role: {key: statistics.median(g[role][key] for g in measures)
                                              for key in ["calls", "logical_tokens", "captured_tokens", "padding_ratio"]}
                                       for role in ["target", "draft"]}))
    output = dict(source=str(path.resolve()), complete=data["complete"],
                  profile=data.get("profile"), pairs=pairs, comparisons=comparisons,
                  conditions=conditions,
                  quality={k: data[k] for k in ["quality_native", "quality_budget"] if k in data},
                  scope="within-round ratios on held-out prompts; native uses the common graph pool with original buckets")
    path.with_suffix(".summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    summary(parser.parse_args().result)
