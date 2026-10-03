"""Summarize direct ratios and audit the first greedy decision divergence."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def decisions(data):
    entries = {}
    coverage = []
    for run in data["diagnostics"]:
        api = {}
        for row in run["rows"]:
            for request in row["rows"]:
                api[request["api_request_id"]] = (row["concurrency"], row["repeat"], request)
        captured = defaultdict(list)
        trace_metadata = run.get('traces', {})
        paths = [Path(r['output']) for r in trace_metadata.get('results', [])]
        if not paths:
            prefix = Path(data["output_path"]).with_suffix(f".diagnostic.{run['mode']}")
            paths = list(prefix.parent.glob(prefix.name+".rank*.jsonl"))
        for path in paths:
            for line in path.read_text().splitlines():
                trace = json.loads(line)
                ids = [name for name in api if trace["request_id"].startswith(name+"-")]
                if len(ids) != 1:
                    raise RuntimeError(f"Cannot map worker request {trace['request_id']}")
                name = ids[0]
                c, repeat, req = api[name]
                for token in trace["emitted"]:
                    index = token["output_index"]
                    if not 0 <= index < len(req["token_ids"]):
                        continue  # The engine may sample beyond the HTTP max_tokens cap.
                    if token["token"] != req["token_ids"][index]:
                        raise RuntimeError("Captured target decision does not match the HTTP token")
                    captured[name,index].append({**token, "rank":trace["rank"],
                                                 "sampler":trace["sampler"], "num_logits":trace["num_logits"],
                                                 "num_reqs":trace["num_reqs"], "num_tokens":trace["num_tokens"],
                                                 "captured_tokens":trace["captured_tokens"], "dtype":trace["logits_dtype"]})
        non_argmax = non_maximum = ties = checked = 0
        for name, (c, repeat, req) in api.items():
            tokens = []
            for i in range(len(req["token_ids"])):
                candidates = captured[name,i]
                if not candidates:
                    raise RuntimeError(f"No target decision for {name} token {i}")
                # Replicated TP ranks must agree, rather than being counted as samples.
                ref = {k:v for k,v in candidates[0].items() if k != "rank"}
                if any({k:v for k,v in t.items() if k != "rank"} != ref for t in candidates):
                    raise RuntimeError("TP target decisions disagree")
                token = candidates[0]
                checked += 1
                non_argmax += token["token"] != token["argmax"]
                logits = dict(zip(token["top_tokens"],token["top_logits"]))
                non_maximum += logits.get(token["token"],float("-inf")) != max(token["top_logits"])
                ties += token["top_logits"][0] == token["top_logits"][1]
                tokens.append(token)
            entries[run["mode"],c,repeat,req["index"]] = dict(request=req, decisions=tokens)
        coverage.append(dict(mode=run["mode"], requests=len(api), checked_tokens=checked,
                             non_argmax_tokens=non_argmax, non_maximum_tokens=non_maximum,
                             top_logit_ties=ties))
    return entries, coverage


def compare(entries, left, right, name):
    pairs = []
    keys = sorted(k for k in entries if k[:3] == left)
    for key in keys:
        other = (*right,key[3])
        if other not in entries:
            continue
        a,b = entries[key],entries[other]
        assert a["request"]["prompt_sha256"] == b["request"]["prompt_sha256"]
        ta,tb = a["request"]["token_ids"],b["request"]["token_ids"]
        first = next((i for i,(x,y) in enumerate(zip(ta,tb)) if x != y),None)
        pair = dict(index=key[3], identical=ta==tb, tokens=len(ta), first_difference_index=first)
        if first is not None:
            da,db = a["decisions"][first],b["decisions"][first]
            va,vb = dict(zip(da["top_tokens"],da["top_logits"])),dict(zip(db["top_tokens"],db["top_logits"]))
            union = sorted(set(va)&set(vb))
            pair.update(left=da,right=db,common_prefix_tokens=first,
                        maximum_shared_topk_logit_change=max(abs(va[t]-vb[t]) for t in union) if union else None,
                        left_margin=da["top_logits"][0]-da["top_logits"][1],
                        right_margin=db["top_logits"][0]-db["top_logits"][1],
                        changed_argmax=da["argmax"]!=db["argmax"],
                        each_emitted_token_is_own_argmax=(da["token"]==da["argmax"] and db["token"]==db["argmax"]))
        pairs.append(pair)
    return dict(comparison=name, requests=len(pairs), identical_sequences=sum(p["identical"] for p in pairs), pairs=pairs)


def analyze(path):
    data = json.loads(path.read_text())
    data["output_path"] = str(path.resolve())
    if not data["complete"] or data.get("failure"):
        raise RuntimeError("Only completed measurements may produce final conclusions")
    lookup = {(row["round"],row["mode"]):row for row in data["rows"]}
    comparisons = []
    for candidate,base in [("native","ar"),("graphs","ar"),("budget","ar"),
                            ("graphs","native"),("budget","graphs"),("budget","native")]:
        rows = []
        for rnd in range(data["settings"]["rounds"]):
            a,b = lookup[rnd,candidate],lookup[rnd,base]
            if a["errors"] or b["errors"]:
                raise RuntimeError("Errors in a paired comparison")
            if (a["workload_sha256"],a["sample_offset"],a["requests"],a["output_tokens"]) != (
                b["workload_sha256"],b["sample_offset"],b["requests"],b["output_tokens"]):
                raise RuntimeError("Workload mismatch")
            rows.append(dict(round=rnd, throughput_ratio=a["summary"]["tokens_per_second"]/b["summary"]["tokens_per_second"],
                             tpot_reduction=1-a["summary"]["tpot_ms_median"]/b["summary"]["tpot_ms_median"]))
        comparisons.append(dict(comparison=f"{candidate}/{base}",rounds=len(rows),pairs=rows,
                                throughput_ratio=statistics.median(r["throughput_ratio"] for r in rows),
                                throughput_ratio_range=[min(r["throughput_ratio"] for r in rows),max(r["throughput_ratio"] for r in rows)],
                                tpot_reduction=statistics.median(r["tpot_reduction"] for r in rows)))
    entries,coverage = decisions(data)
    diag = []
    for c in [1,16]:
        for a,b in [("ar","native"),("ar","budget"),("native","budget")]:
            diag.append(compare(entries,(a,c,0),(b,c,0),f"{a}/{b}, c={c}"))
    for mode in ["ar","native","budget"]:
        diag.append(compare(entries,(mode,1,0),(mode,1,1),f"{mode} isolated repeat"))
        diag.append(compare(entries,(mode,1,0),(mode,16,0),f"{mode} batch-size control"))
    modes=[]
    for mode in ["ar","native","graphs","budget"]:
        rows=[r for r in data["rows"] if r["mode"]==mode]
        item=dict(mode=mode,tokens_per_second=statistics.median(r["summary"]["tokens_per_second"] for r in rows),
                  tpot_ms=statistics.median(r["summary"]["tpot_ms_median"] for r in rows))
        if mode != "ar":
            item["target_padding_ratio"] = statistics.median(r["graph_summary"]["target"]["padding_ratio"] for r in rows)
        modes.append(item)
    output=dict(source=str(path.resolve()),complete=True,settings=data["settings"],modes=modes,
                comparisons=comparisons,decision_coverage=coverage,diagnostic_comparisons=diag,
                scope="Direct same-round c=16 comparisons; target logits captured in separate instrumented requests. No claim of universal losslessness or full kernel root-cause isolation.")
    target=path.with_suffix(".summary.json")
    target.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(dict(comparisons=[{k:v for k,v in r.items() if k!='pairs'} for r in comparisons],
                         decision_coverage=coverage,
                         diagnostic_summary=[{k:v for k,v in r.items() if k!='pairs'} for r in diag]),indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("result",type=Path)
    analyze(parser.parse_args().result)
