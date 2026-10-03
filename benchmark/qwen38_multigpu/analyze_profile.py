"""Associate kernels with opt-in CPU NVTX ranges through CUDA launch IDs.

Kernel durations are cumulative, including overlap and NCCL waiting. They are
not request latency shares, transferred bytes, or hardware bandwidth counters.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
import json
from pathlib import Path
import sqlite3


def category(name):
    n = name.lower()
    if any(x in n for x in ["nccl", "allreduce", "all_reduce", "reduce_scatter"]):
        return "communication"
    # Attention names contain cutlass scalar types; test the operator first.
    if any(x in n for x in ["flash", "attention", "paged_attn"]):
        return "attention"
    if any(x in n for x in ["gemm", "gemv", "xmma", "cutlass"]):
        return "matrix_multiply"
    if any(x in n for x in ["gdn_", "gated_delta", "delta_rule", "fused_recurrent"]):
        return "gdn"
    if "causal_conv" in n:
        return "convolution"
    if any(x in n for x in ["norm", "silu", "act_and_mul", "sigmoid"]):
        return "norm_or_activation"
    return "other"


def analyze(path):
    con = sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro", uri=True)
    ranges = defaultdict(list)
    for tid, start, end, name in con.execute("""SELECT n.globalTid,n.start,n.end,coalesce(s.value,n.text)
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
        WHERE n.end IS NOT NULL AND coalesce(s.value,n.text) LIKE 'dspark_lab/%'"""):
        ranges[tid].append((start, end, name))
    lookup = {}
    for tid, rows in ranges.items():
        rows.sort()
        lookup[tid] = (rows, [r[0] for r in rows], max(e-s for s,e,_ in rows))
    launches = {}
    for tid, correlation, start in con.execute("SELECT globalTid,correlationId,start FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
        if tid not in lookup:
            continue
        rows, starts, longest = lookup[tid]
        enclosed = [r for r in rows[bisect_left(starts,start-longest):bisect_right(starts,start)] if r[1] >= start]
        if enclosed:
            chosen = min(enclosed, key=lambda r:r[1]-r[0])
            launches[tid & 0xFFFFFFFFFF000000, correlation] = chosen[2]
    devices = defaultdict(lambda: defaultdict(lambda:dict(kernel_count=0, ns=0, categories=defaultdict(int), kernels=defaultdict(lambda:[0,0]))))
    count=0
    for device,pid,correlation,start,end,name in con.execute("""SELECT k.deviceId,k.globalPid,k.correlationId,k.start,k.end,s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id"""):
        phase = launches.get((pid,correlation), "unscoped")
        key = "target_graph" if "/target_graph/" in phase else "draft_graph" if "/draft_graph/" in phase else "draft_other" if phase.endswith("/draft_total") else phase
        row = devices[device][key]
        row["kernel_count"] += 1; row["ns"] += end-start
        row["categories"][category(name)] += end-start
        row["kernels"][name][0] += end-start; row["kernels"][name][1] += 1
        count+=1
    if not count:
        raise RuntimeError("No captured GPU kernels")
    output = dict(source=str(Path(path).resolve()), devices={}, scoped_launches=len(launches),
                  scope="CUDA runtime launch correlation with the innermost local NVTX phase; cumulative kernel time includes overlap and device waiting")
    for device, phases in devices.items():
        output["devices"][device]={}
        for phase, r in phases.items():
            output["devices"][device][phase] = dict(kernel_count=r["kernel_count"],cumulative_kernel_ms=r["ns"]/1e6,
                categories={k:dict(cumulative_ms=v/1e6, share=v/r["ns"]) for k,v in sorted(r["categories"].items(),key=lambda x:-x[1])},
                kernels=[dict(name=k,cumulative_ms=v[0]/1e6,calls=v[1],average_us=v[0]/v[1]/1e3)
                         for k,v in sorted(r["kernels"].items(),key=lambda x:-x[1][0])[:20]])
    return output


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("sqlite",type=Path);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=analyze(args.sqlite)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({d:{k:{a:b for a,b in v.items() if a!="kernels"} for k,v in phases.items()} for d,phases in result["devices"].items()},indent=2))
