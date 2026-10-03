"""Separate Nsight capture and quality work from unprofiled performance A/B.

Named explicitly to avoid shadowing Python's standard-library profile module.
"""
import argparse
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import time
import urllib.request

from benchmark import measure
from calibrate import rpc
from quality import check, score_sequences
from run import HERE, WORKLOAD, check_idle, command, gpu_snapshot, metrics, runtime_environment, stop, wait_ready

NSYS = os.environ.get("DSPARK_NSYS", shutil.which("nsys") or "nsys")


def post(base, endpoint):
    with urllib.request.urlopen(urllib.request.Request(base+endpoint, data=b"", method="POST"), timeout=120) as r:
        return r.status


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpus", default="1,5,6,7")
    p.add_argument("--port", type=int, default=18876)
    p.add_argument("--k", type=int, default=7)
    p.add_argument("--kv-gib", type=float, default=4.5)
    p.add_argument("--modes", default="ar,fixed")
    p.add_argument("--score", type=Path, action="append", default=[])
    args = p.parse_args(); args.gpus=list(map(int,args.gpus.split(","))); args.eager=False
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock=(HERE/".experiment.lock").open("a"); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=check_idle(args.gpus)
    sources=args.output.with_suffix(".sources"); sources.mkdir()
    for f in HERE.glob("*.py"):
        (sources/f.name).write_bytes(f.read_bytes())
    env=runtime_environment()
    env.update(CUDA_VISIBLE_DEVICES=",".join(before[i]["uuid"] for i in args.gpus),
               VLLM_WORKER_MULTIPROC_METHOD="spawn", OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
               VLLM_USE_FLASHINFER_SAMPLER="0", VLLM_NVTX_SCOPES_FOR_PROFILING="1", VLLM_SERVER_DEV_MODE="1")
    result=dict(gpu_before=before, runs=[], complete=False,
                scope="Nsight captures contain prefill and decode; profiled throughput is not performance evidence",
                source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in HERE.glob("*.py")})
    def save():
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    proc=None; base=f"http://127.0.0.1:{args.port}"
    try:
        for mode in args.modes.split(","):
            check_idle(args.gpus)
            server=command(mode,args)+["--profiler-config",'{"profiler":"cuda"}',
                                       "--worker-extension-cls","phases.PhaseWorker"]
            prefix=args.output.with_suffix("."+mode)
            cmd=[NSYS,"profile","--sample=none","--cpuctxsw=none","--trace=cuda,nvtx,osrt",
                 "--trace-fork-before-exec=true","--cuda-graph-trace=node",
                 "--capture-range=cudaProfilerApi","--capture-range-end=repeat",
                 "-o",str(prefix.resolve())]+server
            with args.output.with_suffix("."+mode+".log").open("w") as stream:
                proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env,cwd=HERE,start_new_session=True)
                entry=dict(mode=mode,pid=proc.pid,command=cmd,captures=[]); result["runs"].append(entry); save()
                print(f"START profile {mode} pid={proc.pid}",flush=True)
                wait_ready(proc,base,timeout=1200)
                entry["quality"]=asyncio.run(check(base));save()
                if entry["quality"]["errors"]:
                    raise RuntimeError("Quality requests failed")
                if mode=="ar" and args.score:
                    entry["reference_scores"]=asyncio.run(score_sequences(base,args.score));save()
                entry["phase_install"]=rpc(base,"lab_install_phases");save()
                for c in [1,16]:
                    warm=asyncio.run(measure(base,WORKLOAD,concurrency=c,requests=c,output_tokens=32,sample_offset=16))
                    if warm["errors"]:
                        raise RuntimeError("Profile warmup failed")
                    pre=metrics(base)
                    post(base,"/start_profile")
                    try:
                        row=asyncio.run(measure(base,WORKLOAD,concurrency=c,requests=c,output_tokens=64,sample_offset=16))
                    finally:
                        post(base,"/stop_profile")
                    row.update(metrics_before=pre,metrics_after=metrics(base))
                    entry["captures"].append(row);save()
                    if row["errors"]:
                        raise RuntimeError("Capture request failed")
                    print(f"CAPTURED {mode} c={c}",flush=True)
                stop(proc);proc=None
            for _ in range(30):
                try:check_idle(args.gpus);break
                except RuntimeError:time.sleep(1)
            check_idle(args.gpus)
        result["complete"]=True
    except Exception as exc:
        result["failure"]=repr(exc);raise
    finally:
        stop(proc);result["gpu_after"]=gpu_snapshot(args.gpus);save()


if __name__=="__main__":
    main()
