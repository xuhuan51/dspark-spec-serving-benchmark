"""Separate AR/DSpark cache-allocation trace on the frozen c=16 workload."""
import argparse
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from benchmark import measure
from calibrate import rpc
from quality import check
from run import HERE, WORKLOAD, check_idle, command, gpu_snapshot, runtime_environment, stop, wait_ready


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="1,5,6,7")
    parser.add_argument("--port", type=int, default=18876)
    parser.add_argument("--kv-gib", type=float, default=4.5)
    parser.add_argument("--k", type=int, default=7)
    args = parser.parse_args()
    args.gpus = list(map(int,args.gpus.split(",")))
    args.eager = False
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = (HERE/".experiment.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    result = dict(complete=False, runs=[], gpu_before=check_idle(args.gpus),
                  scope="Actual per-group allocation decisions; diagnostics are not throughput measurements.",
                  workload_sha256=hashlib.sha256(WORKLOAD.read_bytes()).hexdigest(),
                  sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")})
    def save():
        temporary=args.output.with_name(args.output.name+'.tmp')
        temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
        temporary.replace(args.output)
    save()
    env = runtime_environment()
    env.update(VLLM_WORKER_MULTIPROC_METHOD="spawn", OMP_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false", VLLM_USE_FLASHINFER_SAMPLER="0",
               VLLM_NVTX_SCOPES_FOR_PROFILING="0", VLLM_SERVER_DEV_MODE="1")
    proc = None
    try:
        for mode in ["ar", "fixed"]:
            before = check_idle(args.gpus)
            env["CUDA_VISIBLE_DEVICES"] = ",".join(before[i]["uuid"] for i in args.gpus)
            trace = args.output.with_suffix(f".{mode}.jsonl")
            env["DSPARK_ADMISSION_TRACE"] = str(trace.resolve())
            cmd = command(mode,args)+["--scheduler-cls", "admission_audit.AdmissionAuditScheduler",
                                      "--worker-extension-cls", "decision_audit.DecisionAuditWorker"]
            entry = dict(mode=mode, command=cmd, trace=str(trace), gpu_before=before)
            result["runs"].append(entry)
            with args.output.with_suffix(f".{mode}.log").open("x") as stream:
                proc = subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,
                                        env=env,cwd=HERE,start_new_session=True)
                entry["pid"] = proc.pid
                save()
                print(f"START CAPACITY {mode} pid={proc.pid}",flush=True)
                wait_ready(proc,f"http://127.0.0.1:{args.port}")
                entry["trace_start"] = trace.stat().st_size
                decision_path = args.output.with_suffix(f".diagnostic.{mode}")
                entry["decision_traces"] = rpc(f"http://127.0.0.1:{args.port}", "decision_audit_start", str(decision_path.resolve()))
                entry["measurement"] = asyncio.run(measure(f"http://127.0.0.1:{args.port}",WORKLOAD,
                                                             concurrency=16,requests=16,output_tokens=128,
                                                             sample_offset=16))
                entry["trace_end"] = trace.stat().st_size
                if entry["measurement"]["errors"]:
                    raise RuntimeError("Capacity requests failed")
                if mode == "ar":
                    entry["serial_control"] = asyncio.run(measure(f"http://127.0.0.1:{args.port}",WORKLOAD,
                                                                  concurrency=1,requests=16,output_tokens=128,
                                                                  sample_offset=16))
                    if entry["serial_control"]["errors"]:
                        raise RuntimeError("Autoregressive batch-size control failed")
                entry["decisions_stopped"] = rpc(f"http://127.0.0.1:{args.port}", "decision_audit_stop")
                entry["quality"] = asyncio.run(check(f"http://127.0.0.1:{args.port}"))
                q=entry['quality']
                if (q['errors'] or q['natural_exact_match'] != q['natural_count']
                        or q['natural_ended'] != q['natural_count']
                        or any(not row.get('exact_match') or row.get('finish_reason') != 'stop' for row in q['rows'][-2:])):
                    raise RuntimeError("EOS/stop regression checks failed")
                save()
                print(f"CAPACITY COMPLETE {mode}",flush=True)
                stop(proc)
                proc = None
            for _ in range(40):
                try:
                    check_idle(args.gpus)
                    break
                except RuntimeError:
                    time.sleep(1)
            check_idle(args.gpus)
        result["complete"] = True
    except Exception as exc:
        result["failure"] = repr(exc)
        raise
    finally:
        stop(proc)
        result["gpu_after"] = gpu_snapshot(args.gpus)
        save()


if __name__ == "__main__":
    main()
