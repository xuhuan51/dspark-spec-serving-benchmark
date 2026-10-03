"""Direct AR/native/optimized comparison plus separate greedy diagnostics.

Each paired round launches an AR engine and a DSpark engine in randomized
order. The latter uses the same graph pool for native, graph-only and budget
variants. The existing calibration is frozen, never fitted on these rows.
"""
import argparse
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import time

from benchmark import measure
from calibrate import rpc
from run import HERE, ROOT, WORKLOAD, check_idle, command, gpu_snapshot, metrics, runtime_environment, stop, wait_ready
from summarize import acceptance
from summarize_budget import graphs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="1,5,6,7")
    parser.add_argument("--port", type=int, default=18876)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--kv-gib", type=float, default=4.5)
    parser.add_argument("--profile", type=Path, default=ROOT/"configs/qwen38_uniform_budget.json")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    args.gpus = list(map(int, args.gpus.split(",")))
    args.eager = False
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = (HERE/".experiment.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError("Port is occupied")
    frozen = json.loads(args.profile.read_text())
    sources = args.output.with_suffix(".sources")
    sources.mkdir()
    for file in HERE.glob("*.py"):
        (sources/file.name).write_bytes(file.read_bytes())
    result = dict(complete=False, rows=[], diagnostics=[], launches=[],
                  gpu_before=check_idle(args.gpus),
                  profile=frozen, profile_sha256=hashlib.sha256(args.profile.read_bytes()).hexdigest(),
                  workload_sha256=hashlib.sha256(WORKLOAD.read_bytes()).hexdigest(),
                  settings=dict(rounds=args.rounds, concurrency=16, requests=16,
                                output_tokens=256, sample_offset=16, kv_gib=args.kv_gib),
                  source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")},
                  scope="Direct paired ratios; DSpark variants share one 45-graph pool per round. Diagnostics excluded from timing.")
    if args.resume:
        previous = json.loads(args.resume.read_text())
        for key in ["settings", "profile_sha256", "workload_sha256"]:
            if previous[key] != result[key]:
                raise ValueError(f"Resume changed {key}")
        if sorted(v['uuid'] for v in previous['gpu_before'].values()) != sorted(v['uuid'] for v in result['gpu_before'].values()):
            raise ValueError("Resume changed the GPU topology")
        if any(row['errors'] for row in previous['rows']):
            raise ValueError("Do not reuse incomplete/failed HTTP measurements")
        result['rows'] = previous['rows']
        result['diagnostics'] = previous['diagnostics']
        result['launches'] = previous['launches']
        result['resumed_from'] = dict(path=str(args.resume.resolve()),
                                      sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest(),
                                      failure=previous.get('failure'), sources=previous['source_hashes'])
    def save():
        temporary = args.output.with_name(args.output.name+'.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
        temporary.replace(args.output)
    save()
    base = f"http://127.0.0.1:{args.port}"
    rng = random.Random(261003)
    proc = None
    env = runtime_environment()
    env.update(VLLM_WORKER_MULTIPROC_METHOD="spawn", OMP_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false", VLLM_USE_FLASHINFER_SAMPLER="0",
               VLLM_NVTX_SCOPES_FOR_PROFILING="0", VLLM_SERVER_DEV_MODE="1",
               DSPARK_BUDGET_PROFILE=str(args.profile.resolve()))
    try:
        for rnd in range(args.rounds):
            engines = ["ar", "dspark"]
            rng.shuffle(engines)
            for engine in engines:
                cases = ['ar'] if engine == 'ar' else ['native', 'graphs', 'budget']
                if engine == 'dspark':
                    rng.shuffle(cases)
                completed = {(row['round'], row['mode']) for row in result['rows']}
                if all((rnd, mode) in completed for mode in cases):
                    continue
                before = check_idle(args.gpus)
                env["CUDA_VISIBLE_DEVICES"] = ",".join(before[i]["uuid"] for i in args.gpus)
                trace = args.output.with_suffix(f".r{rnd}.steps.jsonl")
                env["DSPARK_BUDGET_TRACE"] = str(trace.resolve())
                cmd = command("ar" if engine == "ar" else "fixed", args)
                if engine == "dspark":
                    index = cmd.index("--speculative-config")+1
                    spec = json.loads(cmd[index])
                    spec["num_speculative_tokens_per_batch_size"] = [[1,1,1],[2,2,2],[3,3,4],[4,32,7]]
                    cmd[index] = json.dumps(spec)
                    cmd += ["--scheduler-cls", "uniform_budget.UniformBudgetScheduler",
                            "--worker-extension-cls", "closure_graphs.ClosureGraphWorker"]
                else:
                    cmd += ["--worker-extension-cls", "decision_audit.DecisionAuditWorker"]
                log = args.output.with_suffix(f".r{rnd}.{engine}.log")
                with log.open("x") as stream:
                    proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT,
                                            env=env, cwd=HERE, start_new_session=True)
                    launch = dict(round=rnd, engine=engine, pid=proc.pid, command=cmd,
                                  log=str(log), gpu_before=before)
                    result["launches"].append(launch)
                    save()
                    print(f"START round={rnd} engine={engine} pid={proc.pid}", flush=True)
                    wait_ready(proc, base)
                    if engine == "dspark":
                        launch["graph_pool"] = rpc(base, "budget_install_metrics")
                    for mode in cases:
                        if (rnd, mode) in completed:
                            continue
                        extra = None if mode == "ar" else {"lab_mode": "budget" if mode == "budget" else "native"}
                        if engine == "dspark":
                            rpc(base, "budget_set_graph_policy", "original" if mode == "native" else "full")
                        warm = asyncio.run(measure(base, WORKLOAD, concurrency=16, requests=16,
                                                   output_tokens=32, sample_offset=16, extra_args=extra))
                        if warm["errors"]:
                            raise RuntimeError("Warmup failed")
                        if engine == "dspark":
                            rpc(base, "budget_reset_metrics")
                        pre = metrics(base)
                        row = asyncio.run(measure(base, WORKLOAD, concurrency=16, requests=16,
                                                  output_tokens=256, sample_offset=16, extra_args=extra))
                        time.sleep(1.1)
                        row.update(round=rnd, mode=mode, metrics_before=pre, metrics_after=metrics(base))
                        row["acceptance"] = acceptance(row)
                        if engine == "dspark":
                            row["graph_metrics"] = rpc(base, "budget_metrics")
                            row["graph_summary"] = graphs(row)
                        result["rows"].append(row)
                        save()
                        print("RESULT "+json.dumps(dict(round=rnd, mode=mode, summary=row.get("summary"))), flush=True)
                        if row["errors"]:
                            raise RuntimeError("Measurement errors retained")
                    if args.diagnostics and rnd == 0:
                        for mode in (["ar"] if engine == "ar" else ["native", "budget"]):
                            extra = None if mode == "ar" else {"lab_mode": mode}
                            if engine == "dspark":
                                rpc(base, "budget_set_graph_policy", "original" if mode == "native" else "full")
                            path = args.output.with_suffix(f".diagnostic.{mode}")
                            entry = dict(mode=mode, traces=rpc(base, "decision_audit_start", str(path.resolve())), rows=[])
                            result["diagnostics"].append(entry)
                            save()
                            try:
                                # Repeated isolated requests distinguish output instability
                                # from correctness of the rejection decision itself.
                                for c, repeat in [(1,0), (1,1), (16,0)]:
                                    row = asyncio.run(measure(base, WORKLOAD, concurrency=c, requests=c,
                                                              output_tokens=128, sample_offset=16, extra_args=extra))
                                    row.update(repeat=repeat)
                                    entry["rows"].append(row)
                                    save()
                                    if row["errors"]:
                                        raise RuntimeError("Decision diagnostic failed")
                                    print(f"DIAGNOSTIC mode={mode} c={c} repeat={repeat}", flush=True)
                            finally:
                                entry["stopped"] = rpc(base, "decision_audit_stop")
                                save()
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
