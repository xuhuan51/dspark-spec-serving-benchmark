"""Calibrate uniform prefixes, freeze the table, then test held-out prompts.

All prefix variants use one engine, the same full draft block and graph pool.
Native DSpark is the increment baseline. Upstream dynamic-SD interfaces are
reused; this is a local engineering study, not a new speculative algorithm.
"""
import argparse
import asyncio
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time
import urllib.request

from benchmark import measure
from quality import check
from run import HERE, DRAFT, WORKLOAD, check_idle, command, gpu_snapshot, metrics, runtime_environment, stop, wait_ready
from summarize import acceptance


def rpc(base, method, *args):
    req = urllib.request.Request(base + "/collective_rpc", data=json.dumps({"method": method, "args": list(args)}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpus", default="1,5,6,7")
    p.add_argument("--port", type=int, default=18876)
    p.add_argument("--k", type=int, default=7)
    p.add_argument("--kv-gib", type=float, default=4.5)
    p.add_argument("--calibration-rounds", type=int, default=2)
    p.add_argument("--validation-rounds", type=int, default=5)
    args = p.parse_args(); args.gpus = list(map(int, args.gpus.split(","))); args.eager = False
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not (DRAFT / "download-manifest.json").exists():
        raise RuntimeError("Draft weights are not verified")
    lock = (HERE / ".experiment.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    before = check_idle(args.gpus)
    profile = args.output.with_suffix(".profile.json")
    trace = args.output.with_suffix(".steps.jsonl")
    if profile.exists() or trace.exists():
        raise FileExistsError("Profile/trace must be fresh")
    sources = args.output.with_suffix(".sources"); sources.mkdir()
    for f in HERE.glob("*.py"):
        (sources / f.name).write_bytes(f.read_bytes())
    env = runtime_environment()
    env.update(CUDA_VISIBLE_DEVICES=",".join(before[i]["uuid"] for i in args.gpus),
               VLLM_WORKER_MULTIPROC_METHOD="spawn", OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
               VLLM_USE_FLASHINFER_SAMPLER="0", VLLM_NVTX_SCOPES_FOR_PROFILING="0", VLLM_SERVER_DEV_MODE="1",
               DSPARK_BUDGET_PROFILE=str(profile.resolve()), DSPARK_BUDGET_TRACE=str(trace.resolve()))
    cmd = command("fixed", args)
    ci = cmd.index("--speculative-config") + 1
    config = json.loads(cmd[ci])
    # Declare all capture widths; the scheduler keeps drafting all seven tokens.
    config["num_speculative_tokens_per_batch_size"] = [[1, 1, 1], [2, 2, 2], [3, 3, 4], [4, 32, 7]]
    cmd[ci] = json.dumps(config)
    cmd += ["--scheduler-cls", "uniform_budget.UniformBudgetScheduler",
            "--worker-extension-cls", "uniform_graphs.UniformGraphWorker"]
    result = dict(command=cmd, gpu_before=before, calibration=[], validation=[], complete=False,
                  policy_boundary="frozen calibration table; uniform width, full native drafting, native rejection and state management",
                  source_hashes={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in HERE.glob("*.py")})
    def save():
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    proc = None; rng = random.Random(91003); base = f"http://127.0.0.1:{args.port}"
    def run_row(stage, rnd, c, mode, k=None):
        kwargs = {"lab_mode": mode, "lab_k": k} if mode == "prefix" else {"lab_mode": "native" if mode == "graphs" else mode}
        rpc(base, "budget_set_graph_policy", "original" if mode == "native" else "full")
        offset = 0 if stage == "calibration" else 16
        warm = asyncio.run(measure(base, WORKLOAD, concurrency=c, requests=c, output_tokens=32,
                                   sample_offset=offset, extra_args=kwargs))
        if warm["errors"]:
            raise RuntimeError(f"Warmup failed: {warm['errors']}")
        rpc(base, "budget_reset_metrics")
        pre = metrics(base)
        trace_start = trace.stat().st_size
        row = asyncio.run(measure(base, WORKLOAD, concurrency=c, requests=max(4, c),
                                  output_tokens=96 if stage == "calibration" else 256,
                                  sample_offset=offset, extra_args=kwargs))
        time.sleep(1.1)
        row.update(round=rnd, mode=mode, prefix=k, metrics_before=pre, metrics_after=metrics(base),
                   graph_metrics=rpc(base, "budget_metrics"))
        trace_end = trace.stat().st_size
        with trace.open("rb") as stream:
            stream.seek(trace_start)
            steps = [json.loads(line) for line in stream.read(trace_end-trace_start).splitlines()]
        histogram = Counter(s["requests"] for s in steps if s["pure_verification"])
        row["active_batch_histogram"] = dict(histogram)
        row["dominant_active_batch"] = max(histogram, key=lambda n: (n*histogram[n], n)) if histogram else c
        row["trace_byte_range"] = [trace_start, trace_end]
        row["acceptance"] = acceptance(row)
        result[stage].append(row); save()
        print("RESULT " + json.dumps(dict(stage=stage, round=rnd, concurrency=c, mode=mode,
                                         prefix=k, summary=row.get("summary"))), flush=True)
        if row["errors"]:
            raise RuntimeError("Measurement failed")
        return row
    try:
        with args.output.with_suffix(".log").open("w") as stream:
            proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT,
                                    env=env, cwd=HERE, start_new_session=True)
            result["pid"] = proc.pid; save(); print(f"START calibration pid={proc.pid}", flush=True)
            wait_ready(proc, base)
            result["initial_graphs"] = rpc(base, "budget_install_metrics"); save()
            for r in range(args.calibration_rounds):
                cases = [(c, k) for c in [1, 4, 16] for k in [1, 2, 4, 7]]
                rng.shuffle(cases)
                for c, k in cases:
                    run_row("calibration", r, c, "prefix", k)
            table, costs = {}, []
            for c in [1, 4, 16]:
                rates = {k: statistics.median(row["summary"]["tokens_per_second"] for row in result["calibration"]
                                            if row["concurrency"] == c and row["prefix"] == k)
                         for k in [1, 2, 4, 7]}
                best = max(rates, key=rates.get)
                active = round(statistics.median(row["dominant_active_batch"] for row in result["calibration"]
                                                 if row["concurrency"] == c and row["prefix"] == 7))
                # Use the admitted GPU batch, not offered HTTP concurrency.
                # Keep native K=7 when the calibration gain is smaller than 3%.
                chosen = best if rates[best] > rates[7] * 1.03 else 7
                if str(active) not in table:
                    table[str(active)] = chosen
                costs.append(dict(concurrency=c, active_bucket=active, tokens_per_second=rates, chosen=table[str(active)]))
            frozen = dict(batch_to_prefix=table, calibration=costs,
                          calibration_offset=0, validation_offset=16, gain_gate=1.03)
            profile.write_text(json.dumps(frozen, indent=2) + "\n"); result["profile"] = frozen; save()
            for r in range(args.validation_rounds):
                cases = [(c, mode) for c in [1, 4, 16] for mode in ["native", "graphs", "budget"]]
                rng.shuffle(cases)
                for c, mode in cases:
                    run_row("validation", r, c, mode)
            rpc(base, "budget_set_graph_policy", "original")
            result["quality_native"] = asyncio.run(check(base, {"lab_mode": "native"}))
            result["quality_native"]["graph_policy"] = "original"
            rpc(base, "budget_set_graph_policy", "full")
            result["quality_budget"] = asyncio.run(check(base, {"lab_mode": "budget"}))
            result["quality_budget"]["graph_policy"] = "full"
            if result["quality_native"]["errors"] or result["quality_budget"]["errors"]:
                raise RuntimeError("Quality checks failed")
            result["complete"] = True
    except Exception as exc:
        result["failure"] = repr(exc)
        raise
    finally:
        stop(proc)
        for _ in range(30):
            if all(v["memory_mib"] < 100 for v in gpu_snapshot(args.gpus).values()):
                break
            time.sleep(1)
        result["gpu_after"] = gpu_snapshot(args.gpus); save()


if __name__ == "__main__":
    main()
