"""Start only owned server processes and retain every measurement and failure."""
import argparse
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from benchmark import measure

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHON = Path(os.environ.get("DSPARK_PYTHON", sys.executable))
MODEL = Path(os.environ.get("DSPARK_MODEL", str(Path.home()/"models/Qwen3.8-27B"))).expanduser()
DRAFT = Path(os.environ.get("DSPARK_DRAFT_MODEL", str(Path.home()/"models/Qwen3.8-27B-speculator.dspark"))).expanduser()
WORKLOAD = Path(os.environ.get("DSPARK_WORKLOAD", str(ROOT/"results/qwen38_multigpu_20261003/workload.json"))).expanduser()


def runtime_environment():
    """Expose the local adapters and an optional separately installed runtime."""
    env = os.environ.copy()
    paths = [env.get("DSPARK_RUNTIME"), str(HERE), env.get("PYTHONPATH")]
    env["PYTHONPATH"] = os.pathsep.join(p for p in paths if p)
    return env


def gpu_snapshot(indices):
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True)
    rows = {}
    for line in raw.splitlines():
        index, uuid, memory, util = [x.strip() for x in line.split(",")]
        if int(index) in indices:
            rows[int(index)] = dict(uuid=uuid, memory_mib=int(memory), utilization=int(util))
    if len(rows) != len(indices):
        raise RuntimeError("Missing requested GPU")
    return rows


def check_idle(indices):
    rows = gpu_snapshot(indices)
    # Utilization can retain its last busy sample after context destruction.
    # Actual CUDA owners and memory, rather than that sample, gate admission.
    raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                   "--format=csv,noheader,nounits"], text=True)
    owners = {line.split(",", 1)[0].strip() for line in raw.splitlines() if line.strip()}
    for i, row in rows.items():
        if row["memory_mib"] > 100 or row["uuid"] in owners:
            raise RuntimeError(f"GPU {i} is not idle: {row}")
    return rows


def metrics(base):
    with urllib.request.urlopen(base+"/metrics", timeout=10) as r:
        return r.read().decode()


def command(mode, args):
    cmd = [str(PYTHON), "-m", "vllm.entrypoints.openai.api_server", "--model", str(MODEL),
           "--served-model-name", "qwen38-dspark-lab", "--host", "127.0.0.1", "--port", str(args.port),
           "--tensor-parallel-size", str(len(args.gpus)), "--dtype", "bfloat16", "--language-model-only",
           "--max-model-len", "4096", "--max-num-seqs", "32", "--max-num-batched-tokens", "2048",
           "--gpu-memory-utilization", "0.88", "--kv-cache-memory-bytes", str(int(args.kv_gib*1024**3)),
           "--no-enable-prefix-caching", "--enable-chunked-prefill", "--generation-config", "vllm",
           "--seed", "17", "--mamba-ssm-cache-dtype", "float32",
           "--compilation-config", json.dumps({"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY",
               "cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,256]})]
    if mode != "ar":
        config = dict(model=str(DRAFT), method="dspark", num_speculative_tokens=args.k,
                      enable_adaptive_verification=mode == "adaptive")
        cmd += ["--speculative-config", json.dumps(config)]
    if args.eager:
        cmd += ["--enforce-eager"]
    return cmd


def wait_ready(proc, base, timeout=900):
    start = time.monotonic()
    while time.monotonic()-start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(base+"/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("Server startup timed out")


def stop(proc):
    if proc is None:
        return
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpus", default="1,5,6,7")
    p.add_argument("--modes", default="ar,fixed")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--concurrency", default="1,4,16")
    p.add_argument("--requests", type=int, default=16)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--k", type=int, default=7)
    p.add_argument("--kv-gib", type=float, default=4.5)
    p.add_argument("--sample-offset", type=int, default=16)
    p.add_argument("--port", type=int, default=18876)
    p.add_argument("--eager", action="store_true")
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    args.gpus = list(map(int,args.gpus.split(",")))
    modes=args.modes.split(","); batches=list(map(int,args.concurrency.split(",")))
    if set(modes)-{"ar","fixed","adaptive"} or min(batches)<1 or max(batches)>32:
        raise ValueError("Invalid modes/concurrency")
    if args.requests < 1 or args.sample_offset < 0:
        raise ValueError("Invalid request count or sample offset")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    lock=(HERE/".experiment.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1",args.port)) == 0:
            raise RuntimeError("Port is occupied")
    env=runtime_environment()
    env.update(VLLM_WORKER_MULTIPROC_METHOD="spawn", OMP_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false", VLLM_USE_FLASHINFER_SAMPLER="0",
               VLLM_NVTX_SCOPES_FOR_PROFILING="0")
    if "fixed" in modes or "adaptive" in modes:
        if not (DRAFT/"download-manifest.json").exists():
            raise RuntimeError("Draft weights have not passed pinned SHA-256 verification")
    source_dir=args.output.with_suffix(".sources")
    source_dir.mkdir()
    for f in HERE.glob("*.py"):
        (source_dir/f.name).write_bytes(f.read_bytes())
    result=dict(scope="Qwen3.8-27B BF16 TP4, actual HTTP requests; same frozen workload and KV budget",
                gpu_before=check_idle(args.gpus), draft_revision="87ca2fdc67f316f6c1f9ebc7ef7bbb0ad299a4ce",
                settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                sources={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in HERE.glob("*.py")},
                rows=[], launches=[], complete=False)
    if args.resume:
        previous=json.loads(args.resume.read_text())
        for key in ["gpus","modes","rounds","concurrency","requests","tokens","k","kv_gib","sample_offset","eager"]:
            if previous["settings"][key]!=result["settings"][key]:
                raise ValueError(f"Cannot resume with changed setting {key}")
        result["rows"]=[r for r in previous["rows"] if not r["errors"]]
        result["launches"]=previous["launches"]
        result["resumed_from"]=dict(path=str(args.resume.resolve()),
                                    sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest(),
                                    failure=previous.get("failure"), sources=previous["sources"])
    def save():
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    save();rng=random.Random(91003);base=f"http://127.0.0.1:{args.port}"
    proc=None
    try:
        for r in range(args.rounds):
            order=modes.copy();rng.shuffle(order)
            for mode in order:
                completed={(row["round"],row["mode"],row["concurrency"]) for row in result["rows"]}
                if all((r,mode,c) in completed for c in batches):
                    continue
                before=check_idle(args.gpus)
                env["CUDA_VISIBLE_DEVICES"]=",".join(before[i]["uuid"] for i in args.gpus)
                cmd=command(mode,args)
                log=args.output.with_suffix(f".r{r}.{mode}.log")
                with log.open("w") as stream:
                    proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,
                                          env=env,cwd=HERE,start_new_session=True)
                    result["launches"].append(dict(round=r,mode=mode,command=cmd,pid=proc.pid,log=str(log),gpus=before))
                    save();print(f"START round={r} mode={mode} pid={proc.pid}",flush=True)
                    wait_ready(proc,base)
                    for c in batches:
                        if (r,mode,c) in completed:
                            continue
                        warm=asyncio.run(measure(base,WORKLOAD,concurrency=c,requests=c,output_tokens=32,
                                                sample_offset=args.sample_offset))
                        if warm["errors"]:
                            raise RuntimeError(f"Warmup error: {warm['errors']}")
                        pre=metrics(base)
                        row=asyncio.run(measure(base,WORKLOAD,concurrency=c,requests=max(args.requests,c),output_tokens=args.tokens,
                                               sample_offset=args.sample_offset))
                        time.sleep(1.1)
                        row.update(round=r,mode=mode,metrics_before=pre,metrics_after=metrics(base))
                        result["rows"].append(row);save()
                        print("RESULT "+json.dumps({k:row[k] for k in ["round","mode","concurrency","summary"]}),flush=True)
                        if row["errors"]:
                            raise RuntimeError("Measurement errors retained; stop comparison")
                    stop(proc);proc=None
                # Allow process exit and CUDA context cleanup, then demand empty GPUs again.
                for _ in range(30):
                    try:
                        check_idle(args.gpus);break
                    except RuntimeError:
                        time.sleep(1)
                check_idle(args.gpus)
        result["complete"]=True
    except Exception as exc:
        result["failure"]=repr(exc)
        raise
    finally:
        stop(proc);result["gpu_after"]=gpu_snapshot(args.gpus);save()


if __name__=="__main__":
    main()
