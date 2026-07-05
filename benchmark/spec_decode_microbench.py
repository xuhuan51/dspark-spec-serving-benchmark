#!/usr/bin/env python3
"""Decode-only microbenchmark for speculative decoding.

This runner keeps prompt/payload shape intentionally simple so baseline and
speculative decoding differ only in the engine launch config.
"""

import argparse
import asyncio
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path

from openai import AsyncOpenAI


PROMPT = (
    "Write a detailed but direct explanation of how LLM inference decode works. "
    "Use concrete systems terms and continue until the token budget is exhausted."
)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(len(xs) - 1, int(len(xs) * p / 100))
    return xs[idx]


def scrape_metrics(base_root: str) -> dict:
    out = {"ts": time.time()}
    try:
        with urllib.request.urlopen(base_root + "/metrics", timeout=3) as resp:
            text = resp.read().decode("utf-8", "replace")
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    wanted = {
        "sglang:gen_throughput": "gen_throughput",
        "sglang:spec_accept_length": "spec_accept_length",
        "sglang:num_running_reqs": "num_running_reqs",
        "sglang:num_queue_reqs": "num_queue_reqs",
        "sglang:token_usage": "token_usage",
    }
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        for metric, key in wanted.items():
            if line.startswith(metric):
                try:
                    out[key] = float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
                break
    return out


class MetricsSampler(threading.Thread):
    def __init__(self, base_root: str, out_path: Path, interval_s: float = 1.0):
        super().__init__(daemon=True)
        self.base_root = base_root
        self.out_path = out_path
        self.interval_s = interval_s
        self.stop = False
        self.samples: list[dict] = []

    def run(self) -> None:
        with self.out_path.open("w") as fh:
            while not self.stop:
                row = scrape_metrics(self.base_root)
                self.samples.append(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                time.sleep(self.interval_s)


async def one_request(
    client: AsyncOpenAI,
    model: str,
    request_id: int,
    max_tokens: int,
    ignore_eos: bool,
) -> dict:
    t0 = time.perf_counter()
    first = None
    chunks = 0
    chars = 0
    err = ""
    ok = True
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=max_tokens,
            temperature=0.0,
            stream=True,
            extra_body={"ignore_eos": True} if ignore_eos else None,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            text = chunk.choices[0].delta.content or ""
            if not text:
                continue
            if first is None:
                first = time.perf_counter()
            chunks += 1
            chars += len(text)
    except Exception as exc:
        ok = False
        err = f"{type(exc).__name__}: {exc}"
    t1 = time.perf_counter()
    ttft_ms = ((first or t1) - t0) * 1000
    total_ms = (t1 - t0) * 1000
    decode_ms = max(0.0, total_ms - ttft_ms)
    tpot_ms = decode_ms / max(chunks - 1, 1)
    output_tokens_est = max_tokens if ok and ignore_eos else chunks
    return {
        "request_id": request_id,
        "success": ok,
        "error": err,
        "chunks": chunks,
        "output_tokens_est": output_tokens_est,
        "chars": chars,
        "ttft_ms": ttft_ms,
        "decode_ms": decode_ms,
        "total_ms": total_ms,
        "tpot_ms": decode_ms / max(output_tokens_est - 1, 1),
        "chunk_tpot_ms": tpot_ms,
    }


async def run_case(args: argparse.Namespace, concurrency: int, out_dir: Path) -> dict:
    client = AsyncOpenAI(base_url=args.base_url, api_key="not-needed", timeout=args.timeout_s)
    sem = asyncio.Semaphore(concurrency)
    nreq = max(args.min_requests, args.requests_per_concurrency * concurrency)

    async def guarded(i: int) -> dict:
        async with sem:
            return await one_request(client, args.model, i, args.max_tokens, args.ignore_eos)

    # Warm the exact request shape before measurement.
    for i in range(args.warmup_requests):
        await one_request(client, args.model, -1 - i, args.max_tokens, args.ignore_eos)

    sampler = MetricsSampler(args.base_url.removesuffix("/v1"), out_dir / "metrics.jsonl")
    sampler.start()
    t0 = time.perf_counter()
    rows = await asyncio.gather(*(guarded(i) for i in range(nreq)))
    wall_s = time.perf_counter() - t0
    sampler.stop = True
    sampler.join(timeout=3)

    with (out_dir / "requests.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    ok = [r for r in rows if r["success"]]
    tpots = [r["tpot_ms"] for r in ok]
    ttfts = [r["ttft_ms"] for r in ok]
    total_chunks = sum(r["chunks"] for r in ok)
    total_output_tokens = sum(r["output_tokens_est"] for r in ok)
    gen_tp = [
        s.get("gen_throughput")
        for s in sampler.samples
        if isinstance(s.get("gen_throughput"), float) and s.get("gen_throughput", 0) > 0
    ]
    accept = [
        s.get("spec_accept_length")
        for s in sampler.samples
        if isinstance(s.get("spec_accept_length"), float) and s.get("spec_accept_length", 0) > 0
    ]
    summary = {
        "tag": args.tag,
        "concurrency": concurrency,
        "requests": nreq,
        "success": len(ok),
        "wall_s": wall_s,
        "chunks": total_chunks,
        "output_tokens_est": total_output_tokens,
        "client_chunk_throughput": total_chunks / wall_s if wall_s > 0 else 0.0,
        "client_token_throughput_est": (
            total_output_tokens / wall_s if wall_s > 0 else 0.0
        ),
        "ttft_ms_avg": statistics.mean(ttfts) if ttfts else 0.0,
        "ttft_ms_p50": percentile(ttfts, 50),
        "ttft_ms_p95": percentile(ttfts, 95),
        "tpot_ms_avg": statistics.mean(tpots) if tpots else 0.0,
        "tpot_ms_p50": percentile(tpots, 50),
        "tpot_ms_p95": percentile(tpots, 95),
        "metrics_gen_throughput_median": statistics.median(gen_tp) if gen_tp else 0.0,
        "metrics_spec_accept_length_median": statistics.median(accept) if accept else 0.0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8550/v1")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--concurrencies", default="1,2,4,8")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--requests-per-concurrency", type=int, default=4)
    parser.add_argument("--min-requests", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--ignore-eos", action="store_true")
    args = parser.parse_args()

    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    ladder = root / "ladder.csv"
    ladder.write_text(
        "tag,c,requests,success,wall_s,chunks,client_chunk_throughput,"
        "output_tokens_est,client_token_throughput_est,ttft_ms_p50,"
        "tpot_ms_p50,tpot_ms_p95,gen_tp_median,accept_len_median\n"
    )
    for c in [int(x) for x in args.concurrencies.split(",") if x.strip()]:
        case_dir = root / f"c{c}"
        case_dir.mkdir(exist_ok=True)
        summary = await run_case(args, c, case_dir)
        with ladder.open("a") as fh:
            fh.write(
                f"{args.tag},{c},{summary['requests']},{summary['success']},"
                f"{summary['wall_s']:.3f},{summary['chunks']},"
                f"{summary['client_chunk_throughput']:.3f},"
                f"{summary['output_tokens_est']},"
                f"{summary['client_token_throughput_est']:.3f},"
                f"{summary['ttft_ms_p50']:.3f},{summary['tpot_ms_p50']:.3f},"
                f"{summary['tpot_ms_p95']:.3f},"
                f"{summary['metrics_gen_throughput_median']:.3f},"
                f"{summary['metrics_spec_accept_length_median']:.3f}\n"
            )
        print(json.dumps(summary, indent=2), flush=True)
    print(ladder.read_text())


if __name__ == "__main__":
    asyncio.run(main())
