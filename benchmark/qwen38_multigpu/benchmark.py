"""Token-counted HTTP measurements on one frozen workload, with raw SSE output."""
import asyncio
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import aiohttp


async def request(session, base, sample, output_tokens, index, *, natural=False, extra_args=None, stop=None):
    started = time.perf_counter()
    first = last = None
    usage = None
    api_request_id = None
    pieces, events, token_ids, finish = [], [], [], None
    first_chunk_tokens = None
    payload = dict(model="qwen38-dspark-lab", prompt=sample["prompt_token_ids"],
                   temperature=0, seed=17, max_tokens=output_tokens,
                   ignore_eos=not natural, stream=True,
                   return_token_ids=True, stream_options={"include_usage": True})
    if extra_args:
        payload["vllm_xargs"] = extra_args
    if stop:
        payload["stop"] = stop
    async with session.post(base + "/v1/completions", json=payload) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:1000]}")
        # read lines explicitly: an SSE data event may cross TCP packet boundaries.
        while True:
            raw = await response.content.readline()
            if not raw:
                break
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            event = json.loads(line[5:].strip())
            if event.get("id"):
                api_request_id = event["id"]
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                text = choice.get("text", "")
                delta = choice.get("token_ids") or []
                token_ids.extend(delta)
                if text or delta:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                        first_chunk_tokens = len(delta)
                    last = now
                    pieces.append(text)
                    events.append({"elapsed_s": now-started, "text": text, "token_ids": delta})
    ended = time.perf_counter()
    if not usage or first is None:
        raise RuntimeError("Missing actual token usage or nonempty stream")
    if usage["prompt_tokens"] != len(sample["prompt_token_ids"]):
        raise RuntimeError("Input token count changed")
    count = usage["completion_tokens"]
    if len(token_ids) != count:
        raise RuntimeError(f"Token ID count {len(token_ids)} differs from usage {count}")
    if not natural and count != output_tokens:
        raise RuntimeError(f"Expected {output_tokens} output tokens, got {count}")
    return dict(index=index, input_tokens=usage["prompt_tokens"], output_tokens=count,
                api_request_id=api_request_id,
                prompt_sha256=sample["prompt_sha256"], finish_reason=finish,
                ttft_ms=1000*(first-started), e2e_ms=1000*(ended-started),
                tpot_ms=1000*(last-first)/(count-1) if count > 1 else None,
                text="".join(pieces), token_ids=token_ids, first_chunk_tokens=first_chunk_tokens,
                post_first_chunk_ms_per_token=1000*(last-first)/(count-first_chunk_tokens)
                if first_chunk_tokens is not None and count>first_chunk_tokens else None,
                stream_events=events)


async def measure(base, workload, *, concurrency, requests, output_tokens, sample_offset=0, extra_args=None):
    samples = json.loads(Path(workload).read_text())["samples"]["main"]
    queue = asyncio.Queue()
    for i in range(requests):
        queue.put_nowait(i)
    rows, errors = [], []
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def worker():
            while True:
                try:
                    i = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    rows.append(await request(session, base, samples[(sample_offset+i) % len(samples)],
                                              output_tokens, i, extra_args=extra_args))
                except Exception as exc:
                    errors.append(dict(index=i, error=repr(exc)))
        start = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        seconds = time.perf_counter()-start
    result = dict(concurrency=concurrency, requests=requests, output_tokens=output_tokens,
                  sample_offset=sample_offset, extra_args=extra_args,
                  elapsed_s=seconds, rows=sorted(rows, key=lambda x:x["index"]), errors=errors,
                  workload_sha256=hashlib.sha256(Path(workload).read_bytes()).hexdigest(),
                  tpot_definition="first-to-last nonempty SSE arrival divided by actual output tokens minus one; not GPU step latency")
    if rows:
        summary = dict(tokens_per_second=sum(r["output_tokens"] for r in rows)/seconds,
                       completed=len(rows), errors=len(errors))
        for metric in ["ttft_ms", "tpot_ms", "e2e_ms"]:
            values = sorted(r[metric] for r in rows if r[metric] is not None)
            if values:
                summary[metric+"_median"] = statistics.median(values)
                summary[metric+"_p95"] = values[math.ceil(.95*len(values))-1]
        result["summary"] = summary
    return result
