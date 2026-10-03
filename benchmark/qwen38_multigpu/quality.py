"""Small fixed natural-EOS and stop checks, separate from throughput tests."""
import asyncio
import hashlib
import json
import os
from pathlib import Path

import aiohttp

from benchmark import request
from run import HERE, ROOT, MODEL, WORKLOAD

QUALITY_CASES = Path(os.environ.get("DSPARK_QUALITY_CASES", str(ROOT/"results/qwen38_multigpu_20261003/quality-cases.json")))

CASES = [
    ("sum", "Reply with only the integer: 17 + 25.", "42"),
    ("product", "Reply with only the integer: 8 * 9.", "72"),
    ("power", "Reply with only the integer: 2 to the fifth power.", "32"),
    ("factorial", "Reply with only the integer: factorial of 5.", "120"),
    ("reverse", "Reverse abcd. Reply with only the reversed string.", "dcba"),
    ("sort", "Sort these numbers ascending: 3,1,2. Reply exactly in comma-separated format without spaces.", "1,2,3"),
    ("json", 'Reply with only this JSON, without code fences or spaces: {"ok":true}', '{"ok":true}'),
    ("boolean", "Is 9 a prime number? Reply with only false or true.", "false"),
    ("france", "What is the capital of France? Reply with only the city name.", "Paris"),
    ("china", "中国的首都是哪里？只输出城市名。", "北京"),
    ("translation", "Translate 猫 into English. Reply with only the lowercase word.", "cat"),
    ("days", "How many days are in one week? Reply with only the integer.", "7"),
    ("letter", "What letter comes immediately after a in the English alphabet? Reply with only the lowercase letter.", "b"),
    ("count", "Count the words in 'red green blue'. Reply with only the integer.", "3"),
    ("binary", "Convert the binary number 1010 to decimal. Reply with only the integer.", "10"),
    ("minimum", "What is the minimum of 18, 4, 7? Reply with only the integer.", "4"),
]


def prepare():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    cases = []
    for name, text, expected in CASES + [
        ("stop_newline", "Write exactly these three lines, without code fences:\nalpha\nbeta\ngamma", "alpha"),
        ("stop_marker", "Print exactly: first END second. Do not add anything else.", "first"),
    ]:
        ids = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                             tokenize=True, add_generation_prompt=True,
                                             enable_thinking=False)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        cases.append(dict(name=name, expected=expected, prompt_token_ids=ids,
                          prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                          stop=["\n"] if name == "stop_newline" else ["END"] if name == "stop_marker" else None))
    output = QUALITY_CASES
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(cases=cases, model=str(MODEL)), ensure_ascii=False, indent=2) + "\n")
    print(output)


async def check(base, extra_args=None):
    cases = json.loads(QUALITY_CASES.read_text())["cases"]
    rows = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        for i, sample in enumerate(cases):
            try:
                row = await request(session, base, sample, 128, i, natural=True,
                                    stop=sample["stop"], extra_args=extra_args)
                row.update(name=sample["name"], expected=sample["expected"],
                           exact_match=row["text"].strip() == sample["expected"], stop=sample["stop"])
            except Exception as exc:
                row = dict(name=sample["name"], error=repr(exc))
            rows.append(row)
    natural = rows[:len(CASES)]
    return dict(rows=rows, errors=sum("error" in r for r in rows),
                natural_exact_match=sum(r.get("exact_match", False) for r in natural),
                natural_count=len(CASES),
                natural_ended=sum(r.get("finish_reason") == "stop" for r in natural),
                scope="16 small deterministic tasks plus 2 explicit stop checks; not a general model-quality evaluation")


async def score_sequences(base, sources):
    """Teacher-force frozen generated sequences on a target-only HTTP engine.

    This diagnoses numerical differences, not answer correctness or distribution
    preservation. The target-only AR sequences are checked too, as a control.
    """
    workload = json.loads(WORKLOAD.read_text())
    prompts = {s["prompt_sha256"]: s["prompt_token_ids"] for s in workload["samples"]["main"]}
    unique, groups = {}, []
    for path in sources:
        x = json.loads(Path(path).read_text())
        for measurement in x.get("rows", x.get("validation", [])):
            if measurement["round"] != 0:
                continue
            group = dict(source=str(Path(path).resolve()), mode=measurement["mode"],
                         concurrency=measurement["concurrency"], sequences=[])
            for row in measurement["rows"]:
                key = hashlib.sha256(json.dumps([row["prompt_sha256"], row["token_ids"]]).encode()).hexdigest()
                unique[key] = (prompts[row["prompt_sha256"]], row["token_ids"])
                group["sequences"].append(key)
            groups.append(group)
    scores = {}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        for key, (prompt, generated) in unique.items():
            payload = dict(model="qwen38-dspark-lab", prompt=prompt+generated, temperature=0,
                           max_tokens=1, stream=False, prompt_logprobs=1, return_token_ids=True)
            async with session.post(base + "/v1/completions", json=payload) as response:
                data = await response.json()
                if response.status != 200:
                    raise RuntimeError(f"Reference scoring HTTP {response.status}: {data}")
            logprobs = data["choices"][0]["prompt_logprobs"]
            if len(logprobs) != len(prompt)+len(generated):
                raise RuntimeError("Reference score length changed")
            details = []
            for i, token in enumerate(generated):
                candidates = logprobs[len(prompt)+i]
                true = candidates[str(token)]
                top = max(v["logprob"] for v in candidates.values())
                details.append(dict(token=token, rank=true["rank"], logprob=true["logprob"],
                                    regret=max(0.0, top-true["logprob"])))
            scores[key] = dict(prompt_tokens=len(prompt), details=details)
    summary = []
    for group in groups:
        details = [item for key in group["sequences"] for item in scores[key]["details"]]
        summary.append({**group, "tokens": len(details),
                        "rank1_fraction": sum(d["rank"] == 1 for d in details)/len(details),
                        "mean_regret": sum(d["regret"] for d in details)/len(details),
                        "max_regret": max(d["regret"] for d in details),
                        "regret_over_0_1": sum(d["regret"] > .1 for d in details)})
    return dict(scores=scores, summary=summary, unique_sequences=len(scores),
                scope="target-only BF16 teacher forcing; rank1 and logprob regret are numerical diagnostics, not task accuracy")


if __name__ == "__main__":
    prepare()
