"""Resumable public checkpoint ranges; validate ranges and the upstream LFS hash."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.request

REPO="RedHatAI/Qwen3.8-27B-speculator.dspark"
REV="87ca2fdc67f316f6c1f9ebc7ef7bbb0ad299a4ce"
DEST=Path(os.environ.get("DSPARK_DRAFT_MODEL", str(Path.home()/"models/Qwen3.8-27B-speculator.dspark"))).expanduser()
URL=f"https://huggingface.co/{REPO}/resolve/{REV}/model.safetensors?download=true"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    meta=json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}/revision/{REV}?blobs=true",timeout=30))
    file=next(x for x in meta["siblings"] if x["rfilename"]=="model.safetensors")
    size=file["lfs"]["size"];expected=file["lfs"]["sha256"]
    target=DEST/"model.safetensors"
    if target.exists():
        raise FileExistsError(target)
    parts=DEST/".range-download";parts.mkdir(exist_ok=True)
    chunk=1024**2
    start=time.monotonic()
    with urllib.request.urlopen(urllib.request.Request(URL,headers={"Range":"bytes=0-1023"}),timeout=30) as r:
        cdn=r.url
    lock=threading.Lock();done=0
    def fetch(i):
        nonlocal done
        a=i*chunk;b=min(size,a+chunk)-1;dest=parts/f"{i:05}.part"
        if not dest.exists() or dest.stat().st_size!=b-a+1:
            for retry in range(5):
                try:
                    request=urllib.request.Request(cdn,headers={"Range":f"bytes={a}-{b}"})
                    with urllib.request.urlopen(request,timeout=120) as r:
                        correct=f"bytes {a}-{b}/{size}"
                        if r.status!=206 or r.headers.get("Content-Range")!=correct:
                            raise ValueError(f"Unexpected HTTP range at part {i}: {r.status}, {r.headers.get('Content-Range')}")
                        data=r.read(b-a+2)
                    if len(data)!=b-a+1:
                        raise ValueError(f"Part {i} truncated: {len(data)}")
                    temp=dest.with_suffix(".partial");temp.write_bytes(data);temp.replace(dest)
                    break
                except Exception:
                    if retry==4:raise
                    time.sleep(min(10,1+retry))
        with lock:
            done+=1
            if done%64==0:
                seconds=time.monotonic()-start
                print(json.dumps(dict(parts=done,total=(size+chunk-1)//chunk,MiB=round(min(size,done*chunk)/1024**2,1),MiB_per_s=round(done*chunk/1024**2/seconds,2))),flush=True)
    with ThreadPoolExecutor(max_workers=int(os.environ.get("DRAFT_DOWNLOAD_WORKERS","64"))) as pool:
        for future in as_completed([pool.submit(fetch,i) for i in range((size+chunk-1)//chunk)]):
            future.result()
    assembled=target.with_suffix(".assembling");digest=hashlib.sha256()
    with assembled.open("wb") as out:
        for i in range((size+chunk-1)//chunk):
            data=(parts/f"{i:05}.part").read_bytes();digest.update(data);out.write(data)
    if digest.hexdigest()!=expected or assembled.stat().st_size!=size:
        raise RuntimeError("Checkpoint hash/size mismatch; assembled file not accepted")
    assembled.replace(target)
    manifest=dict(repo=REPO,revision=REV,bytes=size,sha256=expected,
                  download_seconds=time.monotonic()-start,method="validated concurrent HTTP ranges")
    (DEST/"download-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    for part in parts.glob("*.part"):
        part.unlink()
    print("DOWNLOAD_VERIFIED "+json.dumps(manifest),flush=True)


if __name__=="__main__":main()
