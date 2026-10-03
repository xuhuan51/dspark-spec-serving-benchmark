"""Verify the recorded evidence and recompute paired results without a GPU."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from analyze_capacity import analyze as analyze_capacity
from analyze_closure import analyze as analyze_closure
from summarize import summarize
from summarize_budget import summary as summarize_budget

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "results/qwen38_multigpu_20261003"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def relocate(value, paths):
    """Relocate recorded filenames; preserve measurements and trace byte offsets."""
    if isinstance(value, dict):
        return {key: relocate(item, paths) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate(item, paths) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        target = paths.get(Path(value).name)
        if target is not None:
            return str(target.resolve())
    return value


def comparable(value):
    # The analyzers' source path changes when evidence is moved to this checkout.
    if isinstance(value, dict):
        return {key: comparable(item) for key, item in value.items() if key != "source"}
    if isinstance(value, list):
        return [comparable(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/qwen38-evidence")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError("Choose a fresh output directory to preserve earlier evidence")
    manifest = json.loads((args.bundle / "bundle-manifest.json").read_text())
    if digest((args.bundle / "workload.json").read_bytes()) != manifest["workload_sha256"]:
        raise RuntimeError("Frozen workload hash differs")
    profile = ROOT / "configs/qwen38_uniform_budget.json"
    if digest(profile.read_bytes()) != manifest["frozen_budget_profile_sha256"]:
        raise RuntimeError("Frozen calibration hash differs")
    paths = {}
    for name, record in manifest["archives"].items():
        target_name = record["uncompressed_file"]
        if Path(name).name != name or Path(target_name).name != target_name:
            raise ValueError("Archive filenames must be flat")
        archive = (args.bundle / "records" / name).read_bytes()
        if len(archive) != record["archive_bytes"] or digest(archive) != record["archive_sha256"]:
            raise RuntimeError(f"Archive hash/size mismatch: {name}")
        raw = gzip.decompress(archive)
        if len(raw) != record["uncompressed_bytes"] or digest(raw) != record["uncompressed_sha256"]:
            raise RuntimeError(f"Original record hash/size mismatch: {name}")
        target = args.output / target_name
        target.write_bytes(raw)
        paths[target_name] = target
    # JSONL records stay byte-identical, so recorded capacity window offsets work.
    for name, target in paths.items():
        if name.endswith(".json"):
            data = relocate(json.loads(target.read_text()), paths)
            target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    provenance = {
        "scope": "Original archives verified before path relocation. Only absolute references to bundled files change; JSONL bytes and measured values are preserved.",
        "manifest_sha256": digest((args.bundle / "bundle-manifest.json").read_bytes()),
        "archives_verified": len(paths),
    }
    (args.output / "relocation-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    jobs = [
        ("native-paired-resumed.json", summarize),
        ("budget-v2.json", summarize_budget),
        ("threeway-resumed.json", analyze_closure),
        ("capacity-final.json", analyze_capacity),
    ]
    for name, analyzer in jobs:
        analyzer(paths[name])
        computed = paths[name].with_suffix(".summary.json")
        expected = args.bundle / computed.name
        if comparable(json.loads(computed.read_text())) != comparable(json.loads(expected.read_text())):
            raise RuntimeError(f"Recomputed summary differs: {computed.name}")
    closure = json.loads((args.output / "threeway-resumed.summary.json").read_text())
    capacity = json.loads((args.output / "capacity-final.summary.json").read_text())
    coverage = closure["decision_coverage"] + capacity["decision_coverage"]
    checked = sum(row["checked_tokens"] for row in coverage)
    if checked != 13056 or any(row["non_argmax_tokens"] or row["non_maximum_tokens"] for row in coverage):
        raise RuntimeError("Target decision validation changed")
    print(json.dumps({"verified_archives": len(paths), "matching_summaries": len(jobs),
                      "target_decisions_checked": checked, "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
