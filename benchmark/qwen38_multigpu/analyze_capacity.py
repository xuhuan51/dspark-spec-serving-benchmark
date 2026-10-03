"""Summarize measured allocation denials and an AR batch-size control."""
import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_closure import decisions, compare


def analyze(path):
    data = json.loads(path.read_text())
    if not data["complete"] or data.get("failure"):
        raise RuntimeError("Capacity probe must be complete")
    summary = []
    diagnostic = dict(output_path=str(path.resolve()),diagnostics=[])
    for run in data["runs"]:
        lines=[json.loads(line) for line in Path(run["trace"]).read_text().splitlines()]
        layout=next(row for row in lines if row["event"]=="layout")
        with Path(run["trace"]).open("rb") as stream:
            stream.seek(run["trace_start"])
            rows=[json.loads(line) for line in stream.read(run["trace_end"]-run["trace_start"]).splitlines()]
        rows=[r for r in rows if r["event"]=="allocate"]
        failed=[r for r in rows if not r["success"]]
        first={}
        for row in rows:
            if row["success"] and row["request_id"] not in first:
                first[row["request_id"]]=row
        first_admissions=[]
        for req_id,row in first.items():
            query=row["queries"][0]
            first_admissions.append(dict(request_id=req_id,required_blocks=query["required_blocks"],
                                         components=query["components"],free_before=row["free_before"],
                                         blocks_consumed=row["free_before"]-row["free_after"]))
        item=dict(mode=run["mode"],layout=layout,allocation_attempts=len(rows),denials=len(failed),
                  denied_requests=len({r['request_id'] for r in failed}),first_admissions=first_admissions,
                  denial_examples=failed[:8],
                  admission_component_histogram={kind:dict(Counter(c['blocks'] for row in first_admissions
                                                                   for c in row['components'] if c['manager']==kind))
                                                 for kind in {c['manager'] for row in first_admissions for c in row['components']}})
        summary.append(item)
        measurement={**run['measurement'],"repeat":0}
        diag=dict(mode=run['mode'],rows=[measurement])
        if 'serial_control' in run:
            diag['rows'].append({**run['serial_control'],"repeat":0})
        diagnostic['diagnostics'].append(diag)
    entries,coverage=decisions(diagnostic)
    batch=compare(entries,('ar',1,0),('ar',16,0),'AR serial/concurrent, 16 prompts')
    native=compare(entries,('ar',16,0),('fixed',16,0),'AR/native, c=16 capacity probe')
    output=dict(source=str(path.resolve()),complete=True,capacity=summary,decision_coverage=coverage,
                autoregressive_batch_control=batch,native_comparison=native,
                scope="Actual allocation component counts, no isolated GPU timing of state save/rollback. Diagnostics do not establish universal distribution preservation.")
    path.with_suffix('.summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(capacity=[{k:v for k,v in r.items() if k not in ['first_admissions','denial_examples']} for r in summary],
                         decision_coverage=coverage,autoregressive_batch_control={k:v for k,v in batch.items() if k!='pairs'}),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('result',type=Path)
    analyze(parser.parse_args().result)
