"""CPU check of actual vLLM dispatch and original-bucket ablation shapes."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import uniform_graphs
from uniform_graphs import UniformGraphWorker
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.config.compilation import CUDAGraphMode


def manager(cls):
    m=cls.__new__(cls)
    m.vllm_config=SimpleNamespace(speculative_config=SimpleNamespace(method="dspark",num_speculative_tokens=7))
    m.cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY;m.lora_capture_cases=[0];m.varlen_decode=False
    m.decode_query_len=7 if cls.__name__ == "DFlashCudaGraphManager" else 8;m.max_num_reqs=32
    m.compilation_config=SimpleNamespace(max_cudagraph_capture_size=256)
    m._lora_dispatch_map={0:0};m._max_lora_case=0
    m._init_candidates();m._graphs_captured=True
    m.graphs={d:None for d in m._capture_descs[CUDAGraphMode.FULL]}
    return m


def main():
    target=manager(CudaGraphManager)
    draft=manager(type("DFlashCudaGraphManager",(CudaGraphManager,),{}))
    checked=0
    for m,widths in [(target,[2,3,5,8]),(draft,[7])]:
        for n in range(1,33):
            for q in widths:
                d=m.dispatch(n,n*q,q,0,max_query_len=q)
                assert d.cg_mode==CUDAGraphMode.FULL and d.num_reqs>=n and d.uniform_token_count==q
                checked+=1
    w=UniformGraphWorker()
    w.model_runner=SimpleNamespace(cudagraph_manager=target,speculator=SimpleNamespace(query_cudagraph_manager=draft))
    w.budget_install_metrics();w.budget_set_graph_policy("original")
    for m in [target,draft]:
        for n in range(1,33):
            q=m.decode_query_len
            d=m.dispatch(n,n*q,q,0,max_query_len=q)
            expected=min(b for b in [1,2,4,8,16,32] if b>=n)
            assert d.num_reqs==expected and d.uniform_token_count==q
            checked+=1
    w.budget_set_graph_policy("full")
    assert target.dispatch(10,80,8,0,max_query_len=8).num_reqs==12
    assert draft.dispatch(10,70,7,0,max_query_len=7).num_reqs==12
    json.dumps(w.budget_metrics())  # RPC result must remain JSON serializable.
    here=Path(__file__).resolve().parent
    result=dict(runtime="vLLM 0.29.0",coverage_checks=checked,ablation_examples=2,
                target_plan=target._lab_plan,draft_plan=draft._lab_plan,
                source_sha256=hashlib.sha256((here/"uniform_graphs.py").read_bytes()).hexdigest(),
                scope="CPU graph descriptor and dispatch checks; not GPU numerical correctness")
    output=here/"runs/graph-coverage.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
