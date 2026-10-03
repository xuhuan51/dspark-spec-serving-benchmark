"""Opt-in Nsight phase scopes; no numerical or scheduling changes."""
import functools


class PhaseWorker:
    def lab_install_phases(self):
        import torch
        runner = self.model_runner
        manager = runner.cudagraph_manager
        original = manager.run_fullgraph
        @functools.wraps(original)
        def target(desc):
            with torch.cuda.nvtx.range(f"dspark_lab/target_graph/requests={desc.num_reqs}/width={desc.uniform_token_count}"):
                return original(desc)
        manager.run_fullgraph = target
        # Target graph replay covers the backbone, not its LM head/sampler.
        # Keep those scopes separate to avoid comparing unlike phase totals.
        forward = runner.model.forward
        @functools.wraps(forward)
        def target_eager(*args, **kwargs):
            with torch.cuda.nvtx.range("dspark_lab/target_eager"):
                return forward(*args, **kwargs)
        runner.model.forward = target_eager
        sample = runner.sample
        @functools.wraps(sample)
        def target_sample(*args, **kwargs):
            with torch.cuda.nvtx.range("dspark_lab/target_sample"):
                return sample(*args, **kwargs)
        runner.sample = target_sample
        for name in ["compute_logits", "compute_logits_local"]:
            if not hasattr(runner.model, name):
                continue
            method = getattr(runner.model, name)
            @functools.wraps(method)
            def target_logits(*args, _method=method, **kwargs):
                with torch.cuda.nvtx.range("dspark_lab/target_logits"):
                    return _method(*args, **kwargs)
            setattr(runner.model, name, target_logits)
        speculator = getattr(runner, "speculator", None)
        if speculator:
            propose = speculator.propose
            @functools.wraps(propose)
            def draft(*args, **kwargs):
                with torch.cuda.nvtx.range("dspark_lab/draft_total"):
                    return propose(*args, **kwargs)
            speculator.propose = draft
            graph = speculator.query_cudagraph_manager
            replay = graph.run_fullgraph
            @functools.wraps(replay)
            def draft_graph(desc):
                with torch.cuda.nvtx.range(f"dspark_lab/draft_graph/requests={desc.num_reqs}/width={desc.uniform_token_count}"):
                    return replay(desc)
            graph.run_fullgraph = draft_graph
        return dict(rank=self.rank, has_draft=bool(speculator))
