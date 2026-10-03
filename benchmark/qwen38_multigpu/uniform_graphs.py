"""Role-specific FULL_DECODE_ONLY graphs for uniform DSpark verification.

Pinned to vLLM 0.29.0. Capture in request units so each verification width
covers the actual maximum batch. This checkpoint samples from the anchor and
keeps draft width 7 while the target verifies widths 2/3/5/8. All modes use
this same pool. This does not enable unsupported varlen GDN graphs.
"""
from collections import Counter


def install_plan():
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager, BatchExecutionDescriptor
    if getattr(CudaGraphManager, "_dspark_uniform_lab_plan", False):
        return
    original = CudaGraphManager._init_candidates
    def planned(self):
        spec = self.vllm_config.speculative_config
        if (not spec or spec.method != "dspark" or spec.num_speculative_tokens != 7
                or self.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY
                or self.lora_capture_cases != [0] or self.varlen_decode):
            return original(self)
        role = "draft" if type(self).__name__ == "DFlashCudaGraphManager" else "target"
        expected = 7 if role == "draft" else 8
        if self.decode_query_len != expected:
            raise ValueError(f"Unexpected {role} query width {self.decode_query_len}")
        widths = [expected] if role == "draft" else [2, 3, 5, 8]
        counts = sorted({n for n in [1, 2, 4, 6, 8, 12, 16, 24, 32, self.max_num_reqs] if n <= self.max_num_reqs})
        limit = self.compilation_config.max_cudagraph_capture_size
        if limit < max(counts) * max(widths):
            raise ValueError("Capture budget does not cover maximum request batch")
        descs = [BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=n*q,
                                         num_reqs=n, uniform_token_count=q)
                 for q in widths for n in counts]
        ascending = sorted(descs, key=lambda d: (d.num_tokens, d.uniform_token_count))
        self._candidates = {(n, 0): [d for d in ascending if d.num_tokens >= n]
                            for n in range(max(d.num_tokens for d in descs) + 1)}
        self._capture_descs = {CUDAGraphMode.FULL: sorted(descs, key=lambda d: d.num_tokens, reverse=True)}
        self._lab_plan = dict(role=role, widths=widths, requests=counts, descriptors=len(descs))
    CudaGraphManager._init_candidates = planned
    CudaGraphManager._dspark_uniform_lab_plan = True


install_plan()


class UniformGraphWorker:
    def budget_install_metrics(self):
        from vllm.config.compilation import CUDAGraphMode
        from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, _is_compatible
        runner = self.model_runner
        self._budget_graph_counts = Counter()
        self._budget_graph_policy = "full"
        self._budget_original_cache = {}
        self._budget_managers = {"target": runner.cudagraph_manager,
                                "draft": runner.speculator.query_cudagraph_manager}
        for role, manager in self._budget_managers.items():
            original = manager.dispatch
            def dispatch(num_reqs, num_tokens, uniform_token_count, num_active_loras,
                         max_query_len=None, _original=original, _role=role, _manager=manager):
                desc = _original(num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len)
                if self._budget_graph_policy == "original":
                    key = (_role, num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len)
                    if key not in self._budget_original_cache:
                        candidates = [d for d in _manager.graphs
                                      if d.num_reqs in [1, 2, 4, 8, 16, 32]
                                      and d.uniform_token_count == _manager.decode_query_len
                                      and _is_compatible(d, num_reqs, num_tokens, uniform_token_count,
                                                         num_active_loras, max_query_len)]
                        self._budget_original_cache[key] = min(candidates, key=lambda d: d.num_tokens) if candidates else BatchExecutionDescriptor(
                            cg_mode=CUDAGraphMode.NONE, num_tokens=num_tokens, num_reqs=num_reqs,
                            num_active_loras=num_active_loras)
                    desc = self._budget_original_cache[key]
                self._budget_graph_counts[(_role, str(desc.cg_mode), uniform_token_count,
                                           num_reqs, desc.num_reqs, desc.num_tokens)] += 1
                return desc
            manager.dispatch = dispatch
        self._budget_full_mode = str(CUDAGraphMode.FULL)
        return self.budget_metrics()

    def budget_metrics(self):
        return dict(policy=self._budget_graph_policy,
                    graphs={role: len(manager.graphs) for role, manager in self._budget_managers.items()},
                    plans={role: getattr(manager, "_lab_plan", None) for role, manager in self._budget_managers.items()},
                    dispatch_fields=["role", "mode", "query_width", "logical_requests", "captured_requests", "captured_tokens", "calls"],
                    dispatch_counts=[[*key, n] for key, n in self._budget_graph_counts.items()])

    def budget_reset_metrics(self):
        self._budget_graph_counts.clear()
        return self.budget_metrics()

    def budget_set_graph_policy(self, policy):
        if policy not in ("original", "full"):
            raise ValueError(policy)
        self._budget_graph_policy = policy
        return self.budget_metrics()
