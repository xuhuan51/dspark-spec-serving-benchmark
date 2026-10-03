"""Record actual cache allocation decisions, without changing their results.

Used in separate capacity diagnostics only, never for throughput claims.
"""
import json
import os

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class AdmissionAuditScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._admission_stream = open(os.environ["DSPARK_ADMISSION_TRACE"], "x", buffering=1)
        coordinator = self.kv_cache_manager.coordinator
        groups = []
        for index, manager in enumerate(coordinator.single_type_managers):
            spec = manager.kv_cache_spec
            item = dict(group=index, manager=type(manager).__name__, spec=type(spec).__name__,
                        block_size=manager.block_size, page_size_bytes=spec.page_size_bytes,
                        layers=len(self.kv_cache_manager.kv_cache_config.kv_cache_groups[index].layer_names))
            for field in ["num_speculative_blocks", "mamba_cache_mode", "state_shapes", "state_dtypes"]:
                value = getattr(spec, field, getattr(manager, field, None))
                if value is not None:
                    item[field] = str(value) if field == "state_dtypes" else value
            groups.append(item)
            original = manager.get_num_blocks_to_allocate
            def counted(*args, _original=original, _index=index, _manager=manager, **kwargs):
                count = _original(*args, **kwargs)
                self._admission_components.append(dict(group=_index, manager=type(_manager).__name__,
                                                        blocks=int(count)))
                return count
            manager.get_num_blocks_to_allocate = counted
        self._admission_stream.write(json.dumps(dict(event="layout", num_blocks=self.kv_cache_manager.kv_cache_config.num_blocks,
                                                    watermark_blocks=self.kv_cache_manager.watermark_blocks,
                                                    num_spec_tokens=self.num_spec_tokens, groups=groups)) + "\n")
        self._admission_components = []
        self._admission_queries = []
        original_coordinator = coordinator.get_num_blocks_to_allocate
        def coordinator_counted(*args, **kwargs):
            self._admission_components = []
            total = original_coordinator(*args, **kwargs)
            self._admission_queries.append(dict(required_blocks=int(total),
                                                 admission_cap=kwargs.get("apply_admission_cap", False),
                                                 components=self._admission_components.copy()))
            return total
        coordinator.get_num_blocks_to_allocate = coordinator_counted
        original_allocate = self.kv_cache_manager.allocate_slots
        def allocated(request, *args, **kwargs):
            self._admission_queries = []
            free_before = self.kv_cache_manager.block_pool.get_num_free_blocks()
            status = str(request.status)
            result = original_allocate(request, *args, **kwargs)
            self._admission_stream.write(json.dumps(dict(event="allocate", request_id=request.request_id,
                                                         status=status, running=len(self.running), waiting=len(self.waiting),
                                                         free_before=free_before,
                                                         free_after=self.kv_cache_manager.block_pool.get_num_free_blocks(),
                                                         computed_tokens=request.num_computed_tokens,
                                                         prompt_tokens=request.num_prompt_tokens,
                                                         success=result is not None, queries=self._admission_queries)) + "\n")
            return result
        self.kv_cache_manager.allocate_slots = allocated
