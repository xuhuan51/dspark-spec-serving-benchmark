"""Uniform verification prefixes for the pinned vLLM 0.29 hybrid-model path.

The checkpoint always proposes its full block. This local adapter selects the
prefix verified by the target; native rejection, state rollback and lifecycle
management remain in vLLM. All active requests use the same policy mode.
"""
import json
import os
from pathlib import Path
import time

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


def pick_prefix(table, active):
    """Use a frozen calibration bucket; never fit on validation responses."""
    points = sorted((int(b), int(k)) for b, k in table.items())
    if not points:
        raise ValueError("Empty prefix table")
    return min(points, key=lambda pair: (abs(pair[0] - active), pair[0]))[1]


class UniformBudgetScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The config declares the capture widths, not the runtime draft length.
        # Keep full-block drafting and shorten only the submitted verification.
        self.dynamic_sd_lookup = None
        self._profile_path = Path(os.environ["DSPARK_BUDGET_PROFILE"])
        self._profile = None
        self._trace = open(os.environ["DSPARK_BUDGET_TRACE"], "a", buffering=1)
        self._pending = {}
        self._step = 0

    def schedule(self, *args, **kwargs):
        started = time.perf_counter()
        configs = [(r.sampling_params.extra_args or {}) for r in self.running]
        modes = {cfg.get("lab_mode", "native") for cfg in configs}
        if len(modes) > 1:
            raise ValueError("Do not mix policy modes in one active batch")
        mode = next(iter(modes), "native")
        prefix = self.num_spec_tokens
        if mode == "prefix":
            prefixes = {int(cfg["lab_k"]) for cfg in configs}
            if len(prefixes) != 1:
                raise ValueError("Verification width must be uniform")
            prefix = prefixes.pop()
        elif mode == "budget":
            if self._profile is None:
                self._profile = json.loads(self._profile_path.read_text())
            prefix = pick_prefix(self._profile["batch_to_prefix"], len(self.running))
        elif mode != "native":
            raise ValueError(mode)
        if not 1 <= prefix <= self.num_spec_tokens:
            raise ValueError(f"Invalid prefix {prefix}")
        for r in self.running:
            # Async placeholders are shared: copy, never mutate in place.
            if mode != "native":
                r.spec_token_ids = r.spec_token_ids[:prefix]
        output = super().schedule(*args, **kwargs)
        self._step += 1
        self._pending[id(output)] = (started, mode, prefix, self._step)
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        started, mode, prefix, step = self._pending.pop(id(scheduler_output))
        proposals = scheduler_output.scheduled_spec_decode_tokens
        # Logical scheduler work, not FLOPs or pure CUDA time.
        row = dict(step=step, mode=mode, prefix=prefix,
                   logical_tokens=scheduler_output.total_num_scheduled_tokens,
                   requests=len(scheduler_output.num_scheduled_tokens),
                   verification_tokens=sum(map(len, proposals.values())),
                   pure_verification=bool(proposals) and
                       len(proposals) == len(scheduler_output.num_scheduled_tokens),
                   turnaround_s=time.perf_counter() - started)
        result = super().update_from_output(scheduler_output, model_runner_output)
        self._trace.write(json.dumps(row) + "\n")
        return result
