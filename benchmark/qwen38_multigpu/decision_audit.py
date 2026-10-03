"""Opt-in target sampling diagnostics; never enabled during performance rows."""
import json
from pathlib import Path


class DecisionAuditWorker:
    def decision_audit_start(self, path):
        import torch
        from vllm.v1.worker.gpu.sample.sampler import Sampler
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

        if getattr(self, "_decision_audit_active", False):
            raise RuntimeError("Decision audit already enabled")
        rank = self.rank
        output = Path(f"{path}.rank{rank}.jsonl")
        if output.exists():
            raise FileExistsError(output)
        stream = output.open("x", buffering=1)
        runner = self.model_runner
        originals = {}

        def wrapper(original, selected, kind):
            def audited(sampler, logits, input_batch, *args, **kwargs):
                result = original(sampler, logits, input_batch, *args, **kwargs)
                if sampler is not selected or not self._decision_audit_active:
                    return result
                # Only selected target sampler calls, not the draft sampler.
                values, indices = torch.topk(logits, 8, dim=-1)
                values = values.float().cpu().tolist()
                indices = indices.cpu().tolist()
                argmax = logits.argmax(dim=-1).cpu().tolist()
                sampled = result.sampled_token_ids.cpu().tolist()
                num_sampled = result.num_sampled.cpu().tolist()
                positions = input_batch.positions[input_batch.logits_indices].cpu().tolist()
                boundaries = input_batch.cu_num_logits_np.tolist()
                for i, req_id in enumerate(input_batch.req_ids):
                    lo, hi = boundaries[i:i+2]
                    req_index = int(input_batch.idx_mapping_np[i])
                    prompt_len = int(runner.req_states.prompt_len.np[req_index])
                    emitted = []
                    for j in range(min(int(num_sampled[i]), hi-lo)):
                        token = int(sampled[i][j])
                        if token < 0:
                            continue
                        row = lo+j
                        emitted.append(dict(output_index=int(positions[row])+1-prompt_len,
                                            absolute_position=int(positions[row])+1,
                                            token=token, argmax=int(argmax[row]),
                                            top_tokens=indices[row], top_logits=values[row]))
                    stream.write(json.dumps(dict(request_id=req_id, rank=rank, sampler=kind,
                                                 prompt_tokens=prompt_len,
                                                 num_reqs=input_batch.num_reqs,
                                                 num_tokens=input_batch.num_tokens,
                                                 captured_tokens=input_batch.num_tokens_after_padding,
                                                 num_logits=hi-lo, logits_dtype=str(logits.dtype),
                                                 emitted=emitted)) + "\n")
                return result
            return audited

        self._decision_audit_active = True
        for cls, selected, name in [(Sampler, runner.sampler, "ar"),
                                    (RejectionSampler, runner.rejection_sampler, "verify")]:
            if selected is None:
                continue
            originals[cls] = cls.__call__
            cls.__call__ = wrapper(cls.__call__, selected, name)
        self._decision_audit_originals = originals
        self._decision_audit_stream = stream
        return dict(rank=rank, output=str(output), performance_safe=False)

    def decision_audit_stop(self):
        for cls, original in self._decision_audit_originals.items():
            cls.__call__ = original
        self._decision_audit_active = False
        self._decision_audit_stream.close()
        return dict(rank=self.rank, stopped=True)
