# SPDX-License-Identifier: Apache-2.0
# Adapted from ykcombat/sglang@slo_config (eeac514):
# python/sglang/srt/multiplex/multiplexing.py (SchedulerMultiplexMixin).
# Copyright the SGLang project contributors.
"""MuxWise's scheduling loop as a vLLM scheduler.

Each engine step follows one round of MuxWise's event loop: when no
prefill batch is in flight, the waiting requests form one (first come,
first served, up to max_prefill_tokens and the free KV blocks, without
chunking); the decode batch takes one more token slot per request; when a
prefill batch starts or finishes, or the decode batch empties, the stream
group is chosen anew; then the step runs one slice of the prefill batch's
layers beside one decode step. A finished prefill batch joins the decode
batch once its first tokens are back. KV blocks come from vLLM's KV-cache
manager, and a decode batch short of blocks gives up its newest request
(recomputed later, as in vLLM).
"""
import logging
from collections import defaultdict
from dataclasses import dataclass, field

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.request import RequestStatus

from monoserve.baselines.muxwise.config import load, select_group

log = logging.getLogger(__name__)


@dataclass
class MuxStep:
    group: int = 0
    prefill_new: list | None = None      # [(request id, token ids, block ids)]: a new batch
    prefill_temps: list = field(default_factory=list)
    prefill_layers: tuple | None = None  # (start, end): this step's slice
    decode: list = field(default_factory=list)      # [(token, position, block ids)]
    decode_ids: list = field(default_factory=list)
    decode_temps: list = field(default_factory=list)


class MuxWiseScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        conf = (self.vllm_config.additional_config or {}).get("muxwise", {})
        self.mux = load(conf.get("config"))
        self.num_layers = self.vllm_config.model_config.hf_text_config.num_hidden_layers
        self.decode = []      # requests past their prefill, oldest first
        self.split = None     # the prefill batch in flight: dict(reqs, tokens, layer)
        self.group = 0
        self.adjust = False
        self.pending_free = {}   # aborted members of the batch in flight

    # ------------------------------------------------------------------
    def _blocks(self, request):
        return list(self.kv_cache_manager.get_block_ids(request.request_id)[0])

    def _temp(self, request):
        return float(request.sampling_params.temperature or 0.0)

    def _form_prefill_batch(self):
        batch, tokens = [], 0
        while self.waiting:
            r = self.waiting.peek_request()
            n = r.num_tokens - r.num_computed_tokens
            if batch and tokens + n > self.mux.max_prefill_tokens:
                break
            if len(self.running) >= self.max_num_running_reqs:
                break
            if self.kv_cache_manager.allocate_slots(r, n) is None:
                break
            self.waiting.pop_request()
            r.status = RequestStatus.RUNNING
            self.running.append(r)
            batch.append(r)
            tokens += n
        return batch, tokens

    def _preempt(self, r):
        """Back to the waiting queue; its tokens are recomputed later."""
        self.kv_cache_manager.free(r)
        r.status = RequestStatus.PREEMPTED
        r.num_computed_tokens = 0
        r.num_preemptions = getattr(r, "num_preemptions", 0) + 1
        self.running.remove(r)
        self.waiting.prepend_request(r)

    def _grow_decode(self):
        """One more KV slot for every decode request; the newest ones give
        way when the blocks run out."""
        self.decode = [r for r in self.decode if not r.is_finished()]
        i = 0
        while i < len(self.decode):
            r = self.decode[i]
            if self.kv_cache_manager.allocate_slots(r, 1) is not None:
                i += 1
                continue
            victim = self.decode.pop()
            self._preempt(victim)

    def schedule(self, throttle_prefills=False):
        self.current_step += 1
        step = MuxStep()
        if self.split is None and self.waiting:
            batch, tokens = self._form_prefill_batch()
            if batch:
                self.split = {"reqs": batch, "tokens": tokens, "layer": 0}
                step.prefill_new = [(r.request_id, r.all_token_ids[:r.num_tokens], self._blocks(r))
                                    for r in batch]
                step.prefill_temps = [self._temp(r) for r in batch]
                self.adjust = True
        self._grow_decode()
        if self.group > 0 and not self.decode:
            self.adjust = True
        if self.adjust:
            self.group = select_group(self.mux, len(self.decode), self.split is not None,
                                      bool(self.decode))
            self.adjust = False
        step.group = self.group
        sp = self.split
        if sp is not None and sp["layer"] < self.num_layers:
            count = (max(1, self.mux.split_forward_token_budget // sp["tokens"]) if sp["tokens"]
                     else self.num_layers)
            end = min(self.num_layers, sp["layer"] + count)
            step.prefill_layers = (sp["layer"], end)
            sp["layer"] = end
        for r in self.decode:
            step.decode.append((r.all_token_ids[r.num_computed_tokens], r.num_computed_tokens,
                                self._blocks(r)))
            step.decode_ids.append(r.request_id)
            step.decode_temps.append(self._temp(r))
        out = SchedulerOutput.make_empty()
        out.finished_req_ids = self.finished_req_ids
        self.finished_req_ids = set()
        # a positive count keeps the engine-core loop from sleeping while
        # requests are in flight
        out.total_num_scheduled_tokens = 1 if (self.decode or self.split) else 0
        out.muxwise = step
        return out

    def get_grammar_bitmask(self, scheduler_output):
        return None

    # ------------------------------------------------------------------
    def update_from_output(self, scheduler_output, model_runner_output):
        step = getattr(scheduler_output, "muxwise", None)
        tokens = dict(zip(model_runner_output.req_ids, model_runner_output.sampled_token_ids))
        outputs = defaultdict(list)
        stopped = []

        def emit(r, toks, computed):
            if r.is_finished():
                return
            r.num_computed_tokens = computed
            new, stop = self._update_request_with_output(r, toks)
            reason = None
            if stop:
                reason = r.get_finished_reason()
                self._free_request(r)
                stopped.append(r)
            outputs[r.client_index].append(EngineCoreOutput(
                request_id=r.request_id, new_token_ids=new, finish_reason=reason,
                stop_reason=r.stop_reason, events=r.take_events(), trace_headers=r.trace_headers))
            return stop

        if step is not None:
            for rid in step.decode_ids:
                r = self.requests.get(rid)
                if r is not None and rid in tokens:
                    emit(r, tokens[rid], r.num_computed_tokens + 1)
        sp = self.split
        if sp is not None and any(r.request_id in tokens for r in sp["reqs"]):
            for r in sp["reqs"]:
                if r.request_id in self.pending_free:   # aborted while its prefill ran
                    super()._free_blocks(self.pending_free.pop(r.request_id))
                    continue
                if r.is_finished() or r.request_id not in tokens:
                    continue
                if not emit(r, tokens[r.request_id], r.num_tokens):
                    self.decode.append(r)
            self.split = None
            self.adjust = True
        if stopped:
            self.running = remove_all(self.running, set(stopped))
            self.decode = [r for r in self.decode if not r.is_finished()]
        result = {ci: EngineCoreOutputs(outputs=outs) for ci, outs in outputs.items()}
        stats = self.make_stats()
        if stats is not None:
            eco = next(iter(result.values()), None)
            if eco is None:
                result[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats
        return result

    def _free_blocks(self, request):
        # a member of the prefill batch in flight keeps its blocks until the
        # batch's kernels are done
        if self.split is not None and any(r is request for r in self.split["reqs"]):
            self.pending_free[request.request_id] = request
            return
        super()._free_blocks(request)
