"""vLLM's scheduler, replaced by MonoServe's admission.

vLLM still receives requests, checks stop conditions (EOS, stop tokens,
max_tokens, max_model_len), and emits outputs; which requests run, in which
lane, on how many SMs, with how many link credits and which kernel form, is
decided by the engine's admission at every arrival and completion. A
request enters the engine as soon as vLLM hands it over, and vLLM counts it
as running from then on.

A request may bring its SLO as extra arguments (vllm_xargs in the OpenAI
API): ttft_slo and tpot_slo, in seconds. Without them the engine targets
alpha times the request's estimated solo latency. Sampling uses the
temperature only.
"""
import logging
import time
from collections import defaultdict

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.request import RequestStatus

from monoserve import vllm as registry
from monoserve.runtime.kinds import PAGE

log = logging.getLogger(__name__)
REJECT_LOG_INTERVAL_S = 10.0


class VllmPages:
    """The engine's KV pages from vLLM's KV-cache manager: a request's
    pages are its blocks of 64 tokens. The engine returns them once the
    device no longer uses them, and a finished request's blocks go back to
    vLLM only then."""

    def __init__(self, sched):
        self.sched = sched

    def grow(self, rid, pages, n):
        if n <= len(pages):
            return pages
        s = self.sched
        request = s.requests[rid]
        if s.kv_cache_manager.allocate_slots(request, n * PAGE - request.num_computed_tokens) is None:
            raise MemoryError("out of KV-cache blocks")
        s.engine_pages.add(rid)
        return list(s.kv_cache_manager.get_block_ids(rid)[0])

    def release(self, rid, pages):
        self.sched.engine_released(rid)


class MonoServeScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = registry.engine()
        self.engine.pool = VllmPages(self)
        self.engine_pages = set()   # requests whose pages the engine still holds
        self.deferred = {}          # finished requests waiting for the engine's release
        self.warned = False
        self.rejections, self.reject_logged = 0, None

    def _free_blocks(self, request):
        if request.request_id in self.engine_pages:   # the device may still use them
            self.deferred[request.request_id] = request
            return
        super()._free_blocks(request)

    def engine_released(self, rid):
        self.engine_pages.discard(rid)
        request = self.deferred.pop(rid, None)
        if request is not None:
            super()._free_blocks(request)

    def _engine_finish(self, rid):
        """End a request in the engine. The admission event may reject queued
        requests, which vLLM still counts as running: they are reported as
        finished (returned, and kept in the registry for the next update)."""
        finished = registry.engine_finished()
        rejected = []
        for other, _, reason in self.engine.finish(rid):
            finished[other] = reason
            rejected.append(other)
        return rejected

    def add_request(self, request):
        super().add_request(request)
        sp = request.sampling_params
        if not self.warned and ((sp.top_k or 0) > 0 or (sp.top_p or 1.0) < 1.0 or
                                sp.presence_penalty or sp.frequency_penalty):
            log.warning("MonoServe samples with the temperature only; top-k, top-p, and "
                        "penalties are ignored")
            self.warned = True
        xargs = sp.extra_args or {}
        # arrival on the engine's clock: vLLM stamps it with wall-clock time
        arrival = self.engine.now() - max(0.0, time.time() - request.arrival_time)
        rejected = self.engine.add_request(
            request.request_id, request.prompt_token_ids, max_tokens=request.max_tokens,
            temperature=sp.temperature, ttft=xargs.get("ttft_slo"), tpot=xargs.get("tpot_slo"),
            arrival=arrival)
        for rid, _, reason in rejected:
            registry.engine_finished()[rid] = reason
        # the engine now owns the request: vLLM counts it as running
        self.waiting.remove_requests([request])
        request.status = RequestStatus.RUNNING
        self.running.append(request)

    def schedule(self, throttle_prefills=False):
        self.current_step += 1
        out = SchedulerOutput.make_empty()
        out.finished_req_ids = self.finished_req_ids
        self.finished_req_ids = set()
        # a positive count keeps the engine-core loop from sleeping while
        # requests are in flight; nothing else reads it
        out.total_num_scheduled_tokens = 1 if self.running else 0
        return out

    def get_grammar_bitmask(self, scheduler_output):
        return None

    def update_from_output(self, scheduler_output, model_runner_output):
        outputs = defaultdict(list)
        stopped = set()
        finished = registry.engine_finished()
        tokens = dict(zip(model_runner_output.req_ids, model_runner_output.sampled_token_ids))
        work = list(tokens) + [r for r in finished if r not in tokens]
        for rid in work:   # grows when a finish lets admission reject queued requests
            request = self.requests.get(rid)
            if request is None or request.is_finished():
                finished.pop(rid, None)
                continue
            new, stop = [], False
            if tokens.get(rid):
                new, stop = self._update_request_with_output(request, list(tokens[rid]))
            reason = finished.pop(rid, None)
            if not stop and reason is not None:
                request.status = (RequestStatus.FINISHED_ABORTED if reason == "rejected"
                                  else RequestStatus.FINISHED_LENGTH_CAPPED)
                if reason == "rejected":
                    self.note_rejection()
                stop = True
            finish_reason = None
            if stop:
                finish_reason = request.get_finished_reason()
                self._free_request(request)
                stopped.add(request)
                work += self._engine_finish(rid)
            if new or stop:
                outputs[request.client_index].append(EngineCoreOutput(
                    request_id=rid, new_token_ids=new, finish_reason=finish_reason,
                    stop_reason=request.stop_reason, events=request.take_events(),
                    trace_headers=request.trace_headers))
        if stopped:
            self.running = remove_all(self.running, stopped)
        result = {ci: EngineCoreOutputs(outputs=outs) for ci, outs in outputs.items()}
        stats = self.make_stats()
        if stats is not None:
            eco = next(iter(result.values()), None)
            if eco is None:
                result[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats
        return result

    def note_rejection(self):
        """Admission rejects the requests whose targets no plan can meet;
        vLLM reports them as aborted. Logged at most once per interval."""
        self.rejections += 1
        now = time.monotonic()
        if self.reject_logged is None or now - self.reject_logged >= REJECT_LOG_INTERVAL_S:
            log.warning("admission rejected %d request(s) whose SLO targets no plan can meet "
                        "(finish reason 'abort'); targets come from vllm_xargs ttft_slo and "
                        "tpot_slo, or alpha times the solo latency", self.rejections)
            self.rejections, self.reject_logged = 0, now

    def finish_requests(self, request_ids, finished_status):
        done = super().finish_requests(request_ids, finished_status)
        for request in done:
            self._engine_finish(request.request_id)
        return done
