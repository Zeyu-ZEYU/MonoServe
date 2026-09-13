# SPDX-License-Identifier: Apache-2.0
# The split prefill follows forward_split_prefill of the models in
# ykcombat/sglang@slo_config (eeac514), and the per-division decode graphs
# its cuda_graph_runner.py. Copyright the SGLang project contributors.
"""MuxWise's execution on vLLM's model.

Each engine step launches the current slice of the prefill batch (a range
of layers) on the prefill stream of the current group and one decode step
on its decode stream, so the two run at once on their partitions. Decode
replays a CUDA graph captured for the group and the batch size (padded to
the next captured size); the prefill runs eagerly and keeps its hidden
states between slices. Attention metadata is built by vLLM's attention
backend from the requests' KV-cache blocks.
"""
import numpy as np
import torch

CAPTURE_SIZES = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256)
PAD_SLOT = -1   # slot-mapping value whose KV write the kernels skip


class StreamGroups:
    """One (prefill, decode) stream pair per group: ordinary streams on
    every SM for group 0 (prefill alone) and the last group (decode alone),
    two green contexts for every division in between."""

    def __init__(self, divisions, device):
        from monoserve import _C
        S = torch.cuda.get_device_properties(device).multi_processor_count
        self.parts = []   # kept alive: the streams belong to them
        self.groups = [(torch.cuda.Stream(device), torch.cuda.Stream(device))]
        self.sms = [(S, 0)]
        for p, d in divisions:
            parts = _C.GreenPartitions()
            granted = parts.create([p, d])
            self.parts.append(parts)
            self.groups.append((torch.cuda.ExternalStream(parts.stream(0), device=device),
                                torch.cuda.ExternalStream(parts.stream(1), device=device)))
            self.sms.append(tuple(granted))
        self.groups.append((torch.cuda.Stream(device), torch.cuda.Stream(device)))
        self.sms.append((0, S))


class PrefillBatch:
    """A prefill batch between its slices: attention metadata, positions,
    and the hidden states after the layers run so far."""

    def __init__(self, req_ids, temps):
        self.req_ids, self.temps = req_ids, temps
        self.md = self.slots = self.positions = self.hidden = self.residual = None
        self.last_rows = None
        self.tokens = None      # pinned host tensor of the sampled first tokens
        self.event = None       # recorded after the sampling

    def done(self):
        return self.event is not None and self.event.query()


class MuxRuntime:
    def __init__(self, runner, vllm_config, cfg, capture=True, capture_sizes=CAPTURE_SIZES):
        from monoserve.baselines.muxwise.config import divisions
        self.runner, self.vc, self.cfg = runner, vllm_config, cfg
        self.model = runner.model
        self.inner = self.model.model
        self.layers = self.inner.layers
        self.num_layers = len(self.layers)
        self.device = runner.device
        self.block = vllm_config.cache_config.block_size
        self.max_len = runner.max_model_len
        self.max_blocks = -(-self.max_len // self.block)
        self.attn_groups = [g for groups in runner.attn_groups for g in groups]
        self.layer_names = [n for g in self.attn_groups for n in g.layer_names]
        props = torch.cuda.get_device_properties(self.device)
        self.S = props.multi_processor_count
        self.streams = StreamGroups(divisions(cfg, self.S, (props.major, props.minor)), self.device)
        self.cur = 0
        self.prefill = None
        max_bs = max(s for s in capture_sizes if s <= runner.max_num_reqs) if capture else 0
        self.capture_sizes = [s for s in capture_sizes if s <= max_bs]
        self._buffers(max(max_bs, 1))
        self.graphs = {}
        if capture:
            self.capture()

    # ------------------------------------------------------------------
    # attention metadata
    # ------------------------------------------------------------------
    def metadata(self, qsl, qsl_cpu, seq, seq_cpu, bt, slots, num_tokens, max_query, max_seq,
                 fast=False):
        """Per-layer attention metadata and slot mappings for vLLM's
        forward context."""
        from vllm.v1.attention.backend import CommonAttentionMetadata
        query_lens = qsl_cpu[1:] - qsl_cpu[:-1]
        cm = CommonAttentionMetadata(
            query_start_loc=qsl, query_start_loc_cpu=qsl_cpu, seq_lens=seq,
            num_reqs=int(seq_cpu.numel()), num_actual_tokens=num_tokens, max_query_len=max_query,
            max_seq_len=max_seq, block_table_tensor=bt, slot_mapping=slots, causal=True,
            _seq_lens_cpu=seq_cpu, _num_computed_tokens_cpu=seq_cpu - query_lens)
        md = {}
        for g in self.attn_groups:
            m = g.metadata_builders[0].build(0, cm, fast_build=fast)
            for name in g.layer_names:
                md[name] = m
        return md, {name: slots for name in self.layer_names}

    def forward_context(self, md, slots, num_tokens):
        from vllm.forward_context import set_forward_context
        return set_forward_context(md, self.vc, num_tokens=num_tokens, slot_mapping=slots)

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------
    def _buffers(self, max_bs):
        dev, i32 = self.device, torch.int32
        self.ids = torch.zeros(max_bs, dtype=torch.int64, device=dev)
        self.pos = torch.zeros(max_bs, dtype=torch.int64, device=dev)
        self.qsl = torch.arange(max_bs + 1, dtype=i32, device=dev)
        self.qsl_cpu = torch.arange(max_bs + 1, dtype=i32)
        self.seq = torch.ones(max_bs, dtype=i32, device=dev)
        self.bt = torch.zeros(max_bs, self.max_blocks, dtype=i32, device=dev)
        self.slots = torch.full((max_bs,), PAD_SLOT, dtype=torch.int64, device=dev)
        # pinned staging for one step's rows
        self.h_ids = torch.zeros(max_bs, dtype=torch.int64, pin_memory=True)
        self.h_pos = torch.zeros(max_bs, dtype=torch.int64, pin_memory=True)
        self.h_seq = torch.ones(max_bs, dtype=i32, pin_memory=True)
        self.h_bt = torch.zeros(max_bs, self.max_blocks, dtype=i32, pin_memory=True)
        self.h_slots = torch.full((max_bs,), PAD_SLOT, dtype=torch.int64, pin_memory=True)

    def _stage(self, reqs, bs):
        """Rows of a decode batch into the static buffers, padded to bs rows
        that attend one position of the null block and write nothing."""
        n = len(reqs)
        ids, pos, seq = self.h_ids.numpy(), self.h_pos.numpy(), self.h_seq.numpy()
        bt, slots = self.h_bt.numpy(), self.h_slots.numpy()
        bt[:bs] = 0
        for i, (token, position, blocks) in enumerate(reqs):
            ids[i], pos[i], seq[i] = token, position, position + 1
            bt[i, :len(blocks)] = blocks
            slots[i] = blocks[position // self.block] * self.block + position % self.block
        ids[n:bs], pos[n:bs], seq[n:bs], slots[n:bs] = 0, 0, 1, PAD_SLOT
        for dst, src in ((self.ids, self.h_ids), (self.pos, self.h_pos), (self.seq, self.h_seq),
                         (self.bt, self.h_bt), (self.slots, self.h_slots)):
            dst[:bs].copy_(src[:bs], non_blocking=True)

    def _decode_metadata(self, bs):
        """Metadata over the static buffers (built outside any capture; the
        longest sequence is bounded by the model length)."""
        return self.metadata(self.qsl[:bs + 1], self.qsl_cpu[:bs + 1], self.seq[:bs],
                             self.h_seq[:bs].clone(), self.bt[:bs], self.slots[:bs], bs, 1,
                             self.max_len, fast=True)

    def _decode_run(self, bs, md, slots):
        with self.forward_context(md, slots, bs):
            hidden = self.model(input_ids=self.ids[:bs], positions=self.pos[:bs])
            return self.model.compute_logits(hidden)

    def capture(self):
        """Decode graphs for every group that runs decode and every size,
        each captured on the group's decode stream. A graph keeps its
        metadata, whose tensors it reads."""
        pool = torch.cuda.graph_pool_handle()
        for gi, (_, ds) in enumerate(self.streams.groups):
            if gi == 0:
                continue
            for bs in reversed(self.capture_sizes):
                with torch.cuda.stream(ds):
                    self._stage([], bs)
                    md, slots = self._decode_metadata(bs)
                    self._decode_run(bs, md, slots)   # warm-up
                ds.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=ds):
                    logits = self._decode_run(bs, md, slots)
                self.graphs[(gi, bs)] = (graph, logits, (md, slots))
        torch.cuda.synchronize()

    def decode(self, reqs, temps):
        """One decode step of reqs [(token, position, block ids)] on the
        current group's decode stream; the sampled tokens stay on the device."""
        n = len(reqs)
        bs = next((s for s in self.capture_sizes if s >= n), None)
        if bs is not None and (self.cur, bs) in self.graphs:
            self._stage(reqs, bs)
            graph, logits, _ = self.graphs[(self.cur, bs)]
            graph.replay()
            return sample(logits[:n], temps)
        # beyond the captured sizes: eagerly, on fresh buffers
        return sample(self._decode_eager(reqs), temps)

    def _decode_eager(self, reqs):
        n = len(reqs)
        width = max(len(b) for _, _, b in reqs)
        bt = np.zeros((n, width), dtype=np.int32)
        pos = np.array([p for _, p, _ in reqs], dtype=np.int64)
        for i, (_, _, b) in enumerate(reqs):
            bt[i, :len(b)] = b
        slots = bt[np.arange(n), pos // self.block].astype(np.int64) * self.block + pos % self.block
        dev = self.device
        seq_cpu = torch.from_numpy(pos + 1).to(torch.int32)
        qsl_cpu = torch.arange(n + 1, dtype=torch.int32)
        slots_d = torch.from_numpy(slots).to(dev)
        md, sm = self.metadata(qsl_cpu.to(dev), qsl_cpu, seq_cpu.to(dev), seq_cpu,
                               torch.from_numpy(bt).to(dev), slots_d, n, 1, int(pos.max()) + 1)
        ids = torch.tensor([t for t, _, _ in reqs], dtype=torch.int64, device=dev)
        with self.forward_context(md, sm, n):
            hidden = self.model(input_ids=ids, positions=torch.from_numpy(pos).to(dev))
            return self.model.compute_logits(hidden)

    # ------------------------------------------------------------------
    # prefill, split by layers
    # ------------------------------------------------------------------
    def prefill_begin(self, reqs, temps):
        """reqs: [(request id, token ids, block ids)], on the current stream."""
        lens = [len(t) for _, t, _ in reqs]
        n = sum(lens)
        dev = self.device
        batch = PrefillBatch([r for r, _, _ in reqs], temps)
        tokens = np.concatenate([np.asarray(t, dtype=np.int64) for _, t, _ in reqs])
        pos = np.concatenate([np.arange(k, dtype=np.int64) for k in lens])
        row = np.repeat(np.arange(len(reqs)), lens)
        width = max(len(b) for _, _, b in reqs)
        bt = np.zeros((len(reqs), width), dtype=np.int32)
        for i, (_, _, b) in enumerate(reqs):
            bt[i, :len(b)] = b
        slots = bt[row, pos // self.block].astype(np.int64) * self.block + pos % self.block
        qsl_cpu = torch.tensor(np.concatenate([[0], np.cumsum(lens)]), dtype=torch.int32)
        seq_cpu = torch.tensor(lens, dtype=torch.int32)
        qsl = qsl_cpu.to(dev, non_blocking=True)
        batch.md, batch.slots = self.metadata(
            qsl, qsl_cpu, seq_cpu.to(dev, non_blocking=True), seq_cpu,
            torch.from_numpy(bt).to(dev, non_blocking=True),
            torch.from_numpy(slots).to(dev, non_blocking=True), n, max(lens), max(lens))
        batch.num_tokens = n
        batch.positions = torch.from_numpy(pos).to(dev, non_blocking=True)
        batch.last_rows = (qsl[1:] - 1).long()
        batch.hidden = self.inner.embed_tokens(torch.from_numpy(tokens).to(dev, non_blocking=True))
        return batch

    def prefill_layers(self, batch, start, end):
        """Layers [start, end) of the batch; after the last layer, its first
        tokens are sampled and copied to the host behind an event."""
        with self.forward_context(batch.md, batch.slots, batch.num_tokens):
            for i in range(start, end):
                batch.hidden, batch.residual = self.layers[i](batch.positions, batch.hidden,
                                                              batch.residual)
            if end == self.num_layers:
                hidden, _ = self.inner.norm(batch.hidden, batch.residual)
                logits = self.model.compute_logits(hidden[batch.last_rows])
                batch.hidden = batch.residual = None
                toks = sample(logits, batch.temps)
                batch.tokens = torch.empty(toks.shape, dtype=toks.dtype, pin_memory=True)
                batch.tokens.copy_(toks, non_blocking=True)
                batch.event = torch.cuda.Event()
                batch.event.record()

    # ------------------------------------------------------------------
    # one engine step
    # ------------------------------------------------------------------
    def switch(self, group):
        """Move to another group once the work on the current one is done
        (the streams of different groups share no order)."""
        if group != self.cur:
            for s in self.streams.groups[self.cur]:
                s.synchronize()
            self.cur = group

    def step(self, s):
        """s: the scheduler's MuxStep. Returns ([request id], [[token]])
        for the decode batch and for a prefill batch whose first tokens
        are ready."""
        self.switch(s.group)
        ps, ds = self.streams.groups[self.cur]
        if s.prefill_new is not None:
            with torch.cuda.stream(ps):
                self.prefill = self.prefill_begin(s.prefill_new, s.prefill_temps)
        if s.prefill_layers is not None:
            with torch.cuda.stream(ps):
                self.prefill_layers(self.prefill, *s.prefill_layers)
        ids, toks = [], []
        if s.decode:
            with torch.cuda.stream(ds):
                out = self.decode(s.decode, s.decode_temps)
                host = out.to("cpu", non_blocking=True)
            ds.synchronize()
            ids = list(s.decode_ids)
            toks = [[int(t)] for t in host.tolist()]
        p = self.prefill
        if p is not None and p.event is not None and (p.done() or not s.decode):
            p.event.synchronize()
            ids += p.req_ids
            toks += [[int(t)] for t in p.tokens.tolist()]
            self.prefill = None
        return ids, toks


def sample(logits, temps):
    """Temperature sampling, greedy at temperature 0 (the only sampling
    setting the evaluation uses)."""
    logits = logits.float()
    greedy = logits.argmax(-1)
    if all(t <= 0 for t in temps):
        return greedy
    t = torch.tensor([max(x, 1e-5) for x in temps], device=logits.device)[:, None]
    drawn = torch.multinomial(torch.softmax(logits / t, -1), 1).squeeze(-1)
    zero = torch.tensor([x <= 0 for x in temps], device=logits.device)
    return torch.where(zero, greedy, drawn)
