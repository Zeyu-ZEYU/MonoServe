"""Python side of the host loop (csrc/host/host_loop.h): staging of the
experts a prefill lane is about to use, by the copy engine, into the lane's
staging buffer."""
import numpy as np

from monoserve import _C
from monoserve.runtime.kinds import REGION_HOST, REGION_STAGE0


def staging_order(profile, weights, layer):
    """Experts to stage for a layer, best first: prefill activation
    frequency, skipping experts the hot tier already holds."""
    p = profile[layer] if profile is not None else np.ones(weights.cfg.num_experts)
    hot = set(weights.hot[layer])
    return [int(e) for e in np.argsort(-np.asarray(p)) if int(e) not in hot]


class HostLoop:
    def __init__(self, fab, weights):
        cfg = weights.cfg
        w13_bytes, w2_bytes = weights.expert_bytes()
        # copies read the pinned host copies directly (dense layers have none)
        host13 = [0 if t is None else t.data_ptr() for t in weights.host_w13]
        host2 = [0 if t is None else t.data_ptr() for t in weights.host_w2]
        self.impl = _C.HostLoop(fab, cfg.num_layers, cfg.num_experts, w13_bytes, w2_bytes,
                                host13, host2, REGION_HOST)
        self.weights = weights

    def add_prefill_lane(self, lane, staging_index, profile=None):
        """Stage for `lane` into staging buffer `staging_index`; each half of
        the buffer holds one layer's worth of experts."""
        w = self.weights
        slots = w.staging_slots // 2
        order = [staging_order(profile, w, l) if w.host_w13[l] is not None else []
                 for l in range(w.cfg.num_layers)]
        self.impl.add_lane(lane.lane_id, REGION_STAGE0 + staging_index,
                           w.stage_w13[staging_index].data_ptr(), w.stage_w2[staging_index].data_ptr(),
                           slots, lane.tables.data_ptr(), order)

    def start(self, poll_us=5):
        self.impl.start(poll_us)

    def stop(self):
        self.impl.stop()

    def stats(self):
        return self.impl.stats()
