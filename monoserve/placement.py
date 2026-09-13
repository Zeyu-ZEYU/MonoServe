"""Expert placement: the hot tier and the activation profiles behind it.

Per layer, the hot tier holds the experts with the highest activation
probability p_e, an exponentially weighted average of the decode lane's
router output, seeded from a profile at load time. Prefill routing is not
counted, since one long prompt activates nearly every expert and would
swamp the ranking. The tier's capacity is the HBM left after the non-expert
weights, the KV-cache reservation, and the staging buffers.
"""
import json

import numpy as np


class ActivationProfile:
    def __init__(self, num_layers, num_experts, alpha=0.05):
        self.p = np.full((num_layers, num_experts), 1.0 / num_experts)
        self.alpha = alpha

    @classmethod
    def load(cls, path):
        data = json.load(open(path))
        prof = cls(len(data["p"]), len(data["p"][0]), data.get("alpha", 0.05))
        prof.p = np.asarray(data["p"], dtype=np.float64)
        return prof

    def save(self, path):
        json.dump({"p": self.p.tolist(), "alpha": self.alpha}, open(path, "w"))

    def update(self, layer, counts):
        """Fold one decode step's per-expert activation counts into p_e."""
        c = np.asarray(counts, dtype=np.float64)
        total = c.sum()
        if total > 0:
            self.p[layer] = (1 - self.alpha) * self.p[layer] + self.alpha * c / total

    def expected_misses(self, layer, hot, rows):
        """Expected number of distinct activated experts outside the hot
        tier for a batch of `rows` routed rows (each picking top-k)."""
        p = self.p[layer]
        miss = np.ones_like(p, dtype=bool)
        miss[list(hot)] = False
        # an expert is activated by a batch unless every row misses it
        return float(np.sum(1.0 - np.power(1.0 - np.clip(p[miss], 0, 1), rows)))


def hot_tier(profile, capacity_experts):
    """Experts to keep in HBM per layer, capacity_experts in total, taken
    in order of activation probability across all layers."""
    L, E = profile.p.shape
    order = np.dstack(np.unravel_index(np.argsort(-profile.p, axis=None), (L, E)))[0]
    hot = [[] for _ in range(L)]
    for layer, e in order[:capacity_experts]:
        hot[int(layer)].append(int(e))
    return [sorted(h) for h in hot]


def hot_capacity(hbm_bytes, reserved_bytes, expert_bytes):
    """How many experts fit in the HBM left after the reservations."""
    return max(0, int((hbm_bytes - reserved_bytes) // expert_bytes))
