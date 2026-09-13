# SPDX-License-Identifier: Apache-2.0
# Adapted from ykcombat/sglang@slo_config (eeac514):
# python/sglang/srt/multiplex/pdmux_context.py (PDMuxConfig,
# load_pdmux_config, get_arch_constraints, divide_sm) and
# python/sglang/srt/multiplex/multiplexing.py (adjust_stream_groups).
# Copyright the SGLang project contributors.
"""MuxWise's SM divisions and how it picks one.

A configuration has sm_group_num stream groups. Group 0 runs prefill alone
on every SM, the last group runs decode alone on every SM, and each group
in between is a division of the SMs into a prefill and a decode partition,
listed in manual_divisions as rows [prefill_sm, decode_sm,
decode_bs_threshold] in order of growing decode share. Without rows, the
divisions are spread over the range the architecture allows.
"""
import json
from dataclasses import dataclass, field


@dataclass
class MuxConfig:
    sm_group_num: int = 8
    manual_divisions: list = field(default_factory=list)   # [prefill_sm, decode_sm, decode_bs_threshold]
    split_forward_token_budget: int = 65536   # prefill tokens x layers per step
    decode_bs_divisor: int = 36               # group choice without manual divisions
    max_prefill_tokens: int = 16384           # prompt tokens per prefill batch


def load(path):
    """A configuration from a YAML or JSON file (MuxWise's format); the
    defaults without one."""
    if not path:
        return MuxConfig()
    with open(path) as f:
        text = f.read()
    if path.endswith(".json"):
        raw = json.loads(text)
    else:
        import yaml
        raw = yaml.safe_load(text)
    if "sm_group_num" not in raw:
        raise ValueError("missing required field: sm_group_num")
    if raw["sm_group_num"] < 3:
        raise ValueError("sm_group_num must be at least 3")
    rows = [list(r) for r in raw.get("manual_divisions", [])]
    if rows and len(rows) != raw["sm_group_num"] - 2:
        raise ValueError(f"manual_divisions must have {raw['sm_group_num'] - 2} entries, "
                         f"got {len(rows)}")
    return MuxConfig(sm_group_num=raw["sm_group_num"], manual_divisions=rows,
                     split_forward_token_budget=raw.get("split_forward_token_budget", 65536),
                     decode_bs_divisor=raw.get("decode_bs_divisor", 36),
                     max_prefill_tokens=raw.get("max_prefill_tokens", 16384))


def dump(cfg, path):
    import yaml
    with open(path, "w") as f:
        yaml.safe_dump({"sm_group_num": cfg.sm_group_num,
                        "manual_divisions": [list(map(int, r)) for r in cfg.manual_divisions],
                        "split_forward_token_budget": cfg.split_forward_token_budget,
                        "decode_bs_divisor": cfg.decode_bs_divisor,
                        "max_prefill_tokens": cfg.max_prefill_tokens}, f, sort_keys=False)


def arch_constraints(capability):
    """(smallest partition, granularity) of green contexts."""
    major, _ = capability
    table = {6: (1, 1), 7: (2, 2), 8: (4, 2), 9: (8, 8)}
    if major not in table:
        raise ValueError(f"unsupported compute capability {capability}")
    return table[major]


def divide_sm(total_sms, capability, groups):
    """(prefill_sm, decode_sm) divisions spread over the allowed range, the
    largest prefill share first."""
    least, step = arch_constraints(capability)
    values = [x for x in range(least, total_sms - least + 1, step)
              if x >= total_sms - x and total_sms - x >= 16]
    if not values:
        raise ValueError(f"no division of {total_sms} SMs fits the constraints")
    if len(values) >= groups:
        values = values[::max(1, len(values) // groups)][:groups]
    return [(p, total_sms - p) for p in reversed(values)]


def divisions(cfg, total_sms, capability):
    if cfg.manual_divisions:
        return [(int(p), int(d)) for p, d, _ in cfg.manual_divisions]
    return divide_sm(total_sms, capability, cfg.sm_group_num - 2)


def select_group(cfg, decode_bs, prefill, decode):
    """The stream group to run: a division while both a prefill batch and a
    decode batch run, else the group of the one that runs."""
    last = cfg.sm_group_num - 1
    if prefill and decode:
        if cfg.manual_divisions:
            group = 1   # below every threshold, the smallest decode share (unassigned in the source)
            for i, (_, _, threshold) in enumerate(cfg.manual_divisions):
                if decode_bs >= threshold:
                    group = i + 1
            return group
        return max(1, min(last - 1, decode_bs * (last - 1) // cfg.decode_bs_divisor))
    if decode:
        return last
    return 0
