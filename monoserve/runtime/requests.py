"""Per-request state shared by all lanes, indexed by request slot.

The device keeps each request's last token, length, and produced-token
count; sampling tiles also write the new token and the count to pinned
host memory, which the host polls without any copy.
"""
import numpy as np
import torch

from monoserve import _C


class RequestTable:
    def __init__(self, fab, max_slots, bt_stride, ring=4096):
        self.fab = fab
        self.max_slots = max_slots
        self.bt_stride = bt_stride
        z = lambda dt: torch.zeros(max_slots, dtype=dt, device="cuda")  # noqa: E731
        self.last_token = z(torch.int32)
        self.seq_len = z(torch.int32)
        self.steps = z(torch.int32)
        self.temperature = z(torch.float32)
        self.block_table = torch.zeros(max_slots, bt_stride, dtype=torch.int32, device="cuda")
        self.ring = ring
        self.host_tokens = torch.zeros(max_slots, ring, dtype=torch.int32).pin_memory()
        self.host_steps = torch.zeros(max_slots, dtype=torch.int64).pin_memory()
        self.host_tokens_dev = _C.host_device_pointer(self.host_tokens.data_ptr())
        self.host_steps_dev = _C.host_device_pointer(self.host_steps.data_ptr())

    def _put(self, tensor, slot, value, dtype):
        arr = np.asarray(value, dtype=dtype).reshape(-1)
        addr = tensor.data_ptr() + slot * tensor.element_size() * (tensor.shape[1] if tensor.dim() == 2 else 1)
        self.fab.write(addr, arr.tobytes())

    def set_pages(self, slot, pages):
        row = np.zeros(self.bt_stride, dtype=np.int32)
        row[:len(pages)] = pages
        self._put(self.block_table, slot, row, np.int32)

    def set_length(self, slot, n):
        self._put(self.seq_len, slot, n, np.int32)

    def set_temperature(self, slot, t):
        self._put(self.temperature, slot, t, np.float32)

    def reset_steps(self, slot):
        self._put(self.steps, slot, 0, np.int32)
        self.host_steps[slot] = 0

    def produced(self, slot):
        """Number of tokens produced so far for the request in `slot`."""
        return int(self.host_steps[slot])

    def tokens(self, slot, start, end):
        """Tokens start..end-1 of the request (end - start <= ring)."""
        return [int(self.host_tokens[slot, s % self.ring]) for s in range(start, end)]
