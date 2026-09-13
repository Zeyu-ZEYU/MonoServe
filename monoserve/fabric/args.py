"""Operand blocks of model tiles, packed with the field offsets the native
library reports, so the Python side never hard-codes a C++ layout."""
import struct

import torch

_LAYOUTS = None
_FMT = {"u64": "<Q", "u32": "<I", "f32": "<f"}


def layouts():
    global _LAYOUTS
    if _LAYOUTS is None:
        from monoserve import _C
        _LAYOUTS = {k: (int(v[0]), {f: (int(o), t) for f, (o, t) in v[1].items()})
                    for k, v in _C.struct_layouts().items()}
    return _LAYOUTS


def pack(name, **fields):
    """Bytes of struct `name` with the given fields (others zero)."""
    size, spec = layouts()[name]
    buf = bytearray(size)
    for f, v in fields.items():
        if f not in spec:
            raise KeyError(f"{name} has no field {f}")
        off, t = spec[f]
        if t == "u64" and isinstance(v, torch.Tensor):
            v = v.data_ptr()
        struct.pack_into(_FMT[t], buf, off, v)
    return bytes(buf)


def device_block(name, **fields):
    """A device tensor holding the packed struct; keep it alive while any
    uploaded program refers to it."""
    return torch.frombuffer(bytearray(pack(name, **fields)), dtype=torch.uint8).cuda()


def device_u64(values):
    """Device array of 64-bit values (tensor maps, operand block addresses)."""
    return torch.tensor([int(v) for v in values], dtype=torch.int64).cuda()
