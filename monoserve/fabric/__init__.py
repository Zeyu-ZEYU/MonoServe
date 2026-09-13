"""Python interface to MonoFab, the multi-batch, contention-gated kernel
fabric."""
from monoserve.fabric.program import (KIND, STAGE_LINK, STAGE_MARK,
                                      STAGE_RESET, Program, Tile)

__all__ = ["KIND", "STAGE_LINK", "STAGE_MARK", "STAGE_RESET", "Program",
           "Tile", "Fabric"]


def Fabric(num_workers=0, smem_bytes=200 * 1024, device=0, pool_bytes=1 << 30):
    """The persistent fabric on one GPU (see csrc/host/fabric.h)."""
    from monoserve import _C
    return _C.Fabric(num_workers, smem_bytes, device, pool_bytes)
