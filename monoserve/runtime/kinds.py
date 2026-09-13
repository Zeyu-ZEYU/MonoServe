"""Tile kinds and constants shared with csrc/fabric/model_types.h."""
GEMM, ATTN_PREFILL, ATTN_DECODE, ATTN_COMBINE = 32, 33, 34, 35
RMSNORM, QK_ROPE, ROUTER, EXPAND, PERMUTE, ADD_NORM, EMBED, SAMPLE, ATTN_PLAN = (
    36, 37, 38, 39, 40, 41, 42, 43, 44)
GEMM_ASYM, ASYM_REDUCE = 45, 46

EPI_STORE, EPI_SILU_MUL, EPI_WADD, EPI_F32, EPI_ATOMIC_F32 = 0, 1, 2, 3, 4

TILE_HEIGHTS = (16, 32, 64, 128)   # symmetric tile heights (kernel forms)
FORM_ASYM = 0                      # kernel form of the asymmetric kernel
ASYM_MAX_KT = 8                    # K tiles one asymmetric tile keeps in shared memory
PAGE = 64                          # tokens per KV-cache page
HEAD_DIM = 128

# Expert weight regions of the indirection table.
REGION_HOST = 0    # CPU DRAM, read over the link
REGION_HOT = 1     # hot tier in HBM
REGION_STAGE0 = 2  # staging buffers of the prefill lanes follow


def tile_height(rows):
    """Smallest tile height covering `rows` (at most 128)."""
    for h in TILE_HEIGHTS:
        if rows <= h:
            return h
    return TILE_HEIGHTS[-1]


def height_class(h):
    return TILE_HEIGHTS.index(h)
