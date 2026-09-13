// TMA tensor maps (CUtensorMap) for weights and activations.
#pragma once
#include <cstdint>
#include <vector>

#include <cuda.h>

namespace monofab {

// dims and box are innermost first; strides_bytes has rank - 1 entries, the
// byte strides of dimensions 1..rank-1. dtype: 0 bf16, 1 fp32, 2 fp8 e4m3.
CUtensorMap make_tensor_map(uint64_t base, const std::vector<uint64_t>& dims,
                            const std::vector<uint64_t>& strides_bytes,
                            const std::vector<uint32_t>& box, int dtype, int swizzle_bytes);

}  // namespace monofab
