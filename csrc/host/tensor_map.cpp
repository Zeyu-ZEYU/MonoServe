#include "host/tensor_map.h"

#include <stdexcept>
#include <string>

namespace monofab {

CUtensorMap make_tensor_map(uint64_t base, const std::vector<uint64_t>& dims,
                            const std::vector<uint64_t>& strides_bytes,
                            const std::vector<uint32_t>& box, int dtype, int swizzle_bytes) {
  const size_t rank = dims.size();
  if (rank < 1 || rank > 5 || box.size() != rank || strides_bytes.size() + 1 != rank)
    throw std::runtime_error("tensor map: inconsistent rank");
  CUtensorMapDataType dt;
  switch (dtype) {
    case 0: dt = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16; break;
    case 1: dt = CU_TENSOR_MAP_DATA_TYPE_FLOAT32; break;
    case 2: dt = CU_TENSOR_MAP_DATA_TYPE_UINT8; break;
    default: throw std::runtime_error("tensor map: unknown dtype");
  }
  CUtensorMapSwizzle sw;
  switch (swizzle_bytes) {
    case 0: sw = CU_TENSOR_MAP_SWIZZLE_NONE; break;
    case 32: sw = CU_TENSOR_MAP_SWIZZLE_32B; break;
    case 64: sw = CU_TENSOR_MAP_SWIZZLE_64B; break;
    case 128: sw = CU_TENSOR_MAP_SWIZZLE_128B; break;
    default: throw std::runtime_error("tensor map: bad swizzle");
  }
  cuuint64_t d[5];
  cuuint64_t s[4];
  cuuint32_t b[5];
  cuuint32_t e[5];
  for (size_t i = 0; i < rank; ++i) {
    d[i] = dims[i];
    b[i] = box[i];
    e[i] = 1;
  }
  for (size_t i = 0; i + 1 < rank; ++i) s[i] = strides_bytes[i];
  CUtensorMap map;
  const CUresult r = cuTensorMapEncodeTiled(
      &map, dt, static_cast<cuuint32_t>(rank), reinterpret_cast<void*>(base), d, s, b, e,
      CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    const char* msg = nullptr;
    cuGetErrorString(r, &msg);
    throw std::runtime_error(std::string("cuTensorMapEncodeTiled: ") + (msg ? msg : "?"));
  }
  return map;
}

}  // namespace monofab
