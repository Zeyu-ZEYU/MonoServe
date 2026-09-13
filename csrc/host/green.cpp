#include "host/green.h"

#include <stdexcept>
#include <string>

namespace monofab {

namespace {
void drv(CUresult r, const char* what) {
  if (r != CUDA_SUCCESS) {
    const char* msg = nullptr;
    cuGetErrorString(r, &msg);
    throw std::runtime_error(std::string("green partitions: ") + what + ": " + (msg ? msg : "?"));
  }
}
}  // namespace

GreenPartitions::~GreenPartitions() {
  try {
    destroy();
  } catch (...) {
  }
}

std::vector<int> GreenPartitions::create(const std::vector<int>& counts) {
  if (!parts_.empty()) throw std::runtime_error("green partitions: already created");
  CUdevice dev;
  drv(cuCtxGetDevice(&dev), "cuCtxGetDevice");
  CUdevResource input{};
  drv(cuDeviceGetDevResource(dev, &input, CU_DEV_RESOURCE_TYPE_SM), "cuDeviceGetDevResource");
  std::vector<int> granted;
  for (size_t k = 0; k < counts.size(); ++k) {
    CUdevResource group{}, remaining{};
    unsigned int n = 1;
    drv(cuDevSmResourceSplitByCount(&group, &n, &input, &remaining, 0,
                                    static_cast<unsigned int>(counts[k])),
        "cuDevSmResourceSplitByCount");
    CUdevResourceDesc desc;
    drv(cuDevResourceGenerateDesc(&desc, &group, 1), "cuDevResourceGenerateDesc");
    Part p{};
    drv(cuGreenCtxCreate(&p.gctx, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM), "cuGreenCtxCreate");
    drv(cuCtxFromGreenCtx(&p.ctx, p.gctx), "cuCtxFromGreenCtx");
    drv(cuGreenCtxStreamCreate(&p.stream, p.gctx, CU_STREAM_NON_BLOCKING, 0),
        "cuGreenCtxStreamCreate");
    p.sms = static_cast<int>(group.sm.smCount);
    parts_.push_back(p);
    granted.push_back(p.sms);
    if (k + 1 < counts.size()) {
      // a split takes only resources queried from a device or a context:
      // materialize the remainder as a context and query it
      CUdevResourceDesc rest;
      drv(cuDevResourceGenerateDesc(&rest, &remaining, 1), "cuDevResourceGenerateDesc");
      CUgreenCtx scaffold;
      drv(cuGreenCtxCreate(&scaffold, rest, dev, CU_GREEN_CTX_DEFAULT_STREAM), "cuGreenCtxCreate");
      scaffolds_.push_back(scaffold);
      drv(cuGreenCtxGetDevResource(scaffold, &input, CU_DEV_RESOURCE_TYPE_SM),
          "cuGreenCtxGetDevResource");
    }
  }
  return granted;
}

void GreenPartitions::destroy() {
  for (auto& p : parts_) {
    cuStreamDestroy(p.stream);
    cuGreenCtxDestroy(p.gctx);
  }
  parts_.clear();
  for (auto& s : scaffolds_) cuGreenCtxDestroy(s);
  scaffolds_.clear();
}

uint64_t GreenPartitions::stream(int i) const {
  if (i < 0 || i >= size()) throw std::runtime_error("green partitions: bad index");
  return reinterpret_cast<uint64_t>(parts_[i].stream);
}

uint64_t GreenPartitions::context(int i) const {
  if (i < 0 || i >= size()) throw std::runtime_error("green partitions: bad index");
  return reinterpret_cast<uint64_t>(parts_[i].ctx);
}

}  // namespace monofab
