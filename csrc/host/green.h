// Disjoint SM partitions through green contexts (CUDA driver API), for the
// ablation that runs every lane's fabric kernel in its own partition.
// Successive cuDevSmResourceSplitByCount calls split the running
// remainder; each group gets its own green context and stream.
#pragma once
#include <cstdint>
#include <vector>

#include <cuda.h>

namespace monofab {

class GreenPartitions {
 public:
  GreenPartitions() = default;
  ~GreenPartitions();
  GreenPartitions(const GreenPartitions&) = delete;
  GreenPartitions& operator=(const GreenPartitions&) = delete;

  // Carve disjoint partitions of the requested SM counts, in order; returns
  // the counts the driver granted (rounded to its granularity).
  std::vector<int> create(const std::vector<int>& counts);
  void destroy();
  int size() const { return static_cast<int>(parts_.size()); }
  uint64_t stream(int i) const;    // CUstream bound to partition i
  uint64_t context(int i) const;   // CUcontext of partition i

 private:
  struct Part {
    CUgreenCtx gctx;
    CUcontext ctx;
    CUstream stream;
    int sms;
  };
  std::vector<Part> parts_;
  std::vector<CUgreenCtx> scaffolds_;   // remainders the next split starts from
};

}  // namespace monofab
