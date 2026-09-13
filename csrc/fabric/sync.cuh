// Memory-ordering helpers for the fabric's device-side counters.
#pragma once
#include <cstdint>

namespace monofab {

__device__ __forceinline__ unsigned long long globaltimer_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ unsigned int smid() {
  unsigned int s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}

__device__ __forceinline__ unsigned int ld_acquire(const unsigned int* p) {
  unsigned int v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ unsigned long long ld_acquire(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void st_release(unsigned int* p, unsigned int v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void st_release(unsigned long long* p, unsigned long long v) {
  asm volatile("st.release.gpu.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}

// Stores to pinned host memory that the host thread polls.
__device__ __forceinline__ void st_release_sys(unsigned long long* p, unsigned long long v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}

// Loads that bypass L1, for data other agents may have rewritten.
template <class T>
__device__ __forceinline__ T ld_volatile(const T* p) {
  return *reinterpret_cast<const volatile T*>(p);
}

}  // namespace monofab
