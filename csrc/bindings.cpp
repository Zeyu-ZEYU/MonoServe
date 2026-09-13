// Python bindings of the MonoServe native library.
#include <atomic>

#include <torch/extension.h>

#include "fabric/model_types.h"
#include "fabric/launch.h"
#include "host/fabric.h"
#include "host/green.h"
#include "host/host_loop.h"
#include "host/tensor_map.h"

namespace py = pybind11;
using monofab::Fabric;

void bind_control(py::module_& m);   // control/py_control.cpp

namespace {

void require_cpu_contig(const at::Tensor& t, at::ScalarType dt, const char* name) {
  TORCH_CHECK(t.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == dt, name, " has the wrong dtype");
}

py::dict mirror_dict(const Fabric& f) {
  const monofab::HostMirror& m = f.mirror();
  py::list lanes;
  for (int l = 0; l < monofab::kMaxLanes; ++l) {
    const volatile monofab::LaneMirror& lm = m.lane[l];
    // The device writes running before gen when a program starts, and the
    // count before running when it ends: read in the opposite order, with
    // acquire ordering (a host such as Grace may reorder plain loads).
    const unsigned long long mark = lm.mark;
    std::atomic_thread_fence(std::memory_order_acquire);
    const unsigned long long mark_ns = lm.mark_ns;
    const unsigned long long gen = lm.gen;
    std::atomic_thread_fence(std::memory_order_acquire);
    const unsigned long long running = lm.running;
    std::atomic_thread_fence(std::memory_order_acquire);
    const unsigned long long iterations = lm.iterations;
    py::dict d;
    d["iterations"] = iterations;
    d["gen"] = gen;
    d["running"] = running;
    d["mark"] = mark;
    d["mark_ns"] = mark_ns;
    lanes.append(d);
  }
  py::list epochs, epoch_ns;
  for (int w = 0; w < f.num_workers(); ++w) {
    epochs.append(static_cast<unsigned long long>(
        reinterpret_cast<const volatile unsigned long long*>(m.worker_epoch)[w]));
    epoch_ns.append(static_cast<unsigned long long>(
        reinterpret_cast<const volatile unsigned long long*>(m.worker_epoch_ns)[w]));
  }
  py::dict out;
  out["lanes"] = lanes;
  out["worker_epoch"] = epochs;
  out["worker_epoch_ns"] = epoch_ns;
  return out;
}

}  // namespace

#define MF_FIELD(S, F, T) f[#F] = py::make_tuple(offsetof(S, F), T)
py::dict struct_layouts() {
  using namespace monofab;
  py::dict out;
  {
    py::dict f;
    MF_FIELD(GemmArgs, out, "u64"); MF_FIELD(GemmArgs, pair_token, "u64");
    MF_FIELD(GemmArgs, pair_weight, "u64"); MF_FIELD(GemmArgs, residual, "u64");
    MF_FIELD(GemmArgs, bias, "u64"); MF_FIELD(GemmArgs, ld_out, "u32"); MF_FIELD(GemmArgs, ld_res, "u32");
    MF_FIELD(GemmArgs, k_tiles, "u32"); MF_FIELD(GemmArgs, epi, "u32"); MF_FIELD(GemmArgs, bm, "u32");
    MF_FIELD(GemmArgs, n_limit, "u32"); MF_FIELD(GemmArgs, up_offset, "u32");
    MF_FIELD(GemmArgs, w_fp8, "u32");
    out["GemmArgs"] = py::make_tuple(sizeof(GemmArgs), f);
  }
  {
    py::dict f;
    MF_FIELD(AttnArgs, q_map, "u64"); MF_FIELD(AttnArgs, k_map, "u64"); MF_FIELD(AttnArgs, v_map, "u64");
    MF_FIELD(AttnArgs, out, "u64"); MF_FIELD(AttnArgs, ws_o, "u64"); MF_FIELD(AttnArgs, ws_m, "u64");
    MF_FIELD(AttnArgs, ws_l, "u64"); MF_FIELD(AttnArgs, block_table, "u64"); MF_FIELD(AttnArgs, req_info, "u64");
    MF_FIELD(AttnArgs, scale_log2, "f32"); MF_FIELD(AttnArgs, bt_stride, "u32"); MF_FIELD(AttnArgs, hq, "u32");
    MF_FIELD(AttnArgs, hkv, "u32"); MF_FIELD(AttnArgs, group, "u32"); MF_FIELD(AttnArgs, chunk_blocks, "u32");
    MF_FIELD(AttnArgs, max_chunks, "u32");
    out["AttnArgs"] = py::make_tuple(sizeof(AttnArgs), f);
  }
  {
    py::dict f;
    MF_FIELD(NormArgs, x, "u64"); MF_FIELD(NormArgs, x2, "u64"); MF_FIELD(NormArgs, acc, "u64");
    MF_FIELD(NormArgs, w, "u64"); MF_FIELD(NormArgs, out, "u64"); MF_FIELD(NormArgs, embed, "u64");
    MF_FIELD(NormArgs, tokens, "u64"); MF_FIELD(NormArgs, gather, "u64"); MF_FIELD(NormArgs, out2, "u64");
    MF_FIELD(NormArgs, step, "u64"); MF_FIELD(NormArgs, H, "u32"); MF_FIELD(NormArgs, ld, "u32");
    MF_FIELD(NormArgs, eps, "f32");
    out["NormArgs"] = py::make_tuple(sizeof(NormArgs), f);
  }
  {
    py::dict f;
    MF_FIELD(StepArgs, row_req, "u64"); MF_FIELD(StepArgs, last_token, "u64"); MF_FIELD(StepArgs, seq_len, "u64");
    MF_FIELD(StepArgs, block_table, "u64"); MF_FIELD(StepArgs, tok_pos, "u64"); MF_FIELD(StepArgs, tok_slot, "u64");
    MF_FIELD(StepArgs, req_info, "u64"); MF_FIELD(StepArgs, bt_stride, "u32");
    out["StepArgs"] = py::make_tuple(sizeof(StepArgs), f);
  }
  {
    py::dict f;
    MF_FIELD(QkArgs, qkv, "u64"); MF_FIELD(QkArgs, q_out, "u64"); MF_FIELD(QkArgs, k_cache, "u64");
    MF_FIELD(QkArgs, v_cache, "u64"); MF_FIELD(QkArgs, q_norm, "u64"); MF_FIELD(QkArgs, k_norm, "u64");
    MF_FIELD(QkArgs, tok_pos, "u64"); MF_FIELD(QkArgs, tok_slot, "u64"); MF_FIELD(QkArgs, ld_qkv, "u32");
    MF_FIELD(QkArgs, hq, "u32"); MF_FIELD(QkArgs, hkv, "u32"); MF_FIELD(QkArgs, rot_dim, "u32");
    MF_FIELD(QkArgs, theta, "f32"); MF_FIELD(QkArgs, eps, "f32");
    MF_FIELD(QkArgs, cos_sin, "u64"); MF_FIELD(QkArgs, max_pos, "u32");
    out["QkArgs"] = py::make_tuple(sizeof(QkArgs), f);
  }
  {
    py::dict f;
    MF_FIELD(RouterArgs, logits, "u64"); MF_FIELD(RouterArgs, topk_ids, "u64"); MF_FIELD(RouterArgs, topk_w, "u64");
    MF_FIELD(RouterArgs, counts, "u64"); MF_FIELD(RouterArgs, bias, "u64"); MF_FIELD(RouterArgs, E, "u32");
    MF_FIELD(RouterArgs, K, "u32"); MF_FIELD(RouterArgs, scoring, "u32"); MF_FIELD(RouterArgs, renorm, "u32");
    MF_FIELD(RouterArgs, scale, "f32");
    out["RouterArgs"] = py::make_tuple(sizeof(RouterArgs), f);
  }
  {
    py::dict f;
    MF_FIELD(ExpandArgs, counts, "u64"); MF_FIELD(ExpandArgs, starts, "u64"); MF_FIELD(ExpandArgs, fill, "u64");
    MF_FIELD(ExpandArgs, table, "u64"); MF_FIELD(ExpandArgs, w13_maps, "u64"); MF_FIELD(ExpandArgs, w2_maps, "u64");
    MF_FIELD(ExpandArgs, xp_maps, "u64"); MF_FIELD(ExpandArgs, h_maps, "u64"); MF_FIELD(ExpandArgs, w13_args, "u64");
    MF_FIELD(ExpandArgs, w2_args, "u64"); MF_FIELD(ExpandArgs, hist, "u64"); MF_FIELD(ExpandArgs, E, "u32");
    MF_FIELD(ExpandArgs, H, "u32");
    MF_FIELD(ExpandArgs, I, "u32"); MF_FIELD(ExpandArgs, host_region, "u32"); MF_FIELD(ExpandArgs, w13_stage0, "u32");
    MF_FIELD(ExpandArgs, w2_stage0, "u32"); MF_FIELD(ExpandArgs, dyn_first, "u32"); MF_FIELD(ExpandArgs, dyn_cap, "u32");
    MF_FIELD(ExpandArgs, lane, "u32"); MF_FIELD(ExpandArgs, w13_asym_args, "u64");
    MF_FIELD(ExpandArgs, w2_asym_args, "u64"); MF_FIELD(ExpandArgs, red_args, "u64");
    MF_FIELD(ExpandArgs, red_stage0, "u32"); MF_FIELD(ExpandArgs, w13_scale, "u64");
    MF_FIELD(ExpandArgs, w2_scale, "u64"); MF_FIELD(ExpandArgs, join_stage, "u32");
    out["ExpandArgs"] = py::make_tuple(sizeof(ExpandArgs), f);
  }
  {
    py::dict f;
    MF_FIELD(ReduceArgs, ws, "u64"); MF_FIELD(ReduceArgs, out, "u64"); MF_FIELD(ReduceArgs, I, "u32");
    MF_FIELD(ReduceArgs, ld_ws, "u32"); MF_FIELD(ReduceArgs, ld_out, "u32");
    out["ReduceArgs"] = py::make_tuple(sizeof(ReduceArgs), f);
  }
  {
    py::dict f;
    MF_FIELD(PermuteArgs, topk_ids, "u64"); MF_FIELD(PermuteArgs, topk_w, "u64"); MF_FIELD(PermuteArgs, starts, "u64");
    MF_FIELD(PermuteArgs, fill, "u64"); MF_FIELD(PermuteArgs, pair_token, "u64"); MF_FIELD(PermuteArgs, pair_weight, "u64");
    MF_FIELD(PermuteArgs, src, "u64"); MF_FIELD(PermuteArgs, dst, "u64"); MF_FIELD(PermuteArgs, K, "u32");
    MF_FIELD(PermuteArgs, H, "u32");
    out["PermuteArgs"] = py::make_tuple(sizeof(PermuteArgs), f);
  }
  {
    py::dict f;
    MF_FIELD(SampleArgs, logits, "u64"); MF_FIELD(SampleArgs, temperature, "u64"); MF_FIELD(SampleArgs, last_token, "u64");
    MF_FIELD(SampleArgs, steps, "u64"); MF_FIELD(SampleArgs, host_tokens, "u64"); MF_FIELD(SampleArgs, host_steps, "u64");
    MF_FIELD(SampleArgs, seed, "u64"); MF_FIELD(SampleArgs, V, "u32"); MF_FIELD(SampleArgs, ring, "u32");
    out["SampleArgs"] = py::make_tuple(sizeof(SampleArgs), f);
  }
  {
    py::dict f;
    MF_FIELD(AttnPlanArgs, attn, "u64"); MF_FIELD(AttnPlanArgs, req_info, "u64"); MF_FIELD(AttnPlanArgs, rows, "u32");
    MF_FIELD(AttnPlanArgs, hkv, "u32"); MF_FIELD(AttnPlanArgs, chunk_keys, "u32"); MF_FIELD(AttnPlanArgs, stage_attn, "u32");
    MF_FIELD(AttnPlanArgs, stage_comb, "u32"); MF_FIELD(AttnPlanArgs, dyn_first, "u32"); MF_FIELD(AttnPlanArgs, dyn_cap, "u32");
    out["AttnPlanArgs"] = py::make_tuple(sizeof(AttnPlanArgs), f);
  }
  return out;
}
#undef MF_FIELD

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("struct_layouts", &struct_layouts);
  m.def("host_device_pointer", [](uint64_t host) {
    void* d = nullptr;
    if (cudaHostGetDevicePointer(&d, reinterpret_cast<void*>(host), 0) != cudaSuccess)
      throw std::runtime_error("cudaHostGetDevicePointer failed");
    return reinterpret_cast<uint64_t>(d);
  });
  m.doc() = "MonoServe native library: the MonoFab kernel fabric and the control plane";
  py::module_ control = m.def_submodule("control", "Time estimator, plan search, and admission");
  bind_control(control);
  m.def("wait_note_address", &monofab::wait_note_address,
        "Device address of the fabric's long-wait records (hang diagnosis)");
  py::class_<Fabric>(m, "Fabric")
      .def(py::init<int, size_t, int, size_t>(), py::arg("num_workers") = 0,
           py::arg("smem_bytes") = 200 * 1024, py::arg("device") = 0,
           py::arg("pool_bytes") = size_t(1) << 30)
      .def("start", &Fabric::start, py::call_guard<py::gil_scoped_release>())
      .def("start_on", &Fabric::start_on, py::arg("stream"), py::arg("context"), py::arg("workers"),
           py::arg("first_worker") = 0, py::call_guard<py::gil_scoped_release>())
      .def("stop", &Fabric::stop, py::call_guard<py::gil_scoped_release>())
      .def("set_idle_cap", &Fabric::set_idle_cap, py::arg("ns"))
      .def("set_trace", &Fabric::set_trace, py::arg("ptr"), py::arg("capacity"))
      .def("trace_count", &Fabric::trace_count)
      .def_property_readonly("running", &Fabric::running)
      .def_property_readonly("num_workers", &Fabric::num_workers)
      .def_property_readonly("num_sms", &Fabric::num_sms)
      .def_property_readonly("epoch", &Fabric::epoch)
      .def("upload",
           [](Fabric& f, const at::Tensor& tiles, const at::Tensor& stages, const at::Tensor& succ,
              int first_stage, int reset_stage, int iterations) {
             require_cpu_contig(tiles, at::kLong, "tiles");
             require_cpu_contig(stages, at::kInt, "stages");
             require_cpu_contig(succ, at::kInt, "succ");
             TORCH_CHECK(tiles.dim() == 2 && tiles.size(1) == 8, "tiles must be n x 8 int64");
             TORCH_CHECK(stages.dim() == 2 && stages.size(1) == 8, "stages must be m x 8 int32");
             return f.upload(tiles.data_ptr<int64_t>(), tiles.size(0), stages.data_ptr<int32_t>(),
                             stages.size(0), succ.data_ptr<int32_t>(), succ.numel(), first_stage,
                             reset_stage, iterations);
           })
      .def("release", &Fabric::release)
      .def("generation", &Fabric::generation)
      .def("publish", &Fabric::publish, py::arg("map"), py::arg("caps"), py::arg("order"),
           py::arg("programs"), py::arg("forms") = std::vector<int>{},
           py::call_guard<py::gil_scoped_release>())
      .def("progress", &mirror_dict)
      .def("lane_stats",
           [](Fabric& f, int lane) {
             monofab::LaneStats s = f.lane_stats(lane);
             py::dict d;
             d["slots"] = s.slots;
             d["inflight"] = s.inflight;
             d["inflight_max"] = s.inflight_max;
             d["running"] = s.running;
             d["tag"] = s.tag;
             d["iter_in_prog"] = s.iter_in_prog;
             d["gen"] = s.gen;
             return d;
           })
      .def("reset_inflight_max", &Fabric::reset_inflight_max)
      .def("iterations", &Fabric::iterations)
      .def("iteration_times", &Fabric::iteration_times, py::arg("lane"), py::arg("start"),
           py::arg("count"), py::arg("timeout") = 60.0, py::call_guard<py::gil_scoped_release>())
      .def("read",
           [](Fabric& f, uint64_t src, size_t bytes) {
             std::string s(bytes, '\0');
             {
               py::gil_scoped_release r;
               f.read(src, s.data(), bytes);
             }
             return py::bytes(s);
           },
           py::arg("src"), py::arg("bytes"))
      .def("program_info",
           [](const Fabric& f, int64_t handle) {
             const Fabric::ProgramInfo p = f.program_info(handle);
             py::dict d;
             d["tiles"] = p.tiles;
             d["stages"] = p.stages;
             d["runtime"] = p.runtime;
             d["n_tiles"] = p.n_tiles;
             d["n_stages"] = p.n_stages;
             return d;
           })
      .def("lane_ring",
           [](Fabric& f, int lane, bool link, size_t max_entries) {
             Fabric::RingView v = f.lane_ring(lane, link, max_entries);
             py::dict d;
             d["head"] = v.head;
             d["tail"] = v.tail;
             d["reserve"] = v.reserve;
             d["entries"] = v.entries;
             return d;
           },
           py::arg("lane"), py::arg("link") = false, py::arg("max_entries") = 4096)
      .def("worker_tiles",
           [](Fabric& f) {
             py::list out;
             for (const Fabric::WorkerTile& t : f.worker_tiles()) {
               py::dict d;
               d["running"] = static_cast<int>((t.state >> 63) & 1);
               d["completing"] = static_cast<int>((t.state >> 62) & 1);
               d["lane"] = static_cast<int>((t.state >> 56) & 0x3f);
               d["kind"] = static_cast<int>((t.state >> 44) & 0xfff);
               d["stage"] = static_cast<int>((t.state >> 24) & 0xfffff);
               d["index"] = static_cast<int>(t.state & 0xffffff);
               d["ns"] = t.ns;
               out.append(d);
             }
             return out;
           })
      .def("write",
           [](Fabric& f, uint64_t dst, py::bytes data) {
             std::string s = data;
             f.write(dst, s.data(), s.size());
           },
           py::arg("dst"), py::arg("data"))
      .def("blob_alloc", &Fabric::blob_alloc)
      .def("blob_free", &Fabric::blob_free)
      .def("tensor_map",
           [](Fabric& f, uint64_t base, std::vector<uint64_t> dims, std::vector<uint64_t> strides,
              std::vector<uint32_t> box, int dtype, int swizzle) {
             return f.add_tensor_map(monofab::make_tensor_map(base, dims, strides, box, dtype, swizzle));
           },
           py::arg("base"), py::arg("dims"), py::arg("strides_bytes"), py::arg("box"),
           py::arg("dtype") = 0, py::arg("swizzle") = 128);

  py::class_<monofab::GreenPartitions>(m, "GreenPartitions")
      .def(py::init<>())
      .def("create", &monofab::GreenPartitions::create, py::arg("counts"))
      .def("destroy", &monofab::GreenPartitions::destroy)
      .def("stream", &monofab::GreenPartitions::stream)
      .def("context", &monofab::GreenPartitions::context)
      .def_property_readonly("size", &monofab::GreenPartitions::size);

  py::class_<monofab::HostLoop>(m, "HostLoop")
      .def(py::init([](Fabric& f, int layers, int experts, size_t w13_bytes, size_t w2_bytes,
                       std::vector<uint64_t> host_w13, std::vector<uint64_t> host_w2, int host_region) {
             return new monofab::HostLoop(&f, layers, experts, w13_bytes, w2_bytes, host_w13, host_w2,
                                          host_region);
           }),
           py::keep_alive<1, 2>())
      .def("add_lane",
           [](monofab::HostLoop& h, int lane, int region, uint64_t w13_dst, uint64_t w2_dst, int slots,
              uint64_t tables, std::vector<std::vector<int>> order) {
             monofab::StagingLaneConfig c{lane, region, w13_dst, w2_dst, slots, tables, order};
             h.add_lane(c);
           })
      .def("start", &monofab::HostLoop::start, py::arg("poll_us") = 5)
      .def("stop", &monofab::HostLoop::stop, py::call_guard<py::gil_scoped_release>())
      .def("stats", [](const monofab::HostLoop& h) {
        monofab::HostLoopStats s = h.stats();
        py::dict d;
        d["copies"] = s.copies;
        d["bytes"] = s.bytes;
        d["flips"] = s.flips;
        d["restores"] = s.restores;
        d["polls"] = s.polls;
        return d;
      });

  m.def("gemm_args",
        [](uint64_t out, uint32_t ld_out, uint32_t k_tiles, uint32_t epi, uint32_t bm, uint32_t n_limit,
           uint64_t pair_token, uint64_t pair_weight, uint64_t residual, uint32_t ld_res, uint64_t bias,
           uint32_t up_offset, uint32_t w_fp8) {
          monofab::GemmArgs a{};
          a.out = out;
          a.ld_out = ld_out;
          a.k_tiles = k_tiles;
          a.epi = epi;
          a.bm = bm;
          a.n_limit = n_limit;
          a.pair_token = pair_token;
          a.pair_weight = pair_weight;
          a.residual = residual;
          a.ld_res = ld_res;
          a.bias = bias;
          a.up_offset = up_offset;
          a.w_fp8 = w_fp8;
          return py::bytes(reinterpret_cast<const char*>(&a), sizeof(a));
        },
        py::arg("out"), py::arg("ld_out"), py::arg("k_tiles"), py::arg("epi"), py::arg("bm"),
        py::arg("n_limit"), py::arg("pair_token") = 0, py::arg("pair_weight") = 0,
        py::arg("residual") = 0, py::arg("ld_res") = 0, py::arg("bias") = 0, py::arg("up_offset") = 0,
        py::arg("w_fp8") = 0);

  m.def("attn_args",
        [](uint64_t q_map, uint64_t k_map, uint64_t v_map, uint64_t out, uint64_t ws_o, uint64_t ws_m,
           uint64_t ws_l, uint64_t block_table, uint64_t req_info, float scale_log2,
           uint32_t bt_stride, uint32_t hq, uint32_t hkv, uint32_t chunk_blocks, uint32_t max_chunks) {
          monofab::AttnArgs a{};
          a.q_map = q_map;
          a.k_map = k_map;
          a.v_map = v_map;
          a.out = out;
          a.ws_o = ws_o;
          a.ws_m = ws_m;
          a.ws_l = ws_l;
          a.block_table = block_table;
          a.req_info = req_info;
          a.scale_log2 = scale_log2;
          a.bt_stride = bt_stride;
          a.hq = hq;
          a.hkv = hkv;
          a.group = hq / hkv;
          a.chunk_blocks = chunk_blocks;
          a.max_chunks = max_chunks;
          return py::bytes(reinterpret_cast<const char*>(&a), sizeof(a));
        },
        py::arg("q_map"), py::arg("k_map"), py::arg("v_map"), py::arg("out"), py::arg("ws_o"),
        py::arg("ws_m"), py::arg("ws_l"), py::arg("block_table"), py::arg("req_info"),
        py::arg("scale_log2"), py::arg("bt_stride"), py::arg("hq"), py::arg("hkv"),
        py::arg("chunk_blocks"), py::arg("max_chunks"));
}
