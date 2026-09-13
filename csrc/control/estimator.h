// The time estimator of the control plane (Eq. 1 and Eq. 2 of the paper).
//
// A layer has two windows, attention then MoE. Within a window computation
// and memory reads overlap, so its estimate is the largest of one term per
// resource it can wait for: SM compute, HBM reads stretched by MTQ
// contention, and, for MoE, the link. Every input is calibrated offline
// (solo sweeps, credit sweeps, the gamma probe) or known at plan time (the
// plan, the activation profile, the router's choices).
#pragma once
#include <cstdint>
#include <map>
#include <vector>

#include "control/curve.h"

namespace monoplan {

// Kernel forms for a lane's DRAM experts: a symmetric template is named by
// its tile height BM (16, 32, 64, 128); the asymmetric kernel by kAsym.
constexpr int kAsym = 0;
// K tiles (of 64) per asymmetric tile, as in csrc/fabric/model_types.h.
constexpr int kAsymMaxKTiles = 8;

struct ModelShape {
  int hidden = 0, intermediate = 0, experts = 0, top_k = 0, layers = 0;
  int heads = 0, kv_heads = 0, head_dim = 128, vocab = 0;
  int dense_layers = 0;            // leading layers with a dense MLP instead of MoE
  int dense_intermediate = 0;      // their MLP width
  int shared_intermediate = 0;     // shared expert of every MoE layer (0: none)
  double expert_weight_bytes = 2;  // per expert weight (bf16: 2, fp8 with scales: ~1)
  double dense_weight_bytes = 2;

  double expert_bytes() const;     // w: w13 and w2 of one expert
  double kv_bytes_per_token() const;
};

// One kernel from the solo sweep: its rates against the SM share.
struct KernelRates {
  Curve flops;          // FLOP/s of a compute-bound shape
  Curve bytes;          // HBM bytes/s of a read-bound shape
  double q_per_sm = 0;  // MTQ entries one SM keeps in flight under this kernel
};

struct Calibration {
  int sms = 0;                          // S
  double R_H = 0, R_C = 0;              // residency of an HBM miss and of a link miss (s)
  Curve gamma;                          // residency stretch against exposure (entry-seconds)
  double q_max = 0;                     // most entries one SM keeps in flight, any HBM reader
  KernelRates dense;                    // projections, router, vocabulary projection
  KernelRates attn_prefill, attn_decode;
  std::map<int, KernelRates> expert;    // by form; symmetric forms double as HBM-expert kernels
  std::map<int, Curve> rho;             // link rate (bytes/s) against credits, by form
  std::map<int, double> q_form;         // q(kappa): entries one DRAM-expert tile keeps in flight
  double beta = 0;                      // the link's saturated rate (bytes/s)
  int reference_form = 64;              // the symmetric template that defines Q^link
  double layer_overhead = 0;            // per layer (s)
  double tail_overhead = 0;             // per pass beyond the vocabulary projection (s)

  // Q^link = min{ n : rho(n) = beta } on the reference form's curve (Eq. 4).
  double link_credits() const;
  // X_max = Q^link R_C + q_max S R_H, the exposure bound used at plan time.
  double exposure_bound() const;
  double rho_at(int form, double credits) const;
  double q_of(int form) const;
};

// A sequence of a lane: `tokens` computed this pass after `context` cached.
struct Seq {
  int tokens = 1;
  int context = 0;
};

// What a lane still has to run, as the estimator sees it at plan time.
struct LaneWork {
  bool decode = false;
  // Passes left: a decode lane has one (the next token of every request);
  // a prefill lane has one per remaining chunk of its sealed batch.
  std::vector<std::vector<Seq>> passes;
  int first_layer = 0;             // layers of the first pass already done
  std::vector<double> active;      // expected activated experts per layer (first pass)
  std::vector<double> miss;        // of those, expected outside the hot tier: E-bar_b
  int staging = 0;                 // V, the staging buffer in experts (0: none)
  double deadline = 0;             // D_b - t (s): TPOT for decode, TTFT left for prefill
  bool samples = true;             // the last pass ends in the vocabulary projection
};

// Per-pass quantities derived once from a LaneWork.
struct PassLoad {
  double tokens = 0;               // T, rows through the dense layers
  double attn_flops = 0;           // attention core
  double kv_bytes = 0;             // KV-cache reads
  double dense_attn_flops = 0;     // QKV and output projections
  double dense_attn_bytes = 0;
  double pairs = 0;                // routed rows, T * top_k
  double moe_act_bytes = 0;        // permuted activations, expert outputs, combine
  double shared_flops = 0, shared_bytes = 0;
  double dense_mlp_flops = 0, dense_mlp_bytes = 0;
  double link_rows_bound = 0;      // M_e bound for the link term (decode: batch size)
  double sampled = 0;              // rows through the vocabulary projection
  std::vector<double> active, miss;
  int first_layer = 0;
};

struct LaneLoad {
  bool decode = false;
  int staging = 0;
  double deadline = 0;
  std::vector<PassLoad> passes;
};

struct Estimate {
  double total = 0;                // T-hat_b: every remaining layer and the tails
  double attn = 0;                 // T-hat^a of one layer of the first pass
  double moe = 0;                  // mean T-hat^m over the first pass's MoE layers
  double staged = 0;               // mean g_b over those layers
  // terms of the first MoE layer of the first pass, for inspection
  double a_cmp = 0, a_rd = 0, m_cmp = 0, m_rd = 0, m_link = 0;
};

struct EstimatorOptions {
  bool contention_aware = true;    // false: no gamma, and no credit gate (MCA ablation)
  double ewma = 0.1;               // weight of one observation in the online correction
};

class Estimator {
 public:
  Estimator(const ModelShape& m, const Calibration& c, EstimatorOptions o = {});

  LaneLoad load(const LaneWork& w) const;
  // T-hat_b at SM share s, link credits Q, and DRAM-expert form kappa.
  Estimate evaluate(const LaneLoad& l, int s, double credits, int form) const;

  // gamma(X_max) / gamma(own): the read stretch of a window whose own
  // kernels load the MTQ to `own` alone.
  double gamma_ratio(double own) const;

  // Online correction: observed over estimated window time.
  void observe(bool decode, bool moe, double ratio);
  double correction(bool decode, bool moe) const { return corr_[decode ? 1 : 0][moe ? 1 : 0]; }

  const ModelShape& model() const { return m_; }
  const Calibration& calibration() const { return c_; }
  const EstimatorOptions& options() const { return o_; }
  double link_credits() const { return q_link_; }
  double exposure_bound() const { return x_max_; }

 private:
  const KernelRates& expert_rates(int form) const;

  ModelShape m_;
  Calibration c_;
  EstimatorOptions o_;
  double q_link_ = 0, x_max_ = 0, gamma_max_ = 1, w_ = 0, asym_row_ = 0;
  double corr_[2][2] = {{1, 1}, {1, 1}};
};

// The workload of one pass of a lane: FLOPs, bytes, and routed rows, as the
// estimator counts them. The calibration counts its runs the same way.
PassLoad pass_load(const ModelShape& m, const LaneWork& w, const std::vector<Seq>& seqs, bool first,
                   bool last);
// HBM bytes per routed row of a DRAM expert under the asymmetric form: the
// w13 partial sums (atomics per K split), the reduce's read and zeroing of
// them, its bf16 output, and the w2 atomics per K split.
double asym_row_bytes(const ModelShape& m);
// Tile height the expander picks for an HBM expert with m rows.
int hbm_tile_height(double m);

}  // namespace monoplan
