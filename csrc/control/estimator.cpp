#include "control/estimator.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace monoplan {

namespace {
constexpr double kInf = 1e30;

double safe_div(double num, double den) {
  if (num <= 0) return 0.0;
  return den > 0 ? num / den : kInf;
}

double padded(double m, int bm) { return std::ceil(std::max(m, 0.0) / bm) * bm; }
}  // namespace

double ModelShape::expert_bytes() const {
  return 3.0 * hidden * intermediate * expert_weight_bytes;
}

double ModelShape::kv_bytes_per_token() const {
  return 2.0 * layers * kv_heads * head_dim * 2.0;
}

int hbm_tile_height(double m) { return m <= 16 ? 16 : m <= 32 ? 32 : m <= 64 ? 64 : 128; }

double asym_row_bytes(const ModelShape& m) {
  auto splits = [](int k) { return std::max(1, (k / 64 + kAsymMaxKTiles - 1) / kAsymMaxKTiles); };
  const double I = m.intermediate, H = m.hidden;
  return 2 * I * 4 * splits(m.hidden) + 2 * I * 4 * 2 + I * 2 + H * 4 * splits(m.intermediate);
}

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------
double Calibration::rho_at(int form, double credits) const {
  auto it = rho.find(form);
  if (it == rho.end()) throw std::runtime_error("monoplan: no credit sweep for kernel form");
  return it->second.at(credits);
}

double Calibration::q_of(int form) const {
  auto it = q_form.find(form);
  return it == q_form.end() ? 1.0 : std::max(it->second, 1e-9);
}

double Calibration::link_credits() const {
  auto it = rho.find(reference_form);
  if (it == rho.end() || it->second.empty()) return 0.0;
  const Curve& r = it->second;
  // The fewest credits at which the curve reaches the saturated rate, read
  // on the piecewise-linear curve; a measured curve approaches beta, so
  // "reaches" allows a 1% gap.
  const double target = 0.99 * beta;
  for (size_t i = 0; i < r.x.size(); ++i) {
    if (r.y[i] < target) continue;
    if (i == 0) return r.x[0];
    const double t = (target - r.y[i - 1]) / std::max(r.y[i] - r.y[i - 1], 1e-30);
    return r.x[i - 1] + t * (r.x[i] - r.x[i - 1]);
  }
  return r.x.back();
}

double Calibration::exposure_bound() const {
  return link_credits() * R_C + q_max * sms * R_H;
}

// ---------------------------------------------------------------------------
// Estimator
// ---------------------------------------------------------------------------
Estimator::Estimator(const ModelShape& m, const Calibration& c, EstimatorOptions o)
    : m_(m), c_(c), o_(o) {
  if (c_.sms <= 0) throw std::runtime_error("monoplan: calibration has no SM count");
  q_link_ = c_.link_credits();
  x_max_ = c_.exposure_bound();
  gamma_max_ = c_.gamma.empty() ? 1.0 : std::max(1.0, c_.gamma.at(x_max_));
  w_ = m_.expert_bytes();
  asym_row_ = asym_row_bytes(m_);
}

double Estimator::gamma_ratio(double own) const {
  if (!o_.contention_aware || c_.gamma.empty()) return 1.0;
  const double g_own = std::max(1.0, c_.gamma.at(own));
  return std::max(1.0, gamma_max_ / g_own);
}

void Estimator::observe(bool decode, bool moe, double ratio) {
  if (!(ratio > 0) || !std::isfinite(ratio)) return;
  double& c = corr_[decode ? 1 : 0][moe ? 1 : 0];
  c = (1 - o_.ewma) * c + o_.ewma * std::clamp(ratio, 0.25, 4.0);
}

const KernelRates& Estimator::expert_rates(int form) const {
  auto it = c_.expert.find(form);
  if (it != c_.expert.end()) return it->second;
  it = c_.expert.find(c_.reference_form);
  if (it != c_.expert.end()) return it->second;
  throw std::runtime_error("monoplan: no solo sweep for expert kernel form");
}

PassLoad pass_load(const ModelShape& m, const LaneWork& w, const std::vector<Seq>& seqs, bool first,
                   bool last) {
  const double H = m.hidden, D = m.head_dim, hq = m.heads, hkv = m.kv_heads;
  const double qkv_dim = (hq + 2 * hkv) * D;
  PassLoad p;
  double keys = 0, kv_reads = 0;
  for (const Seq& s : seqs) {
    const double n = std::max(s.tokens, 0), c = std::max(s.context, 0);
    p.tokens += n;
    // causal attention: row j of the pass sees c + j + 1 keys
    keys += n * c + n * (n + 1) / 2;
    if (w.decode) {
      kv_reads += c + n;
    } else {
      // prefill tiles re-read the keys up to each 128-row query block
      for (double q0 = 0; q0 < n; q0 += 128) kv_reads += c + std::min(n, q0 + 128);
    }
  }
  p.attn_flops = 4.0 * hq * D * keys;
  // K and V in bf16; every query head's tiles read the KV of its group
  p.kv_bytes = kv_reads * hkv * D * 2.0 * 2.0 * (w.decode ? 1.0 : hq / std::max(hkv, 1.0));
  p.dense_attn_flops = 2.0 * p.tokens * H * (qkv_dim + hq * D);
  p.dense_attn_bytes = (H * qkv_dim + hq * D * H + H * m.experts) * m.dense_weight_bytes +
                       p.tokens * (2 * H + qkv_dim + 2 * hq * D) * 2.0;
  p.pairs = p.tokens * m.top_k;
  const double I = m.intermediate;
  p.moe_act_bytes = p.pairs * (2 * H + 3 * I) * 2.0 + p.tokens * H * 4.0 * 2;
  if (m.shared_intermediate > 0) {
    p.shared_flops = 6.0 * p.tokens * H * m.shared_intermediate;
    p.shared_bytes = 3.0 * H * m.shared_intermediate * m.dense_weight_bytes;
  }
  if (m.dense_layers > 0) {
    p.dense_mlp_flops = 6.0 * p.tokens * H * m.dense_intermediate;
    p.dense_mlp_bytes = 3.0 * H * m.dense_intermediate * m.dense_weight_bytes;
  }
  // At plan time a decode batch bounds M_e by its batch size; a prefill
  // batch uses its expected rows per activated expert (set per layer).
  p.link_rows_bound = w.decode ? p.tokens : 0.0;
  p.sampled = last && w.samples ? static_cast<double>(seqs.size()) : 0.0;
  p.first_layer = first ? w.first_layer : 0;
  p.active.assign(m.layers, 0.0);
  p.miss.assign(m.layers, 0.0);
  for (int l = 0; l < m.layers; ++l) {
    const double a = l < static_cast<int>(w.active.size()) ? w.active[l] : 0.0;
    const double ms = l < static_cast<int>(w.miss.size()) ? w.miss[l] : 0.0;
    p.active[l] = std::max(a, 0.0);
    p.miss[l] = std::clamp(ms, 0.0, p.active[l]);
  }
  return p;
}

LaneLoad Estimator::load(const LaneWork& w) const {
  LaneLoad l;
  l.decode = w.decode;
  l.staging = w.staging;
  l.deadline = w.deadline;
  for (size_t i = 0; i < w.passes.size(); ++i)
    l.passes.push_back(pass_load(m_, w, w.passes[i], i == 0, i + 1 == w.passes.size()));
  return l;
}

Estimate Estimator::evaluate(const LaneLoad& l, int s_in, double credits, int form) const {
  Estimate est;
  const int S = c_.sms;
  const double s = std::clamp(s_in, 1, S);
  const double H = m_.hidden, I = m_.intermediate, w = w_;
  const double RH = c_.R_H;
  const bool ca = o_.contention_aware;

  // Rates at this share, looked up once.
  const KernelRates& A = l.decode ? c_.attn_decode : c_.attn_prefill;
  const double A_f = A.flops.at(s), A_b = A.bytes.at(s);
  const double Dn_f = c_.dense.flops.at(s), Dn_b = c_.dense.bytes.at(s);
  const KernelRates& K = expert_rates(form);   // the DRAM-expert kernel
  const double K_f = K.flops.at(s), K_b = K.bytes.at(s);
  double Hf[4], Hb[4], Hq[4];
  const int heights[4] = {16, 32, 64, 128};
  for (int i = 0; i < 4; ++i) {
    const KernelRates& R = expert_rates(heights[i]);
    Hf[i] = R.flops.at(s);
    Hb[i] = R.bytes.at(s);
    Hq[i] = R.q_per_sm;
  }
  const double q_k = c_.q_of(form);
  // Credits the lane's SMs keep in flight, capped by its grant; without
  // the gate (MCA) streams keep their full depth.
  const double used = ca ? std::min(q_k * s, credits) : q_k * s;
  const double link_rate = c_.rho_at(form, used);
  const double granted_rate = c_.rho_at(form, ca ? credits : q_link_);
  const double g_attn = gamma_ratio(A.q_per_sm * s * RH);

  bool first_moe = true;
  double moe_sum = 0, staged_sum = 0;
  int moe_layers = 0;
  for (size_t pi = 0; pi < l.passes.size(); ++pi) {
    const PassLoad& p = l.passes[pi];
    const double a_cmp = safe_div(p.attn_flops, A_f) + safe_div(p.dense_attn_flops, Dn_f);
    const double a_rd = safe_div(p.kv_bytes, A_b) + safe_div(p.dense_attn_bytes, Dn_b);
    const double Ta = std::max(a_cmp, g_attn * a_rd) * correction(l.decode, false);
    if (pi == 0) {
      est.attn = Ta;
      est.a_cmp = a_cmp;
      est.a_rd = a_rd;
    }
    // Staging: during the attention window the lane's idle link share
    // moves experts into its buffer, at least g_b of them (Eq. 3).
    const double staged_cap = std::floor(granted_rate * Ta / std::max(w, 1.0));
    for (int layer = p.first_layer; layer < m_.layers; ++layer) {
      double Tm;
      if (layer < m_.dense_layers) {
        const double g = gamma_ratio(Hq[3] * s * RH);
        Tm = std::max(safe_div(p.dense_mlp_flops, Dn_f), g * safe_div(p.dense_mlp_bytes, Dn_b));
      } else {
        const double act = p.active[layer], miss = p.miss[layer];
        const double g = l.staging > 0 ? std::min({miss, static_cast<double>(l.staging), staged_cap}) : 0.0;
        const double e_dram = std::max(0.0, miss - g);
        const double e_hbm = std::max(0.0, act - e_dram);
        const double m_rows = act > 0 ? p.pairs / act : 0.0;   // expected M_e
        const int bh = hbm_tile_height(m_rows);
        const int hi = bh == 16 ? 0 : bh == 32 ? 1 : bh == 64 ? 2 : 3;
        const double fe = 6.0 * H * I;   // FLOPs per routed row of one expert
        const double rows_h = padded(m_rows, bh);
        const int bm_a = m_rows <= 16 ? 16 : m_rows <= 32 ? 32 : 64;   // asymmetric row block
        const double rows_k = form == kAsym ? padded(m_rows, bm_a) : padded(m_rows, form);
        // Compute: the HBM-expert and DRAM-expert kernels share the SMs,
        // so their times add.
        const double m_cmp = safe_div(e_hbm * fe * rows_h + p.shared_flops, Hf[hi]) +
                             safe_div(e_dram * fe * rows_k, K_f);
        // HBM reads: HBM experts once per tile row block, the activations,
        // and the asymmetric kernel's partial sums (its reduce).
        const double reads_h = std::ceil(std::max(m_rows, 1.0) / bh);
        const double reduce = form == kAsym ? e_dram * m_rows * asym_row_ : 0.0;
        const double m_rd = safe_div(e_hbm * w * reads_h + p.shared_bytes, Hb[hi]) +
                            safe_div(p.moe_act_bytes, Dn_b) + safe_div(reduce, K_b);
        const double g_moe = gamma_ratio(Hq[hi] * s * RH);
        // Link: the DRAM experts' bytes, each read once by the asymmetric
        // kernel or ceil(M_e / BM) times by a symmetric template, over the
        // rate the lane's credits buy.
        const double m_link_rows = p.link_rows_bound > 0 ? p.link_rows_bound : m_rows;
        const double reads_k = form == kAsym ? 1.0 : std::ceil(std::max(m_link_rows, 1.0) / form);
        const double m_link = safe_div(e_dram * w * reads_k, link_rate);
        Tm = std::max({m_cmp, g_moe * m_rd, m_link}) * correction(l.decode, true);
        if (pi == 0) {
          if (first_moe) {
            est.m_cmp = m_cmp;
            est.m_rd = g_moe * m_rd;
            est.m_link = m_link;
            first_moe = false;
          }
          moe_sum += Tm;
          staged_sum += g;
          ++moe_layers;
        }
      }
      est.total += Ta + Tm + c_.layer_overhead;
    }
    if (p.sampled > 0) {
      const double lm_flops = 2.0 * p.sampled * H * m_.vocab;
      const double lm_bytes = H * m_.vocab * m_.dense_weight_bytes;
      est.total += std::max(safe_div(lm_flops, Dn_f), safe_div(lm_bytes, Dn_b)) + c_.tail_overhead;
    }
  }
  if (moe_layers > 0) {
    est.moe = moe_sum / moe_layers;
    est.staged = staged_sum / moe_layers;
  }
  return est;
}

}  // namespace monoplan
