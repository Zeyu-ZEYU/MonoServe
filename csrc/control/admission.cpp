#include "control/admission.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace monoplan {

// ---------------------------------------------------------------------------
// ProfileView
// ---------------------------------------------------------------------------
ProfileView::ProfileView(int layers, int experts) : L_(layers), E_(experts) {
  if (layers <= 0 || experts <= 0) throw std::runtime_error("monoplan: empty profile");
  for (int w = 0; w < 2; ++w) p_[w].assign(static_cast<size_t>(L_) * E_, 1.0 / E_);
  hot_.assign(static_cast<size_t>(L_) * E_, 0);
  rebuild(0);
  rebuild(1);
}

void ProfileView::set_profile(bool decode, const std::vector<double>& p) {
  if (p.size() != static_cast<size_t>(L_) * E_) throw std::runtime_error("monoplan: profile shape");
  p_[decode ? 1 : 0] = p;
  rebuild(decode ? 1 : 0);
}

void ProfileView::set_hot(const std::vector<uint8_t>& hot) {
  if (hot.size() != static_cast<size_t>(L_) * E_) throw std::runtime_error("monoplan: hot tier shape");
  hot_ = hot;
  rebuild(0);
  rebuild(1);
}

void ProfileView::rebuild(int w) {
  act_[w].assign(static_cast<size_t>(L_) * kGrid, 0.0);
  miss_[w].assign(static_cast<size_t>(L_) * kGrid, 0.0);
  for (int l = 0; l < L_; ++l) {
    const double* p = &p_[w][static_cast<size_t>(l) * E_];
    const uint8_t* h = &hot_[static_cast<size_t>(l) * E_];
    for (int i = 0; i < kGrid; ++i) {
      const double n = std::exp2(0.5 * i);
      double a = 0, m = 0;
      for (int e = 0; e < E_; ++e) {
        const double pe = std::clamp(p[e], 0.0, 1.0);
        // an expert is activated unless every routed row misses it
        const double on = pe >= 1.0 ? 1.0 : 1.0 - std::exp(n * std::log1p(-pe));
        a += on;
        if (!h[e]) m += on;
      }
      act_[w][static_cast<size_t>(l) * kGrid + i] = a;
      miss_[w][static_cast<size_t>(l) * kGrid + i] = m;
    }
  }
}

std::pair<double, double> ProfileView::expected(bool decode, int layer, double pairs) const {
  if (pairs <= 0 || layer < 0 || layer >= L_) return {0.0, 0.0};
  const int w = decode ? 1 : 0;
  const double* a = &act_[w][static_cast<size_t>(layer) * kGrid];
  const double* m = &miss_[w][static_cast<size_t>(layer) * kGrid];
  if (pairs <= 1) return {a[0] * pairs, m[0] * pairs};
  const double x = 2.0 * std::log2(pairs);
  if (x >= kGrid - 1) return {a[kGrid - 1], m[kGrid - 1]};
  const int i = static_cast<int>(x);
  const double t = x - i;
  return {a[i] + t * (a[i + 1] - a[i]), m[i] + t * (m[i + 1] - m[i])};
}

// ---------------------------------------------------------------------------
// Admission
// ---------------------------------------------------------------------------
Admission::Admission(const Estimator* est, PlanSearch* search, const ProfileView* profile,
                     AdmissionConfig cfg)
    : est_(est), search_(search), prof_(profile), cfg_(cfg) {
  if (!est_ || !search_ || !prof_) throw std::runtime_error("monoplan: admission needs its parts");
  if (cfg_.max_prefill_lanes < 1 || cfg_.max_prefill_lanes > 2)
    throw std::runtime_error("monoplan: one or two prefill lanes");
  lanes_.resize(cfg_.max_prefill_lanes);
  decode_key_ = next_key_++;
}

int Admission::open_lanes() const {
  int n = 0;
  for (const Lane& l : lanes_) n += l.open ? 1 : 0;
  return n;
}

std::vector<int64_t> Admission::lane_requests(int lane) const {
  if (lane == 0) return decode_;
  if (lane < 1 || lane > static_cast<int>(lanes_.size())) return {};
  return lanes_[lane - 1].reqs;
}

double Admission::reservation(const Request& r) const {
  return (static_cast<double>(r.prompt) + std::max(r.max_output, 1)) * cfg_.kv_bytes_per_token;
}

double Admission::deadline_of(int64_t id) const {
  const Request& r = reqs_.at(id).r;
  return r.arrival + r.ttft;
}

bool Admission::memory_fits(const std::vector<int64_t>& ids) const {
  double need = kv_used_;
  for (int64_t id : ids) need += reservation(reqs_.at(id).r);
  if (cfg_.kv_capacity > 0 && need > cfg_.kv_capacity) return false;
  if (cfg_.workspace_capacity > 0 &&
      (open_lanes() + 1) * cfg_.lane_workspace > cfg_.workspace_capacity)
    return false;
  return true;
}

LaneWork Admission::decode_work(const std::vector<int64_t>& ids) const {
  LaneWork w;
  w.decode = true;
  w.staging = 0;
  w.deadline = std::numeric_limits<double>::infinity();
  std::vector<Seq> seqs;
  for (int64_t id : ids) {
    const Tracked& t = reqs_.at(id);
    seqs.push_back({1, t.r.prompt + std::max(t.generated, 1) - 1});
    w.deadline = std::min(w.deadline, t.r.tpot);
  }
  w.passes.push_back(std::move(seqs));
  const ModelShape& m = est_->model();
  const double pairs = static_cast<double>(ids.size()) * m.top_k;
  w.active.resize(m.layers);
  w.miss.resize(m.layers);
  for (int l = 0; l < m.layers; ++l) {
    const auto am = prof_->expected(true, l, pairs);
    w.active[l] = am.first;
    w.miss[l] = am.second;
  }
  return w;
}

LaneWork Admission::prefill_work(const Lane& lane, double now) const {
  LaneWork w;
  w.decode = false;
  w.staging = cfg_.staging_experts;
  w.first_layer = lane.layers_done;
  w.deadline = lane.deadline - now;
  double first_tokens = 0;
  for (size_t i = lane.pass; i < lane.passes.size(); ++i) {
    const PassSpec& p = lane.passes[i];
    std::vector<Seq> seqs;
    for (size_t k = 0; k < p.reqs.size(); ++k) {
      seqs.push_back({p.tokens[k], p.pos0[k]});
      if (i == lane.pass) first_tokens += p.tokens[k];
    }
    w.passes.push_back(std::move(seqs));
  }
  const ModelShape& m = est_->model();
  w.active.resize(m.layers);
  w.miss.resize(m.layers);
  for (int l = 0; l < m.layers; ++l) {
    const auto am = prof_->expected(false, l, first_tokens * m.top_k);
    w.active[l] = am.first;
    w.miss[l] = am.second;
  }
  return w;
}

// A sealed batch runs as one pass, or, for a single prompt longer than the
// token budget, as a few large chunks: chunking only bounds activation
// memory, since every chunk re-streams nearly the full expert set.
std::vector<PassSpec> Admission::make_passes(const std::vector<int64_t>& ids) const {
  std::vector<PassSpec> out;
  if (ids.size() == 1 && reqs_.at(ids[0]).r.prompt > cfg_.token_budget) {
    const int prompt = reqs_.at(ids[0]).r.prompt;
    const int n = (prompt + cfg_.token_budget - 1) / cfg_.token_budget;
    const int chunk = (prompt + n - 1) / n;
    for (int pos = 0; pos < prompt; pos += chunk) {
      const int t = std::min(chunk, prompt - pos);
      PassSpec p;
      p.reqs = {ids[0]};
      p.tokens = {t};
      p.pos0 = {pos};
      p.last = {static_cast<uint8_t>(pos + t == prompt)};
      out.push_back(std::move(p));
    }
    return out;
  }
  PassSpec p;
  for (int64_t id : ids) {
    p.reqs.push_back(id);
    p.tokens.push_back(reqs_.at(id).r.prompt);
    p.pos0.push_back(0);
    p.last.push_back(1);
  }
  out.push_back(std::move(p));
  return out;
}

Plan Admission::solve_for(const std::vector<int64_t>& dec, const std::vector<const Lane*>& pre,
                          double now, uint64_t key, Decision& d) {
  std::vector<LaneLoad> loads;
  if (!dec.empty()) loads.push_back(est_->load(decode_work(dec)));
  for (const Lane* l : pre) loads.push_back(est_->load(prefill_work(*l, now)));
  ++d.searches;
  return search_->solve(loads, dec.empty() ? 0 : key);
}

void Admission::reject_hopeless(double now, Decision& d) {
  if (queue_.empty() || cfg_.reject_checks <= 0) return;
  double t_free = now;
  for (const Lane& l : lanes_)
    if (l.open) t_free = std::max(t_free, l.finish_est);
  std::stable_sort(queue_.begin(), queue_.end(),
                   [&](int64_t a, int64_t b) { return deadline_of(a) < deadline_of(b); });
  int checks = 0;
  for (auto it = queue_.begin(); it != queue_.end() && checks < cfg_.reject_checks;) {
    const int64_t id = *it;
    const double D = deadline_of(id);
    bool reject = D <= t_free;
    if (!reject) {
      // Alone, after every prefill lane has finished: a single-request lane
      // beside the decode lane, with the deadline left at that time.
      Lane l;
      l.open = true;
      l.reqs = {id};
      l.passes = make_passes(l.reqs);
      l.deadline = D;
      reject = !solve_for(decode_, {&l}, t_free, decode_key_, d).feasible;
      ++checks;
    }
    if (reject) {
      reqs_.at(id).phase = kDone;
      d.rejected.push_back(id);
      reqs_.erase(id);
      it = queue_.erase(it);
    } else {
      ++it;
    }
  }
}

Decision Admission::decide(double now, bool changed) {
  const auto t0 = std::chrono::steady_clock::now();
  Decision d;
  Plan plan;
  bool have = false;
  auto open_lanes_list = [&]() {
    std::vector<const Lane*> v;
    for (int i : open_order_) v.push_back(&lanes_[i]);
    return v;
  };

  // 1. Decode first: waiting requests join if a re-solved plan keeps every
  // deadline; otherwise they keep waiting, oldest first.
  if (!join_.empty() && decode_.size() < static_cast<size_t>(cfg_.max_decode)) {
    size_t k = std::min(join_.size(), cfg_.max_decode - decode_.size());
    while (k > 0) {
      std::vector<int64_t> cand = decode_;
      cand.insert(cand.end(), join_.begin(), join_.begin() + static_cast<long>(k));
      const uint64_t key = next_key_++;
      Plan p = solve_for(cand, open_lanes_list(), now, key, d);
      if (p.feasible) {
        for (size_t i = 0; i < k; ++i) {
          const int64_t id = join_.front();
          join_.pop_front();
          reqs_.at(id).phase = kDecode;
          reqs_.at(id).lane = 0;
          d.joined.push_back(id);
        }
        decode_ = std::move(cand);
        decode_key_ = key;
        d.decode_changed = true;
        plan = std::move(p);
        have = changed = true;
        break;
      }
      k /= 2;
    }
  }

  // 2. Prefill: batches from the queue, earliest deadline first, under the
  // token budget; the first batch size whose plan fits opens a lane.
  while (!queue_.empty()) {
    int free_idx = -1;
    for (size_t i = 0; i < lanes_.size(); ++i)
      if (!lanes_[i].open) {
        free_idx = static_cast<int>(i);
        break;
      }
    if (free_idx < 0) break;
    std::stable_sort(queue_.begin(), queue_.end(),
                     [&](int64_t a, int64_t b) { return deadline_of(a) < deadline_of(b); });
    std::vector<int64_t> prefix;
    int tokens = 0;
    for (int64_t id : queue_) {
      const int p = reqs_.at(id).r.prompt;
      if (prefix.empty() && p > cfg_.token_budget) {   // a long prompt goes alone, chunked
        prefix.push_back(id);
        break;
      }
      if (tokens + p > cfg_.token_budget || prefix.size() >= static_cast<size_t>(cfg_.max_batch_requests))
        break;
      prefix.push_back(id);
      tokens += p;
    }
    bool opened = false;
    size_t m = prefix.size();
    for (int trial = 0; trial < cfg_.batch_trials && m > 0 && !opened; ++trial) {
      std::vector<int64_t> cand(prefix.begin(), prefix.begin() + static_cast<long>(m));
      if (memory_fits(cand)) {
        Lane nl;
        nl.open = true;
        nl.reqs = cand;
        nl.passes = make_passes(cand);
        nl.deadline = std::numeric_limits<double>::infinity();
        for (int64_t id : cand) nl.deadline = std::min(nl.deadline, deadline_of(id));
        std::vector<const Lane*> with = open_lanes_list();
        with.push_back(&nl);
        Plan p = solve_for(decode_, with, now, decode_key_, d);
        if (p.feasible) {
          for (int64_t id : cand) {
            Tracked& t = reqs_.at(id);
            t.phase = kPrefill;
            t.lane = free_idx + 1;
            kv_used_ += reservation(t.r);
            queue_.erase(std::find(queue_.begin(), queue_.end(), id));
          }
          lanes_[free_idx] = std::move(nl);
          open_order_.push_back(free_idx);
          d.pass_lanes.push_back(free_idx + 1);
          d.passes.push_back(lanes_[free_idx].passes[0]);
          plan = std::move(p);
          have = changed = opened = true;
        }
      }
      if (m == 1) break;
      m = std::max<size_t>(1, m / 2);
    }
    if (!opened) break;
  }

  // 3. Rejection of requests no plan can serve in time.
  reject_hopeless(now, d);

  // 4. The plan of the resulting lane set.
  if (changed && !have) {
    plan = solve_for(decode_, open_lanes_list(), now, decode_key_, d);
    have = true;
  }
  if (changed) {
    d.publish = true;
    d.feasible = plan.feasible;
    d.lanes.clear();
    if (!decode_.empty()) d.lanes.push_back(0);
    for (int i : open_order_) d.lanes.push_back(i + 1);
    for (size_t i = 0; i < d.lanes.size() && i < plan.lanes.size(); ++i)
      if (d.lanes[i] > 0) lanes_[d.lanes[i] - 1].finish_est = now + plan.lanes[i].est.total;
    plan_ = plan;
    d.plan = std::move(plan);
  }
  d.decode = decode_;
  d.micros = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  return d;
}

Decision Admission::arrive(const Request& r, double now) {
  if (reqs_.count(r.id)) throw std::runtime_error("monoplan: duplicate request id");
  Tracked t;
  t.r = r;
  reqs_.emplace(r.id, t);
  queue_.push_back(r.id);
  return decide(now, false);
}

Decision Admission::pass_done(int lane, double now) {
  if (lane < 1 || lane > static_cast<int>(lanes_.size()) || !lanes_[lane - 1].open)
    throw std::runtime_error("monoplan: pass_done on a lane that is not open");
  Lane& l = lanes_[lane - 1];
  if (l.pass + 1 < l.passes.size()) {
    ++l.pass;
    l.layers_done = 0;
    Decision d = decide(now, true);
    d.pass_lanes.insert(d.pass_lanes.begin(), lane);
    d.passes.insert(d.passes.begin(), lanes_[lane - 1].passes[lanes_[lane - 1].pass]);
    return d;
  }
  // The prompts are done and their first tokens are out: they wait to join
  // the decode batch.
  for (int64_t id : l.reqs) {
    auto it = reqs_.find(id);
    if (it == reqs_.end()) continue;
    if (it->second.phase == kPrefill) {
      it->second.phase = kJoin;
      it->second.generated = std::max(it->second.generated, 1);
      join_.push_back(id);
    } else if (it->second.phase == kDone) {
      reqs_.erase(it);
    }
  }
  l = Lane{};
  open_order_.erase(std::remove(open_order_.begin(), open_order_.end(), lane - 1), open_order_.end());
  return decide(now, true);
}

Decision Admission::finished(int64_t id, double now) {
  auto it = reqs_.find(id);
  if (it == reqs_.end()) return Decision{};
  Tracked& t = it->second;
  bool decode_changed = false;
  switch (t.phase) {
    case kDecode:
      decode_.erase(std::remove(decode_.begin(), decode_.end(), id), decode_.end());
      decode_key_ = next_key_++;
      decode_changed = true;
      break;
    case kJoin:
      join_.erase(std::remove(join_.begin(), join_.end(), id), join_.end());
      break;
    case kQueued:
      queue_.erase(std::remove(queue_.begin(), queue_.end(), id), queue_.end());
      break;
    default:
      break;
  }
  if (t.phase == kDecode || t.phase == kJoin || t.phase == kPrefill) kv_used_ -= reservation(t.r);
  if (kv_used_ < 0) kv_used_ = 0;
  if (t.phase == kPrefill) {
    t.phase = kDone;   // its lane's pass still runs; pass_done drops it
  } else {
    reqs_.erase(it);
  }
  Decision d = decide(now, decode_changed);
  if (decode_changed) d.decode_changed = true;
  return d;
}

Decision Admission::behind(int lane, double now) {
  (void)lane;
  return decide(now, true);
}

void Admission::progress(int lane, int layers_done) {
  if (lane >= 1 && lane <= static_cast<int>(lanes_.size())) lanes_[lane - 1].layers_done = layers_done;
}

void Admission::generated(int64_t id, int tokens) {
  auto it = reqs_.find(id);
  if (it != reqs_.end()) it->second.generated = tokens;
}

}  // namespace monoplan
