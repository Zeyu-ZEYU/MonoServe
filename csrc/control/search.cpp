#include "control/search.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>

namespace monoplan {

namespace {
constexpr double kNeg = -1e30;

double slack_of(const LaneLoad& l, double t) {
  return l.deadline > 0 ? (l.deadline - t) / l.deadline : kNeg;
}
}  // namespace

PlanSearch::PlanSearch(const Estimator* est, SearchOptions opt) : est_(est), opt_(std::move(opt)) {
  if (!est_) throw std::runtime_error("monoplan: plan search needs an estimator");
  if (opt_.forms.empty()) throw std::runtime_error("monoplan: no kernel forms to search");
}

double PlanSearch::credits_for_rate(int form, double rate) const {
  const auto& rho = est_->calibration().rho;
  auto it = rho.find(form);
  if (rate <= 0 || it == rho.end() || it->second.empty()) return 0.0;
  const Curve& c = it->second;
  // the most credits whose rate stays within `rate`: the curve rises to its
  // last point and stays flat beyond it
  if (rate >= c.y.back()) return std::numeric_limits<double>::infinity();
  if (rate < c.y.front())
    return c.origin && c.y.front() > 0 ? c.x.front() * rate / c.y.front() : 0.0;
  for (size_t i = 1; i < c.x.size(); ++i)
    if (c.y[i] > rate) {
      const double t = (rate - c.y[i - 1]) / (c.y[i] - c.y[i - 1]);
      return c.x[i - 1] + t * (c.x[i] - c.x[i - 1]);
    }
  return c.x.back();
}

std::vector<double> PlanSearch::grants(int form) const {
  const double qlink = est_->link_credits();
  std::vector<double> g;
  if (est_->options().contention_aware) {
    auto it = est_->calibration().rho.find(form);
    if (it != est_->calibration().rho.end())
      for (double x : it->second.x)
        if (x > 0 && x < qlink * (1 - 1e-9)) g.push_back(x);
  }
  g.push_back(qlink);
  return g;
}

int PlanSearch::deadline_floor(const LaneLoad& l, double credits, int form, long* evals) const {
  const int S = est_->calibration().sms;
  auto meets = [&](int s) {
    if (evals) ++*evals;
    return est_->evaluate(l, s, credits, form).total <= l.deadline;
  };
  if (!meets(S)) return S + 1;
  if (meets(1)) return 1;
  int lo = 1, hi = S;   // lo misses, hi meets; the estimate falls as the share grows
  while (hi - lo > 1) {
    const int mid = (lo + hi) / 2;
    if (meets(mid)) hi = mid;
    else lo = mid;
  }
  return hi;
}

// Leftover SMs go to the least-slack lane, repeatedly. A lane whose
// estimate stops improving drops out; whatever is left after that still
// goes to the tightest lane, since shares are floors, not fences.
void PlanSearch::pour(const std::vector<LaneLoad>& lanes, Cand& c, long* evals) const {
  const int S = est_->calibration().sms;
  const size_t n = lanes.size();
  c.est.resize(n);
  c.slack.resize(n);
  int used = 0;
  for (size_t b = 0; b < n; ++b) {
    c.est[b] = est_->evaluate(lanes[b], c.share[b], c.credits[b], c.form[b]);
    ++*evals;
    c.slack[b] = slack_of(lanes[b], c.est[b].total);
    used += c.share[b];
  }
  int spare = S - used;
  std::vector<char> done(n, 0);
  while (spare > 0) {
    int b = -1;
    for (size_t i = 0; i < n; ++i)
      if (!done[i] && (b < 0 || c.slack[i] < c.slack[b])) b = static_cast<int>(i);
    if (b < 0) break;
    const int add = std::min(opt_.pour_quantum, spare);
    const Estimate e = est_->evaluate(lanes[b], c.share[b] + add, c.credits[b], c.form[b]);
    ++*evals;
    if (e.total < c.est[b].total * (1 - 1e-9)) {
      c.share[b] += add;
      spare -= add;
      c.est[b] = e;
      c.slack[b] = slack_of(lanes[b], e.total);
    } else {
      done[b] = 1;
    }
  }
  if (spare > 0 && n > 0) {
    size_t b = 0;
    for (size_t i = 1; i < n; ++i)
      if (c.slack[i] < c.slack[b]) b = i;
    c.share[b] += spare;
  }
  c.min_slack = n ? c.slack[0] : 0;
  c.total = 0;
  for (size_t b = 0; b < n; ++b) {
    c.min_slack = std::min(c.min_slack, c.slack[b]);
    c.total += c.est[b].total;
  }
}

// Exact re-evaluation at the rounded shares and grants: a grant becomes a
// whole number of link slots, at least one, so no lane can stall on a
// DRAM expert.
bool PlanSearch::verify(const std::vector<LaneLoad>& lanes, Cand& c, Plan& out, long* evals) const {
  const Calibration& cal = est_->calibration();
  const double qlink = est_->link_credits();
  int used = 0;
  double credits = 0, rate = 0;
  bool ok = true;
  out.lanes.assign(lanes.size(), LanePlan{});
  out.min_slack = lanes.empty() ? 0 : 1e30;
  out.total_time = 0;
  for (size_t b = 0; b < lanes.size(); ++b) {
    LanePlan& p = out.lanes[b];
    const double q = cal.q_of(c.form[b]);
    p.cap = std::max(1, static_cast<int>(std::floor(c.credits[b] / q + 1e-9)));
    p.credits = p.cap * q;
    p.form = c.form[b];
    p.sms = c.share[b];
    p.est = est_->evaluate(lanes[b], p.sms, p.credits, p.form);
    ++*evals;
    p.rate = cal.rho_at(p.form, p.credits);
    p.slack = slack_of(lanes[b], p.est.total);
    if (p.est.total > lanes[b].deadline) ok = false;
    used += p.sms;
    credits += p.credits;
    rate += p.rate;
    out.min_slack = std::min(out.min_slack, p.slack);
    out.total_time += p.est.total;
  }
  if (used > cal.sms) ok = false;
  // without contention awareness there is no credit gate to hold anything to
  if (est_->options().contention_aware) {
    if (credits > qlink * (1 + opt_.tolerance) + cal.q_of(cal.reference_form) * lanes.size()) ok = false;
    if (rate > cal.beta * (1 + opt_.tolerance)) ok = false;
  }
  out.copy_rate = std::max(0.0, cal.beta - rate);
  out.feasible = ok;
  return ok;
}

Plan PlanSearch::best_effort(const std::vector<LaneLoad>& lanes, const std::vector<Choice>& dec,
                             long* evals) const {
  const Calibration& cal = est_->calibration();
  const size_t n = lanes.size();
  Cand c;
  c.form.assign(n, cal.reference_form);
  c.credits.assign(n, est_->link_credits() / std::max<size_t>(n, 1));
  c.share.assign(n, 1);
  if (n && lanes[0].decode && !dec.empty()) c.form[0] = dec.front().form;
  pour(lanes, c, evals);
  Plan out;
  verify(lanes, c, out, evals);
  out.feasible = false;
  return out;
}

Plan PlanSearch::solve(const std::vector<LaneLoad>& lanes, uint64_t decode_key) {
  const auto t0 = std::chrono::steady_clock::now();
  long evals = 0;
  const Calibration& cal = est_->calibration();
  const int S = cal.sms;
  const double qlink = est_->link_credits();
  const size_t n = lanes.size();
  const bool has_dec = n > 0 && lanes[0].decode;
  const size_t first_p = has_dec ? 1 : 0;
  const size_t n_p = n - first_p;
  for (size_t b = first_p; b < n; ++b)
    if (lanes[b].decode) throw std::runtime_error("monoplan: the decode lane must come first");
  if (n_p > 2) throw std::runtime_error("monoplan: at most two prefill lanes per candidate");

  auto finish = [&](Plan p) {
    p.evaluations = evals;
    p.micros = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    return p;
  };
  if (n == 0) {
    Plan p;
    p.feasible = true;
    p.copy_rate = cal.beta;
    return finish(p);
  }

  // Decode first: its cheapest form and grant by floor, then by credits,
  // with the rest as fallbacks; cached until its composition changes.
  std::vector<Choice> dec;
  if (has_dec) {
    auto it = decode_key ? cache_.find(decode_key) : cache_.end();
    if (it != cache_.end()) {
      dec = it->second;
    } else {
      for (int f : opt_.forms)
        for (double q : grants(f)) {
          const int fl = deadline_floor(lanes[0], q, f, &evals);
          if (fl <= S) dec.push_back({f, q, fl});
        }
      std::sort(dec.begin(), dec.end(), [](const Choice& a, const Choice& b) {
        return a.floor != b.floor ? a.floor < b.floor : a.credits < b.credits;
      });
      if (decode_key) {
        if (cache_.size() > 256) cache_.clear();
        cache_[decode_key] = dec;
      }
    }
    if (dec.empty()) return finish(best_effort(lanes, dec, &evals));
  }

  std::map<std::tuple<size_t, int, long long>, int> memo;
  auto pfloor = [&](size_t b, int f, double q) {
    const auto k = std::make_tuple(b, f, std::llround(q * 1024));
    auto it = memo.find(k);
    if (it != memo.end()) return it->second;
    const int v = deadline_floor(lanes[b], q, f, &evals);
    memo[k] = v;
    return v;
  };

  const size_t tries = has_dec ? std::min<size_t>(dec.size(), std::max(1, opt_.decode_choices)) : 1;
  for (size_t di = 0; di < tries; ++di) {
    Cand base;
    base.form.assign(n, cal.reference_form);
    base.credits.assign(n, 0.0);
    base.share.assign(n, 0);
    double q_rem = qlink;
    if (has_dec) {
      const Choice& d = dec[di];
      base.form[0] = d.form;
      if (n_p == 0) {
        // Alone, the decode lane is granted the whole budget.
        base.credits[0] = qlink;
        base.share[0] = std::min(d.floor, pfloor(0, d.form, qlink));
      } else {
        base.credits[0] = d.credits;
        base.share[0] = d.floor;
      }
      q_rem = std::max(0.0, qlink - base.credits[0]);
    }

    std::vector<Cand> cands;
    const bool gate = est_->options().contention_aware;
    auto push = [&](Cand c) {
      int used = 0;
      double rate = 0;
      for (size_t b = 0; b < n; ++b) {
        if (c.share[b] > S) return;
        used += c.share[b];
        rate += cal.rho_at(c.form[b], c.credits[b]);
      }
      if (used > S || (gate && rate > cal.beta * (1 + opt_.tolerance))) return;
      c.spare = S - used;
      cands.push_back(std::move(c));
    };
    // A lane's own estimate never grows with its grant and no other lane's
    // depends on it, so each prefill lane takes the largest grant the
    // credits left and the link rate left allow. Where the credit curve
    // bends below its straight line, the whole budget would buy more rate
    // than the link carries; the credits that could only be spent above it
    // stay unused.
    const double rate_cap = cal.beta * (1 + opt_.tolerance);
    const double rate_dec = has_dec ? cal.rho_at(base.form[0], base.credits[0]) : 0.0;
    auto largest = [&](int f, double credits_left, double rate_left) {
      if (!gate) return credits_left;
      return std::max(0.0, std::min(credits_left, credits_for_rate(f, rate_left)));
    };
    if (n_p == 0) {
      push(base);
    } else if (n_p == 1) {
      for (int f : opt_.forms) {
        Cand c = base;
        c.form[first_p] = f;
        c.credits[first_p] = largest(f, q_rem, rate_cap - rate_dec);
        c.share[first_p] = pfloor(first_p, f, c.credits[first_p]);
        push(c);
      }
    } else {
      for (int f1 : opt_.forms)
        for (int f2 : opt_.forms)
          for (double q1 : grants(f1)) {
            if (q1 > q_rem * (1 + 1e-9)) continue;
            const double left = rate_cap - rate_dec - (gate ? cal.rho_at(f1, q1) : 0.0);
            if (left < 0) continue;
            Cand c = base;
            c.form[first_p] = f1;
            c.form[first_p + 1] = f2;
            c.credits[first_p] = q1;
            c.credits[first_p + 1] = largest(f2, q_rem - q1, left);
            c.share[first_p] = pfloor(first_p, f1, q1);
            if (c.share[first_p] > S) continue;
            c.share[first_p + 1] = pfloor(first_p + 1, f2, c.credits[first_p + 1]);
            push(c);
          }
    }
    if (cands.empty()) continue;   // retry with the next decode choice

    // Pour the most promising candidates and keep the largest minimum
    // slack, ties broken by the smallest total estimated time.
    std::stable_sort(cands.begin(), cands.end(),
                     [](const Cand& a, const Cand& b) { return a.spare > b.spare; });
    if (cands.size() > static_cast<size_t>(opt_.pour_candidates)) cands.resize(opt_.pour_candidates);
    for (Cand& c : cands) pour(lanes, c, &evals);
    std::stable_sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
      if (std::abs(a.min_slack - b.min_slack) > 1e-12) return a.min_slack > b.min_slack;
      return a.total < b.total;
    });
    for (Cand& c : cands) {
      Plan out;
      if (verify(lanes, c, out, &evals)) return finish(out);
    }
  }
  return finish(best_effort(lanes, dec, &evals));
}

}  // namespace monoplan
