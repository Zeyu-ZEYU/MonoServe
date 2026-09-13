// Plan search (Algorithm 1 of the paper), one call per candidate lane set.
//
// Once the exposure bound X_max is fixed, the estimate of one lane never
// depends on another, so the search finds, per lane, kernel form, and grant
// on the rho grid, the deadline floor (the smallest SM share meeting the
// deadline) by binary search. The decode lane is solved first; prefill
// candidates enumerate one form per lane and one split of the remaining
// credits; spare SMs are poured onto the least-slack lane; the kept
// candidate is re-evaluated exactly at its rounded shares and grants.
#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "control/estimator.h"

namespace monoplan {

struct SearchOptions {
  std::vector<int> forms{16, 32, 64, 128, kAsym};
  int pour_quantum = 2;          // SMs per pouring step
  int pour_candidates = 8;       // feasible candidates poured and compared
  int decode_choices = 4;        // decode choices tried before giving up
  double tolerance = 0.01;       // relative slack on the two link budgets
};

struct LanePlan {
  int sms = 0;                   // s_b, the lane's SM share (a floor)
  double credits = 0;            // Q^link_b, rounded to whole tiles
  int form = 64;                 // kappa for its DRAM experts
  int cap = 0;                   // cap_b = floor(Q^link_b / q(kappa)) link slots
  double rate = 0;               // beta_b = rho_kappa(Q^link_b)
  double slack = 0;              // (deadline - T-hat_b) / deadline
  Estimate est;
};

struct Plan {
  bool feasible = false;
  std::vector<LanePlan> lanes;   // in the order of the lanes given to solve()
  double copy_rate = 0;          // link rate the lanes' grants leave to the copy engine
  double min_slack = 0;
  double total_time = 0;
  long evaluations = 0;
  double micros = 0;
};

class PlanSearch {
 public:
  explicit PlanSearch(const Estimator* est, SearchOptions opt = {});

  // lanes: the candidate lane set, at most one decode lane (first) and at
  // most two prefill lanes. decode_key names the decode lane's composition;
  // its choices are cached under it (0: no caching).
  Plan solve(const std::vector<LaneLoad>& lanes, uint64_t decode_key = 0);

  // Smallest SM share meeting the deadline at this grant and form; S + 1
  // when even all S SMs miss it.
  int deadline_floor(const LaneLoad& l, double credits, int form, long* evals = nullptr) const;
  // Grants tried per form: the rho grid below Q^link, and Q^link.
  std::vector<double> grants(int form) const;
  // The most credits of a form whose link rate stays within `rate`.
  double credits_for_rate(int form, double rate) const;

  void clear_cache() { cache_.clear(); }
  const SearchOptions& options() const { return opt_; }
  const Estimator& estimator() const { return *est_; }

 private:
  struct Choice {
    int form;
    double credits;
    int floor;
  };
  struct Cand {
    std::vector<int> form, share;
    std::vector<double> credits, slack;
    std::vector<Estimate> est;
    int spare = 0;
    double min_slack = 0, total = 0;
  };
  void pour(const std::vector<LaneLoad>& lanes, Cand& c, long* evals) const;
  bool verify(const std::vector<LaneLoad>& lanes, Cand& c, Plan& out, long* evals) const;
  Plan best_effort(const std::vector<LaneLoad>& lanes, const std::vector<Choice>& dec, long* evals) const;

  const Estimator* est_;
  SearchOptions opt_;
  std::unordered_map<uint64_t, std::vector<Choice>> cache_;
};

}  // namespace monoplan
