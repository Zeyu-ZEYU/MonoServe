// Admission: decides what runs at admission events (a prefill request
// arriving, a request finishing prefill or decode, a lane falling behind the
// pace its plan published). Each decision is one plan search per candidate
// lane set. Decode comes first: a request that finished prefill joins the
// running decode batch if a re-solved plan keeps every deadline and waits
// otherwise. Prefill requests are then batched from the queue, earliest
// deadline first, under a token budget; a batch that fits opens a lane and
// is sealed. Memory is part of every test. A queued request is rejected only
// when no plan can reach its deadline even after every prefill lane has
// finished.
#pragma once
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <utility>
#include <vector>

#include "control/estimator.h"
#include "control/search.h"

namespace monoplan {

// Expected expert activations per layer from an activation profile, with
// the hot tier marked: what the estimator uses before a router fires.
class ProfileView {
 public:
  ProfileView(int layers, int experts);
  // p: [layers * experts], each layer's activation shares (summing to 1).
  // decode = true sets the decode lane's profile, false the prefill one.
  void set_profile(bool decode, const std::vector<double>& p);
  // hot: [layers * experts], 1 for an expert the hot tier holds.
  void set_hot(const std::vector<uint8_t>& hot);
  // Expected activated experts of a layer, and those outside the hot tier,
  // when `pairs` routed rows (tokens times top-k) pick by the profile.
  std::pair<double, double> expected(bool decode, int layer, double pairs) const;
  int layers() const { return L_; }
  int experts() const { return E_; }

 private:
  void rebuild(int which);
  static constexpr int kGrid = 49;   // pairs = 2^(i/2), i < kGrid
  int L_, E_;
  std::vector<double> p_[2];
  std::vector<uint8_t> hot_;
  std::vector<double> act_[2], miss_[2];   // [layer * kGrid + i]
};

struct Request {
  int64_t id = 0;
  int prompt = 0;          // prompt tokens
  int max_output = 1;      // output tokens reserved in the KV cache
  double arrival = 0;      // seconds
  double ttft = 0;         // TTFT target, seconds after arrival
  double tpot = 0;         // TPOT target, seconds
};

struct AdmissionConfig {
  int max_prefill_lanes = 2;
  int token_budget = 16384;        // prompt tokens per prefill pass (activation memory)
  int max_batch_requests = 64;     // requests per prefill batch
  int max_decode = 256;            // decode batch capacity
  int staging_experts = 0;         // V of each prefill lane's staging buffer
  double kv_capacity = 0;          // KV-cache bytes (0: unlimited)
  double kv_bytes_per_token = 0;
  double workspace_capacity = 0;   // bytes for lane activations and workspaces (0: unlimited)
  double lane_workspace = 0;       // bytes one prefill lane needs
  int batch_trials = 4;            // candidate batch sizes tried per new lane
  int reject_checks = 2;           // queued requests tested for rejection per event
};

// One prefill pass of a lane: the tokens of each request it computes.
struct PassSpec {
  std::vector<int64_t> reqs;
  std::vector<int> tokens;         // tokens of each request in this pass
  std::vector<int> pos0;           // position of each request's first token in it
  std::vector<uint8_t> last;       // 1: the prompt ends here, so its first token is sampled
};

struct Decision {
  bool publish = false;            // a new plan must be published
  bool feasible = true;
  Plan plan;
  std::vector<int> lanes;          // fabric lane of plan.lanes[i] (0: decode)
  bool decode_changed = false;
  std::vector<int64_t> decode;     // the decode batch after this decision
  std::vector<int64_t> joined;     // requests that entered the decode batch
  std::vector<int64_t> rejected;
  std::vector<int> pass_lanes;     // prefill passes to start: the lane ...
  std::vector<PassSpec> passes;    // ... and its pass
  int searches = 0;
  double micros = 0;
};

class Admission {
 public:
  Admission(const Estimator* est, PlanSearch* search, const ProfileView* profile,
            AdmissionConfig cfg);

  // Admission events.
  Decision arrive(const Request& r, double now);
  Decision pass_done(int lane, double now);        // a prefill pass completed
  Decision finished(int64_t id, double now);       // a request produced its last token
  Decision behind(int lane, double now);           // a lane fell behind its published pace
  // Progress between events.
  void progress(int lane, int layers_done);        // layers of the current pass done
  void generated(int64_t id, int tokens);          // output tokens produced so far

  size_t queued() const { return queue_.size(); }
  size_t waiting() const { return join_.size(); }
  const std::vector<int64_t>& decode_batch() const { return decode_; }
  std::vector<int64_t> lane_requests(int lane) const;
  int open_lanes() const;
  double kv_reserved() const { return kv_used_; }
  const Plan& plan() const { return plan_; }
  const AdmissionConfig& config() const { return cfg_; }

 private:
  enum Phase : int { kQueued, kPrefill, kJoin, kDecode, kDone };
  struct Tracked {
    Request r;
    Phase phase = kQueued;
    int generated = 0;
    int lane = -1;
  };
  struct Lane {
    bool open = false;
    std::vector<int64_t> reqs;
    std::vector<PassSpec> passes;
    size_t pass = 0;
    int layers_done = 0;
    double deadline = 0;           // absolute: the tightest member's TTFT deadline
    double finish_est = 0;         // absolute, from the last published plan
  };

  Decision decide(double now, bool changed);
  LaneWork decode_work(const std::vector<int64_t>& ids) const;
  LaneWork prefill_work(const Lane& l, double now) const;
  std::vector<PassSpec> make_passes(const std::vector<int64_t>& ids) const;
  double reservation(const Request& r) const;
  bool memory_fits(const std::vector<int64_t>& ids) const;
  Plan solve_for(const std::vector<int64_t>& dec, const std::vector<const Lane*>& pre, double now,
                 uint64_t key, Decision& d);
  void reject_hopeless(double now, Decision& d);
  double deadline_of(int64_t id) const;

  const Estimator* est_;
  PlanSearch* search_;
  const ProfileView* prof_;
  AdmissionConfig cfg_;
  std::unordered_map<int64_t, Tracked> reqs_;
  std::vector<int64_t> queue_;
  std::deque<int64_t> join_;
  std::vector<int64_t> decode_;
  std::vector<Lane> lanes_;        // prefill lanes; fabric lane = index + 1
  std::vector<int> open_order_;    // open prefill lanes, in plan order
  uint64_t decode_key_ = 0, next_key_ = 1;
  double kv_used_ = 0;
  Plan plan_;
};

}  // namespace monoplan
