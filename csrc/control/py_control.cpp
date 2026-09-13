// Python bindings of the control plane (monoserve._C.control).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "control/admission.h"

namespace py = pybind11;
using namespace monoplan;

void bind_control(py::module_& m) {
  m.attr("ASYM") = kAsym;

  py::class_<Curve>(m, "Curve")
      .def(py::init<>())
      .def(py::init([](std::vector<double> x, std::vector<double> y, bool origin) {
             if (x.size() != y.size()) throw std::runtime_error("curve: x and y differ in length");
             for (size_t i = 1; i < x.size(); ++i)
               if (!(x[i] > x[i - 1])) throw std::runtime_error("curve: x must increase");
             Curve c;
             c.x = std::move(x);
             c.y = std::move(y);
             c.origin = origin;
             return c;
           }),
           py::arg("x"), py::arg("y"), py::arg("origin") = false)
      .def_readwrite("x", &Curve::x)
      .def_readwrite("y", &Curve::y)
      .def_readwrite("origin", &Curve::origin)
      .def("__call__", &Curve::at);

  py::class_<ModelShape>(m, "ModelShape")
      .def(py::init<>())
      .def_readwrite("hidden", &ModelShape::hidden)
      .def_readwrite("intermediate", &ModelShape::intermediate)
      .def_readwrite("experts", &ModelShape::experts)
      .def_readwrite("top_k", &ModelShape::top_k)
      .def_readwrite("layers", &ModelShape::layers)
      .def_readwrite("heads", &ModelShape::heads)
      .def_readwrite("kv_heads", &ModelShape::kv_heads)
      .def_readwrite("head_dim", &ModelShape::head_dim)
      .def_readwrite("vocab", &ModelShape::vocab)
      .def_readwrite("dense_layers", &ModelShape::dense_layers)
      .def_readwrite("dense_intermediate", &ModelShape::dense_intermediate)
      .def_readwrite("shared_intermediate", &ModelShape::shared_intermediate)
      .def_readwrite("expert_weight_bytes", &ModelShape::expert_weight_bytes)
      .def_readwrite("dense_weight_bytes", &ModelShape::dense_weight_bytes)
      .def_property_readonly("expert_bytes", &ModelShape::expert_bytes)
      .def_property_readonly("kv_bytes_per_token", &ModelShape::kv_bytes_per_token);

  py::class_<KernelRates>(m, "KernelRates")
      .def(py::init<>())
      .def_readwrite("flops", &KernelRates::flops)
      .def_readwrite("bytes", &KernelRates::bytes)
      .def_readwrite("q_per_sm", &KernelRates::q_per_sm);

  py::class_<Calibration>(m, "Calibration")
      .def(py::init<>())
      .def_readwrite("sms", &Calibration::sms)
      .def_readwrite("R_H", &Calibration::R_H)
      .def_readwrite("R_C", &Calibration::R_C)
      .def_readwrite("gamma", &Calibration::gamma)
      .def_readwrite("q_max", &Calibration::q_max)
      .def_readwrite("dense", &Calibration::dense)
      .def_readwrite("attn_prefill", &Calibration::attn_prefill)
      .def_readwrite("attn_decode", &Calibration::attn_decode)
      .def_readwrite("expert", &Calibration::expert)
      .def_readwrite("rho", &Calibration::rho)
      .def_readwrite("q_form", &Calibration::q_form)
      .def_readwrite("beta", &Calibration::beta)
      .def_readwrite("reference_form", &Calibration::reference_form)
      .def_readwrite("layer_overhead", &Calibration::layer_overhead)
      .def_readwrite("tail_overhead", &Calibration::tail_overhead)
      .def("link_credits", &Calibration::link_credits)
      .def("exposure_bound", &Calibration::exposure_bound)
      .def("rho_at", &Calibration::rho_at);

  py::class_<Seq>(m, "Seq")
      .def(py::init([](int tokens, int context) { return Seq{tokens, context}; }),
           py::arg("tokens") = 1, py::arg("context") = 0)
      .def_readwrite("tokens", &Seq::tokens)
      .def_readwrite("context", &Seq::context);

  py::class_<LaneWork>(m, "LaneWork")
      .def(py::init<>())
      .def_readwrite("decode", &LaneWork::decode)
      .def_readwrite("passes", &LaneWork::passes)
      .def_readwrite("first_layer", &LaneWork::first_layer)
      .def_readwrite("active", &LaneWork::active)
      .def_readwrite("miss", &LaneWork::miss)
      .def_readwrite("staging", &LaneWork::staging)
      .def_readwrite("deadline", &LaneWork::deadline)
      .def_readwrite("samples", &LaneWork::samples);

  m.def("asym_row_bytes", &asym_row_bytes, py::arg("model"));
  m.def("pass_load",
        [](const ModelShape& model, const LaneWork& w) {
          if (w.passes.empty()) throw std::runtime_error("pass_load: no pass");
          const PassLoad p = pass_load(model, w, w.passes[0], true, w.passes.size() == 1);
          py::dict d;
          d["tokens"] = p.tokens;
          d["attn_flops"] = p.attn_flops;
          d["kv_bytes"] = p.kv_bytes;
          d["dense_attn_flops"] = p.dense_attn_flops;
          d["dense_attn_bytes"] = p.dense_attn_bytes;
          d["pairs"] = p.pairs;
          d["moe_act_bytes"] = p.moe_act_bytes;
          return d;
        },
        py::arg("model"), py::arg("work"));

  py::class_<LaneLoad>(m, "LaneLoad")
      .def_readonly("decode", &LaneLoad::decode)
      .def_readwrite("deadline", &LaneLoad::deadline);

  py::class_<Estimate>(m, "Estimate")
      .def_readonly("total", &Estimate::total)
      .def_readonly("attn", &Estimate::attn)
      .def_readonly("moe", &Estimate::moe)
      .def_readonly("staged", &Estimate::staged)
      .def_readonly("a_cmp", &Estimate::a_cmp)
      .def_readonly("a_rd", &Estimate::a_rd)
      .def_readonly("m_cmp", &Estimate::m_cmp)
      .def_readonly("m_rd", &Estimate::m_rd)
      .def_readonly("m_link", &Estimate::m_link);

  py::class_<EstimatorOptions>(m, "EstimatorOptions")
      .def(py::init<>())
      .def_readwrite("contention_aware", &EstimatorOptions::contention_aware)
      .def_readwrite("ewma", &EstimatorOptions::ewma);

  py::class_<Estimator>(m, "Estimator")
      .def(py::init<const ModelShape&, const Calibration&, EstimatorOptions>(), py::arg("model"),
           py::arg("calibration"), py::arg("options") = EstimatorOptions{})
      .def("load", &Estimator::load)
      .def("evaluate", &Estimator::evaluate, py::arg("load"), py::arg("sms"), py::arg("credits"),
           py::arg("form"))
      .def("gamma_ratio", &Estimator::gamma_ratio)
      .def("observe", &Estimator::observe, py::arg("decode"), py::arg("moe"), py::arg("ratio"))
      .def("correction", &Estimator::correction)
      .def_property_readonly("link_credits", &Estimator::link_credits)
      .def_property_readonly("exposure_bound", &Estimator::exposure_bound)
      .def_property_readonly("model", &Estimator::model)
      .def_property_readonly("calibration", &Estimator::calibration);

  py::class_<SearchOptions>(m, "SearchOptions")
      .def(py::init<>())
      .def_readwrite("forms", &SearchOptions::forms)
      .def_readwrite("pour_quantum", &SearchOptions::pour_quantum)
      .def_readwrite("pour_candidates", &SearchOptions::pour_candidates)
      .def_readwrite("decode_choices", &SearchOptions::decode_choices)
      .def_readwrite("tolerance", &SearchOptions::tolerance);

  py::class_<LanePlan>(m, "LanePlan")
      .def_readonly("sms", &LanePlan::sms)
      .def_readonly("credits", &LanePlan::credits)
      .def_readonly("form", &LanePlan::form)
      .def_readonly("cap", &LanePlan::cap)
      .def_readonly("rate", &LanePlan::rate)
      .def_readonly("slack", &LanePlan::slack)
      .def_readonly("est", &LanePlan::est);

  py::class_<Plan>(m, "Plan")
      .def_readonly("feasible", &Plan::feasible)
      .def_readonly("lanes", &Plan::lanes)
      .def_readonly("copy_rate", &Plan::copy_rate)
      .def_readonly("min_slack", &Plan::min_slack)
      .def_readonly("total_time", &Plan::total_time)
      .def_readonly("evaluations", &Plan::evaluations)
      .def_readonly("micros", &Plan::micros);

  py::class_<PlanSearch>(m, "PlanSearch")
      .def(py::init<const Estimator*, SearchOptions>(), py::arg("estimator"),
           py::arg("options") = SearchOptions{}, py::keep_alive<1, 2>())
      .def("solve", &PlanSearch::solve, py::arg("lanes"), py::arg("decode_key") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("deadline_floor",
           [](const PlanSearch& s, const LaneLoad& l, double credits, int form) {
             return s.deadline_floor(l, credits, form, nullptr);
           })
      .def("grants", &PlanSearch::grants)
      .def("clear_cache", &PlanSearch::clear_cache);

  py::class_<ProfileView>(m, "ProfileView")
      .def(py::init<int, int>(), py::arg("layers"), py::arg("experts"))
      .def("set_profile", &ProfileView::set_profile, py::arg("decode"), py::arg("p"))
      .def("set_hot", &ProfileView::set_hot, py::arg("hot"))
      .def("expected", &ProfileView::expected, py::arg("decode"), py::arg("layer"), py::arg("pairs"))
      .def_property_readonly("layers", &ProfileView::layers)
      .def_property_readonly("experts", &ProfileView::experts);

  py::class_<Request>(m, "Request")
      .def(py::init([](int64_t id, int prompt, int max_output, double arrival, double ttft,
                       double tpot) { return Request{id, prompt, max_output, arrival, ttft, tpot}; }),
           py::arg("id"), py::arg("prompt"), py::arg("max_output"), py::arg("arrival"),
           py::arg("ttft"), py::arg("tpot"))
      .def_readonly("id", &Request::id)
      .def_readonly("prompt", &Request::prompt)
      .def_readonly("max_output", &Request::max_output)
      .def_readonly("arrival", &Request::arrival)
      .def_readonly("ttft", &Request::ttft)
      .def_readonly("tpot", &Request::tpot);

  py::class_<AdmissionConfig>(m, "AdmissionConfig")
      .def(py::init<>())
      .def_readwrite("max_prefill_lanes", &AdmissionConfig::max_prefill_lanes)
      .def_readwrite("token_budget", &AdmissionConfig::token_budget)
      .def_readwrite("max_batch_requests", &AdmissionConfig::max_batch_requests)
      .def_readwrite("max_decode", &AdmissionConfig::max_decode)
      .def_readwrite("staging_experts", &AdmissionConfig::staging_experts)
      .def_readwrite("kv_capacity", &AdmissionConfig::kv_capacity)
      .def_readwrite("kv_bytes_per_token", &AdmissionConfig::kv_bytes_per_token)
      .def_readwrite("workspace_capacity", &AdmissionConfig::workspace_capacity)
      .def_readwrite("lane_workspace", &AdmissionConfig::lane_workspace)
      .def_readwrite("batch_trials", &AdmissionConfig::batch_trials)
      .def_readwrite("reject_checks", &AdmissionConfig::reject_checks);

  py::class_<PassSpec>(m, "PassSpec")
      .def_readonly("reqs", &PassSpec::reqs)
      .def_readonly("tokens", &PassSpec::tokens)
      .def_readonly("pos0", &PassSpec::pos0)
      .def_readonly("last", &PassSpec::last);

  py::class_<Decision>(m, "Decision")
      .def_readonly("publish", &Decision::publish)
      .def_readonly("feasible", &Decision::feasible)
      .def_readonly("plan", &Decision::plan)
      .def_readonly("lanes", &Decision::lanes)
      .def_readonly("decode_changed", &Decision::decode_changed)
      .def_readonly("decode", &Decision::decode)
      .def_readonly("joined", &Decision::joined)
      .def_readonly("rejected", &Decision::rejected)
      .def_readonly("pass_lanes", &Decision::pass_lanes)
      .def_readonly("passes", &Decision::passes)
      .def_readonly("searches", &Decision::searches)
      .def_readonly("micros", &Decision::micros);

  py::class_<Admission>(m, "Admission")
      .def(py::init<const Estimator*, PlanSearch*, const ProfileView*, AdmissionConfig>(),
           py::arg("estimator"), py::arg("search"), py::arg("profile"), py::arg("config"),
           py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::keep_alive<1, 4>())
      .def("arrive", &Admission::arrive, py::arg("request"), py::arg("now"))
      .def("pass_done", &Admission::pass_done, py::arg("lane"), py::arg("now"))
      .def("finished", &Admission::finished, py::arg("id"), py::arg("now"))
      .def("behind", &Admission::behind, py::arg("lane"), py::arg("now"))
      .def("progress", &Admission::progress, py::arg("lane"), py::arg("layers_done"))
      .def("generated", &Admission::generated, py::arg("id"), py::arg("tokens"))
      .def_property_readonly("queued", &Admission::queued)
      .def_property_readonly("waiting", &Admission::waiting)
      .def_property_readonly("decode_batch", &Admission::decode_batch)
      .def_property_readonly("open_lanes", &Admission::open_lanes)
      .def_property_readonly("kv_reserved", &Admission::kv_reserved)
      .def_property_readonly("plan", &Admission::plan)
      .def("lane_requests", &Admission::lane_requests);
}
