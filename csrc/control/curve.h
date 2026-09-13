// Calibrated curves of the control plane: piecewise-linear through measured
// points.
#pragma once
#include <algorithm>
#include <vector>

namespace monoplan {

// y(x) through calibrated points with x increasing. Beyond the last point
// the curve stays constant. Before the first point it stays constant, or,
// for rates that vanish at zero (a rate against SMs or credits), falls
// linearly to the origin.
struct Curve {
  std::vector<double> x, y;
  bool origin = false;

  bool empty() const { return x.empty(); }

  double at(double v) const {
    if (x.empty()) return 0.0;
    if (v <= x.front()) return origin && x.front() > 0 ? y.front() * std::max(v, 0.0) / x.front() : y.front();
    if (v >= x.back()) return y.back();
    const size_t i = static_cast<size_t>(std::upper_bound(x.begin(), x.end(), v) - x.begin());
    const double t = (v - x[i - 1]) / (x[i] - x[i - 1]);
    return y[i - 1] + t * (y[i] - y[i - 1]);
  }
};

}  // namespace monoplan
