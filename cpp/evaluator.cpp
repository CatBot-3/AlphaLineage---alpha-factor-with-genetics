// AlphaLineage native expression evaluator.
//
// The Python side compiles macro-expanded trees into a compact post-order IR. This module
// evaluates that IR over a contiguous (field, date, symbol) panel.  The implementation keeps
// pandas-compatible NaN/tie/ddof semantics, uses linear/sliding rolling kernels, reuses dead
// intermediate buffers, and exposes an ordered multi-program entry point.  All expensive work
// runs with the Python GIL released.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace py = pybind11;

using Vec = std::vector<double>;
static const double NA = std::numeric_limits<double>::quiet_NaN();
static constexpr int ABI_VERSION = 8;
static constexpr int MAX_NATIVE_WORKERS = 32;
static constexpr ssize_t SCORE_COLUMNS = 12;

enum Opcode : int32_t {
  OP_LOAD = 0,
  OP_ADD = 1,
  OP_SUB = 2,
  OP_MUL = 3,
  OP_DIV = 4,
  OP_MUL_SCALAR = 5,
  OP_ADD_SCALAR = 6,
  OP_SIGNED_POWER = 7,
  OP_LOG = 8,
  OP_ABS = 9,
  OP_SIGN = 10,
  OP_NEG = 11,
  OP_TS_MEAN = 12,
  OP_TS_STD = 13,
  OP_TS_SUM = 14,
  OP_TS_MIN = 15,
  OP_TS_MAX = 16,
  OP_DELTA = 17,
  OP_DELAY = 18,
  OP_RANK = 19,
  OP_ZSCORE = 20,
  OP_TS_EMA = 21,
  OP_TS_RANK = 22,
  OP_DECAY_LINEAR = 23,
  OP_TS_CORR = 24,
  OP_TS_COV = 25,
  OP_SCALE = 26,
  OP_GT = 27,
  OP_LT = 28,
  OP_GE = 29,
  OP_LE = 30,
  OP_AND = 31,
  OP_OR = 32,
  OP_NOT = 33,
  OP_WHERE = 34,
  OP_TS_RMA = 35,
  OP_TS_RECURSIVE_SMOOTH = 36,
  OP_TS_STD_POP = 37,
  OP_TS_CUMSUM = 38,
};

template <typename T>
using Arr = py::array_t<T, py::array::c_style | py::array::forcecast>;

struct Plan {
  std::vector<int32_t> op;
  std::vector<int32_t> a;
  std::vector<int32_t> b;
  std::vector<int32_t> ival;
  std::vector<double> fval;
  std::vector<int32_t> field;
  int root = -1;
};

static inline double signum(double x) {
  if (std::isnan(x)) return NA;
  return static_cast<double>((x > 0.0) - (x < 0.0));
}

static inline bool rolling_observation(double x) {
  // pandas' rolling/ewm kernels treat +/- infinity as missing observations.
  return std::isfinite(x);
}

static inline bool truthy(double x) { return x != 0.0; }

static bool uses_b(int op) {
  return (op >= OP_ADD && op <= OP_DIV) || op == OP_TS_CORR || op == OP_TS_COV ||
         (op >= OP_GT && op <= OP_LE) || op == OP_AND || op == OP_OR || op == OP_WHERE;
}

static bool is_window_op(int op) {
  return (op >= OP_TS_MEAN && op <= OP_DELAY) || (op >= OP_TS_EMA && op <= OP_TS_COV) ||
         op == OP_TS_RMA || op == OP_TS_RECURSIVE_SMOOTH || op == OP_TS_STD_POP;
}

template <typename T>
static std::vector<T> copy_1d(const Arr<T>& array, const char* name) {
  auto info = array.request();
  if (info.ndim != 1) throw std::runtime_error(std::string(name) + " must be one-dimensional");
  const auto* begin = static_cast<const T*>(info.ptr);
  return std::vector<T>(begin, begin + info.shape[0]);
}

static Plan make_plan(const Arr<int32_t>& ops, const Arr<int32_t>& a_arr,
                      const Arr<int32_t>& b_arr, const Arr<int32_t>& ival_arr,
                      const Arr<double>& fval_arr, const Arr<int32_t>& field_arr, int root,
                      ssize_t n_fields) {
  Plan p;
  p.op = copy_1d(ops, "ops");
  p.a = copy_1d(a_arr, "a");
  p.b = copy_1d(b_arr, "b");
  p.ival = copy_1d(ival_arr, "ival");
  p.fval = copy_1d(fval_arr, "fval");
  p.field = copy_1d(field_arr, "field");
  const size_t k = p.op.size();
  if (k == 0) throw std::runtime_error("a plan must contain at least one instruction");
  if (p.a.size() != k || p.b.size() != k || p.ival.size() != k || p.fval.size() != k ||
      p.field.size() != k) {
    throw std::runtime_error("all instruction arrays must have the same length");
  }
  if (root < 0 || static_cast<size_t>(root) >= k) throw std::runtime_error("invalid root index");
  p.root = root;

  auto validate_dep = [](int dep, size_t i, const char* name) {
    if (dep < 0 || static_cast<size_t>(dep) >= i) {
      throw std::runtime_error(std::string("invalid ") + name + " dependency");
    }
  };
  for (size_t i = 0; i < k; ++i) {
    const int op = p.op[i];
    if (op < OP_LOAD || op > OP_TS_CUMSUM) throw std::runtime_error("unknown opcode");
    if (op == OP_LOAD) {
      if (p.field[i] < 0 || p.field[i] >= n_fields) {
        throw std::runtime_error("field index out of range");
      }
      continue;
    }
    validate_dep(p.a[i], i, "a");
    if (uses_b(op)) validate_dep(p.b[i], i, "b");
    if (op == OP_WHERE) validate_dep(p.ival[i], i, "where-false");
    if (is_window_op(op) && p.ival[i] <= 0) throw std::runtime_error("window must be positive");
    if ((op == OP_MUL_SCALAR || op == OP_ADD_SCALAR || op == OP_SIGNED_POWER ||
         op == OP_TS_RECURSIVE_SMOOTH) &&
        !std::isfinite(p.fval[i])) {
      throw std::runtime_error("scalar must be finite");
    }
  }
  return p;
}

static Plan tuple_plan(const py::handle& value, ssize_t n_fields) {
  py::tuple item = py::cast<py::tuple>(value);
  if (item.size() != 7) throw std::runtime_error("each plan must be a seven-item tuple");
  return make_plan(Arr<int32_t>(item[0]), Arr<int32_t>(item[1]), Arr<int32_t>(item[2]),
                   Arr<int32_t>(item[3]), Arr<double>(item[4]), Arr<int32_t>(item[5]),
                   py::cast<int>(item[6]), n_fields);
}

// These are the compensated algorithms used by pandas' rolling mean/variance kernels. Matching
// their update order matters for expressions that amplify a correlation very close to +/-1.
struct PandasMean {
  int n = 0;
  int negative = 0;
  int consecutive_same = 0;
  double sum = 0.0;
  double compensation_add = 0.0;
  double compensation_remove = 0.0;
  double previous = NA;

  void add(double value) {
    if (!rolling_observation(value)) return;
    ++n;
    const double adjusted = value - compensation_add;
    const double next = sum + adjusted;
    compensation_add = next - sum - adjusted;
    sum = next;
    if (value < 0.0) ++negative;
    consecutive_same = value == previous ? consecutive_same + 1 : 1;
    previous = value;
  }

  void remove(double value) {
    if (!rolling_observation(value)) return;
    --n;
    const double adjusted = -value - compensation_remove;
    const double next = sum + adjusted;
    compensation_remove = next - sum - adjusted;
    sum = next;
    if (value < 0.0) --negative;
  }

  double mean(int min_periods) const {
    if (n < min_periods || n <= 0) return NA;
    double result = consecutive_same >= n ? previous : sum / n;
    if (negative == 0 && result < 0.0) result = 0.0;
    if (negative == n && result > 0.0) result = 0.0;
    return result;
  }

  double total(int min_periods) const {
    if (n < min_periods) return NA;
    return consecutive_same >= n ? previous * n : sum;
  }
};

struct PandasVariance {
  int n = 0;
  int consecutive_same = 0;
  double mean = 0.0;
  double squared_deviations = 0.0;
  double compensation_add = 0.0;
  double compensation_remove = 0.0;
  double previous = NA;

  void add(double value) {
    if (!rolling_observation(value)) return;
    consecutive_same = value == previous ? consecutive_same + 1 : 1;
    previous = value;
    ++n;
    const double previous_mean = mean - compensation_add;
    const double adjusted = value - compensation_add;
    const double centered = adjusted - mean;
    compensation_add = centered + mean - adjusted;
    mean += centered / n;
    squared_deviations += (value - previous_mean) * (value - mean);
  }

  void remove(double value) {
    if (!rolling_observation(value)) return;
    --n;
    if (n > 0) {
      const double previous_mean = mean - compensation_remove;
      const double adjusted = value - compensation_remove;
      const double centered = adjusted - mean;
      compensation_remove = centered + mean - adjusted;
      mean -= centered / n;
      squared_deviations -= (value - previous_mean) * (value - mean);
    } else {
      mean = 0.0;
      squared_deviations = 0.0;
    }
  }

  double variance(int min_periods, int ddof = 1) const {
    if (n < min_periods || n <= ddof) return NA;
    if (n == 1 || consecutive_same >= n) return 0.0;
    return squared_deviations / (n - ddof);
  }
};

class Fenwick {
 public:
  explicit Fenwick(size_t size) : bit_(size + 1, 0) {}

  void add(size_t index, int delta) {
    for (size_t i = index; i < bit_.size(); i += i & (~i + 1)) bit_[i] += delta;
  }

  int prefix(size_t index) const {
    int result = 0;
    for (size_t i = index; i > 0; i -= i & (~i + 1)) result += bit_[i];
    return result;
  }

 private:
  std::vector<int> bit_;
};

static void rolling_sum_mean_std(const Vec& x, Vec& out, ssize_t t_count, ssize_t n_symbols,
                                 int window, int op) {
  std::fill(out.begin(), out.end(), NA);
  for (ssize_t s = 0; s < n_symbols; ++s) {
    PandasMean mean;
    PandasVariance variance;
    for (ssize_t t = 0; t < t_count; ++t) {
      // pandas' monotonic rolling bounds remove values before adding the new endpoint.
      if (t >= window) {
        const double old = x[(t - window) * n_symbols + s];
        if (op == OP_TS_STD || op == OP_TS_STD_POP)
          variance.remove(old);
        else
          mean.remove(old);
      }
      const double current = x[t * n_symbols + s];
      if (op == OP_TS_STD || op == OP_TS_STD_POP)
        variance.add(current);
      else
        mean.add(current);
      if (t < window - 1) continue;
      if (op == OP_TS_SUM) {
        out[t * n_symbols + s] = mean.total(window);
      } else if (op == OP_TS_MEAN) {
        out[t * n_symbols + s] = mean.mean(window);
      } else {
        const double value = variance.variance(window, op == OP_TS_STD_POP ? 0 : 1);
        // pandas zsqrt clamps tiny negative roundoff to zero.
        if (!std::isnan(value)) out[t * n_symbols + s] = std::sqrt(std::max(0.0, value));
      }
    }
  }
}

static void rolling_min_max(const Vec& x, Vec& out, ssize_t t_count, ssize_t n_symbols,
                            int window, bool take_min) {
  std::fill(out.begin(), out.end(), NA);
  for (ssize_t s = 0; s < n_symbols; ++s) {
    std::deque<std::pair<ssize_t, double>> candidates;
    int invalid = 0;
    for (ssize_t t = 0; t < t_count; ++t) {
      const double current = x[t * n_symbols + s];
      if (rolling_observation(current)) {
        while (!candidates.empty() &&
               (take_min ? candidates.back().second >= current
                         : candidates.back().second <= current)) {
          candidates.pop_back();
        }
        candidates.emplace_back(t, current);
      } else {
        ++invalid;
      }
      if (t >= window) {
        const double old = x[(t - window) * n_symbols + s];
        if (!rolling_observation(old)) --invalid;
      }
      const ssize_t expired = t - window;
      while (!candidates.empty() && candidates.front().first <= expired) candidates.pop_front();
      if (t >= window - 1 && invalid == 0 && !candidates.empty()) {
        out[t * n_symbols + s] = candidates.front().second;
      }
    }
  }
}

static void rolling_ema(const Vec& x, Vec& out, ssize_t t_count, ssize_t n_symbols, int window) {
  std::fill(out.begin(), out.end(), NA);
  const double alpha = 2.0 / (static_cast<double>(window) + 1.0);
  const double old_factor = 1.0 - alpha;
  for (ssize_t s = 0; s < n_symbols; ++s) {
    bool initialized = false;
    double weighted = NA;
    double old_weight = 1.0;
    int observations = 0;
    for (ssize_t t = 0; t < t_count; ++t) {
      const double current = x[t * n_symbols + s];
      const bool observed = rolling_observation(current);
      if (observed) ++observations;
      if (!initialized) {
        if (observed) {
          initialized = true;
          weighted = current;
        }
      } else {
        // pandas EWM defaults: adjust=False, ignore_na=False, normalize=True.
        old_weight *= old_factor;
        if (observed) {
          if (weighted != current) {
            weighted = (old_weight * weighted + alpha * current) / (old_weight + alpha);
          }
          old_weight = 1.0;
        }
      }
      if (observations >= window) out[t * n_symbols + s] = weighted;
    }
  }
}

static void rolling_rma(const Vec& x, Vec& out, ssize_t t_count, ssize_t n_symbols, int window) {
  std::fill(out.begin(), out.end(), NA);
  for (ssize_t s = 0; s < n_symbols; ++s) {
    double state = NA;
    double seed_sum = 0.0;
    int seed_count = 0;
    for (ssize_t t = 0; t < t_count; ++t) {
      const double current = x[t * n_symbols + s];
      if (!rolling_observation(current)) {
        state = NA;
        seed_sum = 0.0;
        seed_count = 0;
        continue;
      }
      if (std::isnan(state)) {
        seed_sum += current;
        ++seed_count;
        if (seed_count < window) continue;
        state = seed_sum / window;
        seed_sum = 0.0;
        seed_count = 0;
      } else {
        state = (state * (window - 1) + current) / window;
      }
      out[t * n_symbols + s] = state;
    }
  }
}

static void rolling_recursive_smooth(const Vec& x, Vec& out, ssize_t t_count,
                                     ssize_t n_symbols, int window, double initial) {
  std::fill(out.begin(), out.end(), NA);
  for (ssize_t s = 0; s < n_symbols; ++s) {
    double state = initial;
    for (ssize_t t = 0; t < t_count; ++t) {
      const double current = x[t * n_symbols + s];
      if (!rolling_observation(current)) continue;
      state = (state * (window - 1) + current) / window;
      out[t * n_symbols + s] = state;
    }
  }
}

static void rolling_rank(const Vec& x, Vec& out, ssize_t t_count, ssize_t n_symbols, int window) {
  std::fill(out.begin(), out.end(), NA);
  for (ssize_t s = 0; s < n_symbols; ++s) {
    std::vector<double> values;
    values.reserve(t_count);
    for (ssize_t t = 0; t < t_count; ++t) {
      const double value = x[t * n_symbols + s];
      if (rolling_observation(value)) values.push_back(value);
    }
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    Fenwick counts(values.size());
    int invalid = 0;
    auto coordinate = [&values](double value) {
      return static_cast<size_t>(std::lower_bound(values.begin(), values.end(), value) -
                                 values.begin()) +
             1;
    };
    for (ssize_t t = 0; t < t_count; ++t) {
      const double current = x[t * n_symbols + s];
      if (rolling_observation(current))
        counts.add(coordinate(current), 1);
      else
        ++invalid;
      if (t >= window) {
        const double old = x[(t - window) * n_symbols + s];
        if (rolling_observation(old))
          counts.add(coordinate(old), -1);
        else
          --invalid;
      }
      if (t >= window - 1 && invalid == 0) {
        const size_t upper = static_cast<size_t>(
            std::upper_bound(values.begin(), values.end(), current) - values.begin());
        out[t * n_symbols + s] = static_cast<double>(counts.prefix(upper)) / window;
      }
    }
  }
}

static void rolling_decay(const Vec& x, Vec& out, ssize_t t_count, ssize_t n_symbols, int window) {
  std::fill(out.begin(), out.end(), NA);
  const long double denominator = static_cast<long double>(window) * (window + 1) / 2.0L;
  for (ssize_t s = 0; s < n_symbols; ++s) {
    long double sum = 0.0L, weighted = 0.0L;
    int invalid = 0;
    for (ssize_t t = 0; t < t_count; ++t) {
      const double raw = x[t * n_symbols + s];
      const bool observed = rolling_observation(raw);
      const long double current = observed ? raw : 0.0L;
      if (!observed) ++invalid;
      if (t < window) {
        sum += current;
        weighted += static_cast<long double>(t + 1) * current;
      } else {
        const double old_raw = x[(t - window) * n_symbols + s];
        const long double old = rolling_observation(old_raw) ? old_raw : 0.0L;
        weighted = weighted - sum + static_cast<long double>(window) * current;
        sum += current - old;
        if (!rolling_observation(old_raw)) --invalid;
      }
      if (t >= window - 1 && invalid == 0) {
        out[t * n_symbols + s] = static_cast<double>(weighted / denominator);
      }
    }
  }
}

static void rolling_pair(const Vec& x, const Vec& y, Vec& out, ssize_t t_count,
                         ssize_t n_symbols, int window, bool correlation) {
  std::fill(out.begin(), out.end(), NA);
  if (window < 2) return;  // pandas cov/corr use ddof=1.
  const ssize_t frame_size = t_count * n_symbols;
  Vec masked_x(frame_size), masked_y(frame_size), product(frame_size);
  Vec mean_product(frame_size), mean_x(frame_size), mean_y(frame_size);
  for (ssize_t q = 0; q < frame_size; ++q) {
    if (rolling_observation(x[q]) && rolling_observation(y[q])) {
      masked_x[q] = x[q];
      masked_y[q] = y[q];
      product[q] = x[q] * y[q];
    } else {
      masked_x[q] = masked_y[q] = product[q] = NA;
    }
  }
  rolling_sum_mean_std(product, mean_product, t_count, n_symbols, window, OP_TS_MEAN);
  rolling_sum_mean_std(masked_x, mean_x, t_count, n_symbols, window, OP_TS_MEAN);
  rolling_sum_mean_std(masked_y, mean_y, t_count, n_symbols, window, OP_TS_MEAN);

  Vec std_y;
  if (correlation) {
    // The DSL primitive computes pair-masked covariance, then multiplies the two independently
    // rolled std frames (rather than pandas Rolling.corr's pair-masked variances).
    rolling_sum_mean_std(x, product, t_count, n_symbols, window, OP_TS_STD);
    std_y.resize(frame_size);
    rolling_sum_mean_std(y, std_y, t_count, n_symbols, window, OP_TS_STD);
  }
  const double ddof_adjustment = static_cast<double>(window) / (window - 1);
  for (ssize_t q = 0; q < frame_size; ++q) {
    const double covariance =
        (mean_product[q] - mean_x[q] * mean_y[q]) * ddof_adjustment;
    const double result = correlation ? covariance / (product[q] * std_y[q]) : covariance;
    if (std::isfinite(result)) out[q] = result;
  }
}

static void cross_rank(const Vec& x, Vec& out, ssize_t t_count, ssize_t n_symbols) {
  std::fill(out.begin(), out.end(), NA);
  std::vector<std::pair<double, ssize_t>> row;
  row.reserve(n_symbols);
  for (ssize_t t = 0; t < t_count; ++t) {
    row.clear();
    for (ssize_t s = 0; s < n_symbols; ++s) {
      const double value = x[t * n_symbols + s];
      if (!std::isnan(value)) row.emplace_back(value, s);
    }
    std::sort(row.begin(), row.end(), [](const auto& left, const auto& right) {
      return left.first < right.first;
    });
    size_t first = 0;
    while (first < row.size()) {
      size_t after = first + 1;
      while (after < row.size() && row[after].first == row[first].first) ++after;
      const double percentile = (static_cast<double>(first + 1) + static_cast<double>(after)) /
                                (2.0 * static_cast<double>(row.size()));
      for (size_t j = first; j < after; ++j) {
        out[t * n_symbols + row[j].second] = percentile;
      }
      first = after;
    }
  }
}

static void compute_plan(const double* fields, ssize_t n_fields, ssize_t t_count,
                         ssize_t n_symbols, const Plan& plan, double* destination) {
  const ssize_t frame_size = t_count * n_symbols;
  const size_t k_count = plan.op.size();

  std::vector<int> remaining(k_count, 0);
  for (size_t i = 0; i < k_count; ++i) {
    const int op = plan.op[i];
    if (op == OP_LOAD) continue;
    ++remaining[plan.a[i]];
    if (uses_b(op)) ++remaining[plan.b[i]];
    if (op == OP_WHERE) ++remaining[plan.ival[i]];
  }

  std::vector<Vec> slots;
  std::vector<int> node_slot(k_count, -1);
  std::vector<int> free_slots;
  slots.reserve(k_count);

  auto acquire_slot = [&]() {
    if (!free_slots.empty()) {
      const int slot = free_slots.back();
      free_slots.pop_back();
      return slot;
    }
    slots.emplace_back(static_cast<size_t>(frame_size));
    return static_cast<int>(slots.size() - 1);
  };
  auto consume = [&](int dependency) {
    if (--remaining[dependency] == 0 && dependency != plan.root) {
      free_slots.push_back(node_slot[dependency]);
      node_slot[dependency] = -1;
    }
  };

  for (size_t i = 0; i < k_count; ++i) {
    const int slot = acquire_slot();
    node_slot[i] = slot;
    Vec& out = slots[slot];
    const int op = plan.op[i];

    if (op == OP_LOAD) {
      const double* source = fields + static_cast<ssize_t>(plan.field[i]) * frame_size;
      std::copy(source, source + frame_size, out.begin());
    } else {
      const Vec& x = slots[node_slot[plan.a[i]]];
      const Vec* y = uses_b(op) ? &slots[node_slot[plan.b[i]]] : nullptr;
      switch (op) {
        case OP_ADD:
          for (ssize_t q = 0; q < frame_size; ++q) out[q] = x[q] + (*y)[q];
          break;
        case OP_SUB:
          for (ssize_t q = 0; q < frame_size; ++q) out[q] = x[q] - (*y)[q];
          break;
        case OP_MUL:
          for (ssize_t q = 0; q < frame_size; ++q) out[q] = x[q] * (*y)[q];
          break;
        case OP_DIV:
          for (ssize_t q = 0; q < frame_size; ++q) {
            const double value = x[q] / (*y)[q];
            out[q] = std::isfinite(value) ? value : NA;
          }
          break;
        case OP_MUL_SCALAR:
          for (ssize_t q = 0; q < frame_size; ++q) out[q] = x[q] * plan.fval[i];
          break;
        case OP_ADD_SCALAR:
          for (ssize_t q = 0; q < frame_size; ++q) out[q] = x[q] + plan.fval[i];
          break;
        case OP_SIGNED_POWER:
          for (ssize_t q = 0; q < frame_size; ++q) {
            const double value = signum(x[q]) * std::pow(std::fabs(x[q]), plan.fval[i]);
            out[q] = std::isfinite(value) ? value : NA;
          }
          break;
        case OP_LOG:
          for (ssize_t q = 0; q < frame_size; ++q)
            out[q] = signum(x[q]) * std::log1p(std::fabs(x[q]));
          break;
        case OP_ABS:
          for (ssize_t q = 0; q < frame_size; ++q) out[q] = std::fabs(x[q]);
          break;
        case OP_SIGN:
          for (ssize_t q = 0; q < frame_size; ++q) out[q] = signum(x[q]);
          break;
        case OP_NEG:
          for (ssize_t q = 0; q < frame_size; ++q) out[q] = -x[q];
          break;
        case OP_TS_MEAN:
        case OP_TS_STD:
        case OP_TS_STD_POP:
        case OP_TS_SUM:
          rolling_sum_mean_std(x, out, t_count, n_symbols, plan.ival[i], op);
          break;
        case OP_TS_MIN:
        case OP_TS_MAX:
          rolling_min_max(x, out, t_count, n_symbols, plan.ival[i], op == OP_TS_MIN);
          break;
        case OP_TS_CUMSUM:
          std::fill(out.begin(), out.end(), NA);
          for (ssize_t s = 0; s < n_symbols; ++s) {
            double total = 0.0;
            for (ssize_t t = 0; t < t_count; ++t) {
              const double current = x[t * n_symbols + s];
              if (!rolling_observation(current)) {
                total = 0.0;
                continue;
              }
              total += current;
              out[t * n_symbols + s] = total;
            }
          }
          break;
        case OP_DELTA: {
          const ssize_t window = plan.ival[i];
          std::fill(out.begin(), out.end(), NA);
          for (ssize_t t = window; t < t_count; ++t)
            for (ssize_t s = 0; s < n_symbols; ++s)
              out[t * n_symbols + s] = x[t * n_symbols + s] - x[(t - window) * n_symbols + s];
          break;
        }
        case OP_DELAY: {
          const ssize_t window = plan.ival[i];
          std::fill(out.begin(), out.end(), NA);
          for (ssize_t t = window; t < t_count; ++t)
            for (ssize_t s = 0; s < n_symbols; ++s)
              out[t * n_symbols + s] = x[(t - window) * n_symbols + s];
          break;
        }
        case OP_RANK:
          cross_rank(x, out, t_count, n_symbols);
          break;
        case OP_ZSCORE:
          for (ssize_t t = 0; t < t_count; ++t) {
            long double sum = 0.0L;
            int count = 0;
            for (ssize_t s = 0; s < n_symbols; ++s) {
              const double value = x[t * n_symbols + s];
              if (std::isnan(value)) continue;
              sum += value;
              ++count;
            }
            const double mean = count ? static_cast<double>(sum / count) : NA;
            long double squared = 0.0L;
            if (count >= 2) {
              for (ssize_t s = 0; s < n_symbols; ++s) {
                const double value = x[t * n_symbols + s];
                if (std::isnan(value)) continue;
                const long double delta = static_cast<long double>(value) - mean;
                squared += delta * delta;
              }
            }
            const double sd = count >= 2 ? std::sqrt(static_cast<double>(squared / (count - 1))) : NA;
            for (ssize_t s = 0; s < n_symbols; ++s) {
              const double value = x[t * n_symbols + s];
              const double result = (value - mean) / sd;
              out[t * n_symbols + s] = std::isfinite(result) ? result : NA;
            }
          }
          break;
        case OP_TS_EMA:
          rolling_ema(x, out, t_count, n_symbols, plan.ival[i]);
          break;
        case OP_TS_RMA:
          rolling_rma(x, out, t_count, n_symbols, plan.ival[i]);
          break;
        case OP_TS_RECURSIVE_SMOOTH:
          rolling_recursive_smooth(x, out, t_count, n_symbols, plan.ival[i], plan.fval[i]);
          break;
        case OP_TS_RANK:
          rolling_rank(x, out, t_count, n_symbols, plan.ival[i]);
          break;
        case OP_DECAY_LINEAR:
          rolling_decay(x, out, t_count, n_symbols, plan.ival[i]);
          break;
        case OP_TS_CORR:
        case OP_TS_COV:
          rolling_pair(x, *y, out, t_count, n_symbols, plan.ival[i], op == OP_TS_CORR);
          break;
        case OP_SCALE:
          for (ssize_t t = 0; t < t_count; ++t) {
            long double denominator = 0.0L;
            for (ssize_t s = 0; s < n_symbols; ++s) {
              const double value = x[t * n_symbols + s];
              if (!std::isnan(value)) denominator += std::fabs(value);
            }
            const double denom = denominator == 0.0L ? NA : static_cast<double>(denominator);
            for (ssize_t s = 0; s < n_symbols; ++s) {
              const double result = x[t * n_symbols + s] / denom;
              out[t * n_symbols + s] = std::isfinite(result) ? result : NA;
            }
          }
          break;
        case OP_GT:
        case OP_LT:
        case OP_GE:
        case OP_LE:
          for (ssize_t q = 0; q < frame_size; ++q) {
            bool result = false;
            if (!std::isnan(x[q]) && !std::isnan((*y)[q])) {
              if (op == OP_GT) result = x[q] > (*y)[q];
              if (op == OP_LT) result = x[q] < (*y)[q];
              if (op == OP_GE) result = x[q] >= (*y)[q];
              if (op == OP_LE) result = x[q] <= (*y)[q];
            }
            out[q] = result ? 1.0 : 0.0;
          }
          break;
        case OP_AND:
          for (ssize_t q = 0; q < frame_size; ++q)
            out[q] = truthy(x[q]) && truthy((*y)[q]) ? 1.0 : 0.0;
          break;
        case OP_OR:
          for (ssize_t q = 0; q < frame_size; ++q)
            out[q] = truthy(x[q]) || truthy((*y)[q]) ? 1.0 : 0.0;
          break;
        case OP_NOT:
          for (ssize_t q = 0; q < frame_size; ++q) out[q] = truthy(x[q]) ? 0.0 : 1.0;
          break;
        case OP_WHERE: {
          const Vec& when_false = slots[node_slot[plan.ival[i]]];
          for (ssize_t q = 0; q < frame_size; ++q)
            out[q] = truthy(x[q]) ? (*y)[q] : when_false[q];
          break;
        }
        default:
          throw std::runtime_error("unknown opcode");
      }
    }

    if (op != OP_LOAD) {
      consume(plan.a[i]);
      if (uses_b(op)) consume(plan.b[i]);
      if (op == OP_WHERE) consume(plan.ival[i]);
    }
  }

  const int root_slot = node_slot[plan.root];
  if (root_slot < 0) throw std::runtime_error("root buffer was released unexpectedly");
  std::copy(slots[root_slot].begin(), slots[root_slot].end(), destination);
}

static py::array_t<double> evaluate(Arr<double> fields, Arr<int32_t> ops, Arr<int32_t> a_arr,
                                    Arr<int32_t> b_arr, Arr<int32_t> ival_arr,
                                    Arr<double> fval_arr, Arr<int32_t> field_arr, int root) {
  const auto field_info = fields.request();
  if (field_info.ndim != 3) throw std::runtime_error("fields must be (n_fields, T, N)");
  const ssize_t n_fields = field_info.shape[0], t_count = field_info.shape[1];
  const ssize_t n_symbols = field_info.shape[2];
  const double* field_data = static_cast<const double*>(field_info.ptr);
  Plan plan = make_plan(ops, a_arr, b_arr, ival_arr, fval_arr, field_arr, root, n_fields);
  py::array_t<double> result(std::vector<ssize_t>{t_count, n_symbols});
  double* destination = result.mutable_data();
  {
    py::gil_scoped_release release;
    compute_plan(field_data, n_fields, t_count, n_symbols, plan, destination);
  }
  return result;
}

static py::array_t<double> evaluate_many(Arr<double> fields, py::iterable plan_values,
                                         int workers) {
  const auto field_info = fields.request();
  if (field_info.ndim != 3) throw std::runtime_error("fields must be (n_fields, T, N)");
  const ssize_t n_fields = field_info.shape[0], t_count = field_info.shape[1];
  const ssize_t n_symbols = field_info.shape[2], frame_size = t_count * n_symbols;
  const double* field_data = static_cast<const double*>(field_info.ptr);

  std::vector<Plan> plans;
  for (const py::handle item : plan_values) plans.push_back(tuple_plan(item, n_fields));
  py::array_t<double> result(
      std::vector<ssize_t>{static_cast<ssize_t>(plans.size()), t_count, n_symbols});
  double* destination = result.mutable_data();
  if (plans.empty()) return result;

  const size_t thread_count = std::min<size_t>(
      plans.size(), static_cast<size_t>(std::clamp(workers, 1, MAX_NATIVE_WORKERS)));
  std::exception_ptr failure;
  std::mutex failure_mutex;
  std::atomic<size_t> next{0};
  std::fenv_t caller_floating_environment;
  std::fegetenv(&caller_floating_environment);
  {
    py::gil_scoped_release release;
    auto run = [&]() {
      try {
        // Python/NumPy may configure the main thread's x87/MXCSR precision state. New Windows
        // threads otherwise start with different defaults, changing last-bit results that can be
        // amplified by signed powers. Copy the caller state so worker count cannot change values.
        std::fesetenv(&caller_floating_environment);
        while (true) {
          const size_t index = next.fetch_add(1, std::memory_order_relaxed);
          if (index >= plans.size()) return;
          compute_plan(field_data, n_fields, t_count, n_symbols, plans[index],
                       destination + static_cast<ssize_t>(index) * frame_size);
        }
      } catch (...) {
        std::lock_guard<std::mutex> lock(failure_mutex);
        if (!failure) failure = std::current_exception();
        next.store(plans.size(), std::memory_order_relaxed);
      }
    };
    if (thread_count == 1) {
      run();
    } else {
      std::vector<std::thread> threads;
      threads.reserve(thread_count);
      for (size_t i = 0; i < thread_count; ++i) threads.emplace_back(run);
      for (auto& thread : threads) thread.join();
    }
  }
  if (failure) std::rethrow_exception(failure);
  return result;
}

static void average_ranks(const std::vector<double>& values, std::vector<double>& ranks) {
  std::vector<std::pair<double, size_t>> order;
  order.reserve(values.size());
  for (size_t i = 0; i < values.size(); ++i) order.emplace_back(values[i], i);
  std::sort(order.begin(), order.end(), [](const auto& left, const auto& right) {
    return left.first < right.first;
  });
  ranks.resize(values.size());
  size_t first = 0;
  while (first < order.size()) {
    size_t after = first + 1;
    while (after < order.size() && order[after].first == order[first].first) ++after;
    const double rank = (static_cast<double>(first + 1) + static_cast<double>(after)) / 2.0;
    for (size_t i = first; i < after; ++i) ranks[order[i].second] = rank;
    first = after;
  }
}

static double row_correlation(const std::vector<double>& left,
                              const std::vector<double>& right) {
  const size_t count = left.size();
  if (count < 2 || right.size() != count) return NA;
  double sum_left = 0.0, sum_right = 0.0, sum_product = 0.0;
  double sum_left_sq = 0.0, sum_right_sq = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const double x = left[i], y = right[i];
    sum_left += x;
    sum_right += y;
    sum_product += x * y;
    sum_left_sq += x * x;
    sum_right_sq += y * y;
  }
  const double n = static_cast<double>(count);
  double numerator = sum_product - sum_left * sum_right / n;
  double ss_left = sum_left_sq - sum_left * sum_left / n;
  double ss_right = sum_right_sq - sum_right * sum_right / n;
  const double tolerance = std::numeric_limits<double>::epsilon() * 16.0;
  const bool unstable =
      !std::isfinite(sum_left + sum_right + sum_product + sum_left_sq + sum_right_sq) ||
      ss_left <= tolerance * std::max(sum_left_sq, 1.0) ||
      ss_right <= tolerance * std::max(sum_right_sq, 1.0);
  if (unstable) {
    const double mean_left = sum_left / n, mean_right = sum_right / n;
    numerator = 0.0;
    ss_left = 0.0;
    ss_right = 0.0;
    bool product_observed = false;
    for (size_t i = 0; i < count; ++i) {
      const double x = left[i] - mean_left, y = right[i] - mean_right;
      const double product = x * y;
      if (!std::isnan(product)) {
        numerator += product;
        product_observed = true;
      }
      if (!std::isnan(x * x)) ss_left += x * x;
      if (!std::isnan(y * y)) ss_right += y * y;
    }
    if (!product_observed) return NA;
  }
  const double denominator = std::sqrt(ss_left * ss_right);
  const double result = numerator / (denominator == 0.0 ? NA : denominator);
  return std::isfinite(result) ? result : NA;
}

static void score_plan(const double* fields, ssize_t n_fields, ssize_t t_count,
                       ssize_t n_symbols, const Plan& plan, const double* forward,
                       int tree_size, int method, bool absolute, double parsimony, int min_names,
                       int min_valid_dates, double* destination) {
  const ssize_t frame_size = t_count * n_symbols;
  Vec factor(frame_size);
  compute_plan(fields, n_fields, t_count, n_symbols, plan, factor.data());

  std::vector<double> factor_values, target_values, factor_ranks, target_ranks, daily;
  std::vector<int> active_names;
  factor_values.reserve(n_symbols);
  target_values.reserve(n_symbols);
  factor_ranks.reserve(n_symbols);
  target_ranks.reserve(n_symbols);
  daily.reserve(t_count);
  active_names.reserve(t_count);
  const int required_names = std::max(2, min_names);
  for (ssize_t t = 0; t < t_count; ++t) {
    factor_values.clear();
    target_values.clear();
    for (ssize_t s = 0; s < n_symbols; ++s) {
      const double x = factor[t * n_symbols + s], y = forward[t * n_symbols + s];
      // Python daily_ic pairs on NaN only; +/- infinity remains a ranked observation.
      if (!std::isnan(x) && !std::isnan(y)) {
        factor_values.push_back(x);
        target_values.push_back(y);
      }
    }
    if (static_cast<int>(factor_values.size()) < required_names) continue;
    const std::vector<double>* x_values = &factor_values;
    const std::vector<double>* y_values = &target_values;
    if (method == 0) {
      average_ranks(factor_values, factor_ranks);
      average_ranks(target_values, target_ranks);
      x_values = &factor_ranks;
      y_values = &target_ranks;
    }
    const double correlation = row_correlation(*x_values, *y_values);
    if (!std::isnan(correlation)) {
      daily.push_back(correlation);
      active_names.push_back(static_cast<int>(factor_values.size()));
    }
  }

  double raw = 0.0, information_ratio = 0.0, signed_mean = 0.0, oriented_mean = 0.0;
  double mean_absolute = 0.0, polarity = 1.0, oriented_information_ratio = 0.0;
  double sign_consistency = 0.0, average_active_names = 0.0, minimum_active_names = 0.0;
  if (static_cast<int>(daily.size()) >= min_valid_dates) {
    double signed_sum = 0.0, absolute_sum = 0.0;
    for (const double value : daily) {
      signed_sum += value;
      absolute_sum += std::fabs(value);
    }
    const double count = static_cast<double>(daily.size());
    signed_mean = signed_sum / count;
    oriented_mean = std::fabs(signed_mean);
    mean_absolute = absolute_sum / count;
    polarity = signed_mean < 0.0 ? -1.0 : 1.0;
    raw = absolute ? oriented_mean : signed_mean;
    if (daily.size() >= 2) {
      double squared = 0.0;
      for (const double value : daily) {
        const double delta = value - signed_mean;
        squared += delta * delta;
      }
      const double standard_deviation = std::sqrt(squared / (count - 1.0));
      if (standard_deviation != 0.0 && std::isfinite(standard_deviation)) {
        information_ratio = signed_mean / standard_deviation;
      }
    }
    oriented_information_ratio = polarity * information_ratio;
    size_t consistent = 0;
    int minimum_names_observed = std::numeric_limits<int>::max();
    double names_sum = 0.0;
    for (size_t index = 0; index < daily.size(); ++index) {
      if (daily[index] * polarity > 0.0) ++consistent;
      names_sum += active_names[index];
      minimum_names_observed = std::min(minimum_names_observed, active_names[index]);
    }
    sign_consistency = static_cast<double>(consistent) / count;
    average_active_names = names_sum / count;
    minimum_active_names = static_cast<double>(minimum_names_observed);
    if (!std::isfinite(raw)) raw = 0.0;
  }
  destination[0] = raw - parsimony * tree_size;
  destination[1] = raw;
  destination[2] = information_ratio;
  destination[3] = signed_mean;
  destination[4] = oriented_mean;
  destination[5] = mean_absolute;
  destination[6] = polarity;
  destination[7] = oriented_information_ratio;
  destination[8] = sign_consistency;
  destination[9] = static_cast<double>(daily.size());
  destination[10] = average_active_names;
  destination[11] = minimum_active_names;
}

static py::array_t<double> score_many(Arr<double> fields, py::iterable plan_values,
                                      Arr<double> forward_returns, Arr<int32_t> tree_sizes,
                                      int method, bool absolute, double parsimony, int min_names,
                                      int min_valid_dates, int workers) {
  const auto field_info = fields.request();
  if (field_info.ndim != 3) throw std::runtime_error("fields must be (n_fields, T, N)");
  const ssize_t n_fields = field_info.shape[0], t_count = field_info.shape[1];
  const ssize_t n_symbols = field_info.shape[2];
  const double* field_data = static_cast<const double*>(field_info.ptr);
  const auto forward_info = forward_returns.request();
  if (forward_info.ndim != 2 || forward_info.shape[0] != t_count ||
      forward_info.shape[1] != n_symbols) {
    throw std::runtime_error("forward_returns must have shape (T, N)");
  }
  const double* forward_data = static_cast<const double*>(forward_info.ptr);
  if (method != 0 && method != 1) throw std::runtime_error("unknown scoring method");
  if (!std::isfinite(parsimony)) throw std::runtime_error("parsimony must be finite");
  if (min_names < 1 || min_valid_dates < 1) {
    throw std::runtime_error("minimum counts must be positive");
  }

  std::vector<Plan> plans;
  for (const py::handle item : plan_values) plans.push_back(tuple_plan(item, n_fields));
  const std::vector<int32_t> sizes = copy_1d(tree_sizes, "tree_sizes");
  if (sizes.size() != plans.size()) throw std::runtime_error("tree_sizes length mismatch");
  py::array_t<double> result(
      std::vector<ssize_t>{static_cast<ssize_t>(plans.size()), SCORE_COLUMNS});
  double* destination = result.mutable_data();
  if (plans.empty()) return result;

  const size_t thread_count = std::min<size_t>(
      plans.size(), static_cast<size_t>(std::clamp(workers, 1, MAX_NATIVE_WORKERS)));
  std::exception_ptr failure;
  std::mutex failure_mutex;
  std::atomic<size_t> next{0};
  std::fenv_t caller_floating_environment;
  std::fegetenv(&caller_floating_environment);
  {
    py::gil_scoped_release release;
    auto run = [&]() {
      try {
        std::fesetenv(&caller_floating_environment);
        while (true) {
          const size_t index = next.fetch_add(1, std::memory_order_relaxed);
          if (index >= plans.size()) return;
          score_plan(field_data, n_fields, t_count, n_symbols, plans[index], forward_data,
                     sizes[index], method, absolute, parsimony, min_names, min_valid_dates,
                     destination + index * SCORE_COLUMNS);
        }
      } catch (...) {
        std::lock_guard<std::mutex> lock(failure_mutex);
        if (!failure) failure = std::current_exception();
        next.store(plans.size(), std::memory_order_relaxed);
      }
    };
    if (thread_count == 1) {
      run();
    } else {
      std::vector<std::thread> threads;
      threads.reserve(thread_count);
      for (size_t i = 0; i < thread_count; ++i) threads.emplace_back(run);
      for (auto& thread : threads) thread.join();
    }
  }
  if (failure) std::rethrow_exception(failure);
  return result;
}

PYBIND11_MODULE(_evaluator, module) {
  module.doc() = "AlphaLineage deterministic native expression evaluator";
  module.attr("ABI_VERSION") = ABI_VERSION;
  module.attr("MAX_WORKERS") = MAX_NATIVE_WORKERS;
  module.def("evaluate", &evaluate, py::arg("fields"), py::arg("ops"), py::arg("a"),
             py::arg("b"), py::arg("ival"), py::arg("fval"), py::arg("field"),
             py::arg("root"), "Evaluate one flattened expression tree.");
  module.def("evaluate_many", &evaluate_many, py::arg("fields"), py::arg("plans"),
             py::arg("workers") = 1,
             "Evaluate flattened trees in input order using bounded native workers.");
  module.def("score_many", &score_many, py::arg("fields"), py::arg("plans"),
             py::arg("forward_returns"), py::arg("tree_sizes"), py::arg("method"),
             py::arg("absolute"), py::arg("parsimony"), py::arg("min_names"),
             py::arg("min_valid_dates"), py::arg("workers") = 1,
             "Evaluate and IC-score flattened trees in input order without materializing frames.");
}
