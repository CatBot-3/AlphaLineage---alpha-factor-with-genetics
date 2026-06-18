// AlphaLineage C++ expression-tree evaluator (the hot path; opt-in accelerator).
//
// Walks a flat instruction list (post-order, built by core/cpp.py:flatten) over the panel's
// stacked operand arrays, computing each op into a T x N row-major buffer. NaN / rolling /
// rank semantics replicate the pandas-based pure-Python evaluator exactly (two-pass variance,
// min_periods == window, average-tie percentile rank), so the parity test holds within a tight
// floating tolerance. Opcodes match core/cpp.py.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using Vec = std::vector<double>;
static const double NA = std::numeric_limits<double>::quiet_NaN();

enum {
  OP_LOAD = 0, OP_ADD = 1, OP_SUB = 2, OP_MUL = 3, OP_DIV = 4,
  OP_MUL_SCALAR = 5, OP_ADD_SCALAR = 6, OP_SIGNED_POWER = 7,
  OP_LOG = 8, OP_ABS = 9, OP_SIGN = 10, OP_NEG = 11,
  OP_TS_MEAN = 12, OP_TS_STD = 13, OP_TS_SUM = 14, OP_TS_MIN = 15, OP_TS_MAX = 16,
  OP_DELTA = 17, OP_DELAY = 18, OP_RANK = 19, OP_ZSCORE = 20
};

static inline double sgn(double x) {
  if (std::isnan(x)) return NA;
  return (x > 0.0) - (x < 0.0);
}

template <typename T>
using Arr = py::array_t<T, py::array::c_style | py::array::forcecast>;

static py::array_t<double> evaluate(Arr<double> fields, Arr<int32_t> ops, Arr<int32_t> a_arr,
                                    Arr<int32_t> b_arr, Arr<int32_t> ival_arr, Arr<double> fval_arr,
                                    Arr<int32_t> field_arr, int root) {
  auto fb = fields.request();
  if (fb.ndim != 3) throw std::runtime_error("fields must be (n_fields, T, N)");
  const ssize_t T = fb.shape[1], N = fb.shape[2], TN = T * N;
  const double* fdata = static_cast<double*>(fb.ptr);

  const int K = static_cast<int>(ops.shape(0));
  const int32_t* OP = ops.data();
  const int32_t* A = a_arr.data();
  const int32_t* B = b_arr.data();
  const int32_t* IV = ival_arr.data();
  const double* FV = fval_arr.data();
  const int32_t* FLD = field_arr.data();

  std::vector<Vec> res(K);

  for (int i = 0; i < K; ++i) {
    Vec out(TN);
    const int op = OP[i];
    switch (op) {
      case OP_LOAD: {
        const double* src = fdata + static_cast<ssize_t>(FLD[i]) * TN;
        std::copy(src, src + TN, out.begin());
        break;
      }
      case OP_ADD: { const Vec& x = res[A[i]]; const Vec& y = res[B[i]];
        for (ssize_t k = 0; k < TN; ++k) out[k] = x[k] + y[k]; break; }
      case OP_SUB: { const Vec& x = res[A[i]]; const Vec& y = res[B[i]];
        for (ssize_t k = 0; k < TN; ++k) out[k] = x[k] - y[k]; break; }
      case OP_MUL: { const Vec& x = res[A[i]]; const Vec& y = res[B[i]];
        for (ssize_t k = 0; k < TN; ++k) out[k] = x[k] * y[k]; break; }
      case OP_DIV: { const Vec& x = res[A[i]]; const Vec& y = res[B[i]];
        for (ssize_t k = 0; k < TN; ++k) { double q = x[k] / y[k]; out[k] = std::isfinite(q) ? q : NA; }
        break; }
      case OP_MUL_SCALAR: { const Vec& x = res[A[i]]; double s = FV[i];
        for (ssize_t k = 0; k < TN; ++k) out[k] = x[k] * s; break; }
      case OP_ADD_SCALAR: { const Vec& x = res[A[i]]; double s = FV[i];
        for (ssize_t k = 0; k < TN; ++k) out[k] = x[k] + s; break; }
      case OP_SIGNED_POWER: { const Vec& x = res[A[i]]; double s = FV[i];
        for (ssize_t k = 0; k < TN; ++k) { double v = sgn(x[k]) * std::pow(std::fabs(x[k]), s);
          out[k] = std::isfinite(v) ? v : NA; } break; }
      case OP_LOG: { const Vec& x = res[A[i]];
        for (ssize_t k = 0; k < TN; ++k) out[k] = sgn(x[k]) * std::log1p(std::fabs(x[k])); break; }
      case OP_ABS: { const Vec& x = res[A[i]];
        for (ssize_t k = 0; k < TN; ++k) out[k] = std::fabs(x[k]); break; }
      case OP_SIGN: { const Vec& x = res[A[i]];
        for (ssize_t k = 0; k < TN; ++k) out[k] = sgn(x[k]); break; }
      case OP_NEG: { const Vec& x = res[A[i]];
        for (ssize_t k = 0; k < TN; ++k) out[k] = -x[k]; break; }
      case OP_TS_SUM: case OP_TS_MEAN: case OP_TS_STD: {
        const Vec& x = res[A[i]]; int w = IV[i] < 1 ? 1 : IV[i];
        for (ssize_t s = 0; s < N; ++s) {
          for (ssize_t t = 0; t < T; ++t) {
            double r = NA;
            if (t >= w - 1) {
              double sum = 0.0; bool ok = true;
              for (int j = 0; j < w; ++j) { double v = x[(t - j) * N + s];
                if (std::isnan(v)) { ok = false; break; } sum += v; }
              if (ok) {
                if (op == OP_TS_SUM) r = sum;
                else if (op == OP_TS_MEAN) r = sum / w;
                else if (w >= 2) {  // two-pass variance, ddof=1
                  double mean = sum / w, s2 = 0.0;
                  for (int j = 0; j < w; ++j) { double d = x[(t - j) * N + s] - mean; s2 += d * d; }
                  r = std::sqrt(s2 / (w - 1));
                }
              }
            }
            out[t * N + s] = r;
          }
        }
        break;
      }
      case OP_TS_MIN: case OP_TS_MAX: {
        const Vec& x = res[A[i]]; int w = IV[i] < 1 ? 1 : IV[i];
        for (ssize_t s = 0; s < N; ++s) {
          for (ssize_t t = 0; t < T; ++t) {
            double r = NA;
            if (t >= w - 1) {
              bool ok = true;
              double m = (op == OP_TS_MIN) ? std::numeric_limits<double>::infinity()
                                           : -std::numeric_limits<double>::infinity();
              for (int j = 0; j < w; ++j) { double v = x[(t - j) * N + s];
                if (std::isnan(v)) { ok = false; break; }
                m = (op == OP_TS_MIN) ? std::min(m, v) : std::max(m, v); }
              if (ok) r = m;
            }
            out[t * N + s] = r;
          }
        }
        break;
      }
      case OP_DELTA: { const Vec& x = res[A[i]]; ssize_t w = IV[i];
        for (ssize_t t = 0; t < T; ++t) for (ssize_t s = 0; s < N; ++s)
          out[t * N + s] = (t >= w) ? x[t * N + s] - x[(t - w) * N + s] : NA; break; }
      case OP_DELAY: { const Vec& x = res[A[i]]; ssize_t w = IV[i];
        for (ssize_t t = 0; t < T; ++t) for (ssize_t s = 0; s < N; ++s)
          out[t * N + s] = (t >= w) ? x[(t - w) * N + s] : NA; break; }
      case OP_RANK: {  // cross-sectional percentile rank, average ties
        const Vec& x = res[A[i]];
        for (ssize_t t = 0; t < T; ++t) {
          for (ssize_t s = 0; s < N; ++s) {
            double v = x[t * N + s];
            if (std::isnan(v)) { out[t * N + s] = NA; continue; }
            int less = 0, eq = 0, cnt = 0;
            for (ssize_t s2 = 0; s2 < N; ++s2) { double u = x[t * N + s2];
              if (std::isnan(u)) continue; ++cnt; if (u < v) ++less; else if (u == v) ++eq; }
            out[t * N + s] = (less + (eq + 1) / 2.0) / cnt;
          }
        }
        break;
      }
      case OP_ZSCORE: {  // cross-sectional z-score, ddof=1
        const Vec& x = res[A[i]];
        for (ssize_t t = 0; t < T; ++t) {
          double sum = 0.0; int cnt = 0;
          for (ssize_t s = 0; s < N; ++s) { double v = x[t * N + s];
            if (std::isnan(v)) continue; ++cnt; sum += v; }
          double mean = cnt ? sum / cnt : NA, sd = NA;
          if (cnt >= 2) {
            double s2 = 0.0;
            for (ssize_t s = 0; s < N; ++s) { double v = x[t * N + s];
              if (std::isnan(v)) continue; double d = v - mean; s2 += d * d; }
            sd = std::sqrt(s2 / (cnt - 1));
          }
          for (ssize_t s = 0; s < N; ++s) { double v = x[t * N + s];
            if (std::isnan(v)) { out[t * N + s] = NA; continue; }
            double q = (v - mean) / sd; out[t * N + s] = std::isfinite(q) ? q : NA; }
        }
        break;
      }
      default:
        throw std::runtime_error("unknown opcode");
    }
    res[i] = std::move(out);
  }

  py::array_t<double> result({T, N});
  std::copy(res[root].begin(), res[root].end(), static_cast<double*>(result.request().ptr));
  return result;
}

PYBIND11_MODULE(_evaluator, m) {
  m.doc() = "AlphaLineage C++ expression-tree evaluator";
  m.def("evaluate", &evaluate, "Evaluate a flattened expression tree over stacked panel arrays.");
}
