
#include "caffe/multi_node/param_helper.hpp"

namespace caffe {

template <>
void ParamHelper<float>::BlasCopy(const int N, const float* X, float* Y) {
  cblas_scopy(N, X, 1, Y, 1);
}

template <>
void ParamHelper<double>::BlasCopy(const int N, const double* X, double* Y) {
  cblas_dcopy(N, X, 1, Y, 1);
}

}  // end namespace caffe



