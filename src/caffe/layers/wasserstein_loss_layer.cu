#include <fstream>
#include <algorithm>
#include <cmath>
#include <vector>
#include "hdf5.h"
#include "hdf5_hl.h"
#include "stdint.h"
#include "caffe/filler.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/layers/wasserstein_loss_layer.hpp"

#define DISTANCE_DATASET_NAME "data"

namespace caffe {

template <typename Dtype>
void WassersteinLossLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* ypred = bottom[0]->gpu_data();
  Dtype* input_ylabel = bottom[1]->mutable_gpu_data();
  Dtype* ylabel = input_ylabel;

  const Dtype* K = K_.gpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  float lambda = this->layer_param_.wasserstein_param().lambda();

  // One-hot encoding.
  // Do this whenever number of label channels is one.
  if (bottom[1]->channels() == 1) {
    tmp_.ReshapeLike(u_);
    ylabel = tmp_.mutable_cpu_data();
    caffe_set(count, Dtype(0), ylabel);
    for (int i = 0; i < num; ++i){
      int label = static_cast<int>(input_ylabel[i]);
      ylabel[i*dim + label] = Dtype(1);
    }
  }

  v_.ReshapeLike(u_);
  Dtype* v = v_.mutable_gpu_data();
  Dtype* u = u_.mutable_gpu_data();
  float val = 1.0;
  for (int i = 0; i < count; ++i) {
      u[i] = Dtype(val);
      v[i] = Dtype(val);
  }
  
  tmp_.ReshapeLike(u_);
  Dtype* tmp = tmp_.mutable_gpu_data();

  uint32_t scaling_iter = this->layer_param_.wasserstein_param().scaling_iter();
  for (int i = 0; i < scaling_iter; i++) {
    // v = ylabel ./ K^t u
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num, dim, dim, Dtype(1.),
                   u, K, Dtype(0.), v);
    caffe_div(count, ylabel, v, v);

    // u = ypred ./ K v
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, num, dim, dim, Dtype(1.),
                   v, K, Dtype(0.), u);
    caffe_div(count, ypred, u, u);
  }

  // Loss.
  Dtype loss;
  Dtype loss_tmp;

  const Dtype* KM = KM_.gpu_data();
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num, dim, dim, Dtype(1.),
                 u, KM, Dtype(0.), tmp);
  caffe_gpu_dot(count, v, tmp, &loss);

  // (u.logu)^t K v 
  caffe_log(count, u, tmp);
  caffe_mul(count, u, tmp, tmp);

  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num, dim, dim, Dtype(1.0/lambda),
                 tmp, K, Dtype(0.), tmp);
  caffe_gpu_dot(count, tmp, v, &loss_tmp);
  loss += loss_tmp;

  // u^t K (v.logv) 
  caffe_log(count, v, tmp);
  caffe_mul(count, v, tmp, tmp);

  caffe_gpu_gemm(CblasNoTrans, CblasTrans, num, dim, dim, Dtype(1.0/lambda),
                 tmp, K, Dtype(0.), tmp);
  caffe_gpu_dot(count, tmp, u, &loss_tmp);
  loss += loss_tmp;

  // u^t (K.logK) v
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num, dim, dim, Dtype(1.0/lambda),
                 u, KlogK_.gpu_data(), Dtype(0.), tmp);
  caffe_gpu_dot(count, tmp, v, &loss_tmp);
  loss += loss_tmp;

  top[0]->mutable_gpu_data()[0] = loss / num;
}

template <typename Dtype>
void WassersteinLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    float lambda = this->layer_param_.wasserstein_param().lambda();
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / num;

    // Gradient.
    // alpha = (1/(lambda*num)) log(u)
    Dtype* alpha = bottom[0]->mutable_gpu_diff();
    caffe_log(bottom[0]->count(), u_.gpu_data(), alpha);
    caffe_scal(bottom[0]->count(), Dtype(1.0/(lambda*num)), alpha);

    if (this->layer_param_.wasserstein_param().shift_gradient()) {
      caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num, dim, dim, Dtype(-1.0/dim), 
		     alpha, one_.gpu_data(), Dtype(1.), alpha);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(WassersteinLossLayer);

}  // namespace caffe
