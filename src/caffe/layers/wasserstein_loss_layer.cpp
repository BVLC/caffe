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
void WassersteinLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  u_.ReshapeLike(*bottom[0]);
  Dtype* u = u_.mutable_cpu_data();
  int dim = bottom[0]->count() / bottom[0]->num();
  float val = 1.0;
  for (int i = 0; i < bottom[0]->count(); ++i) {
      u[i] = Dtype(val);
  }

  CHECK(this->layer_param_.wasserstein_param().has_ground_metric())
      << "Ground metric must be specified.";
  string filename = this->layer_param_.wasserstein_param().ground_metric();
  hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    LOG(FATAL) << "Failed opening HDF5 file: " << filename;
  } else {
    const int MIN_DATA_DIM = 1;
    const int MAX_DATA_DIM = INT_MAX;
    hdf5_load_nd_dataset(file_id, DISTANCE_DATASET_NAME,
			 MIN_DATA_DIM, MAX_DATA_DIM, &groundm_);
    herr_t status = H5Fclose(file_id);
    CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;
  }
  
  float lambda = this->layer_param_.wasserstein_param().lambda();

  K_.ReshapeLike(groundm_);
  caffe_copy(K_.count(), groundm_.cpu_data(), K_.mutable_cpu_data());
  caffe_scal(groundm_.count(), Dtype(-lambda), K_.mutable_cpu_data());
  caffe_add_scalar(groundm_.count(), Dtype(-1.0), K_.mutable_cpu_data());
  caffe_exp(K_.count(), K_.cpu_data(), K_.mutable_cpu_data());

  KM_.ReshapeLike(groundm_);
  caffe_mul(K_.count(), K_.cpu_data(), groundm_.cpu_data(), KM_.mutable_cpu_data());

  KlogK_.ReshapeLike(K_);
  caffe_log(dim*dim, K_.cpu_data(), KlogK_.mutable_cpu_data());
  caffe_mul(dim*dim, K_.cpu_data(), KlogK_.cpu_data(), KlogK_.mutable_cpu_data());

  if (this->layer_param_.wasserstein_param().shift_gradient()) {
    one_.Reshape(dim, dim, 1, 1);
    Dtype* one = one_.mutable_cpu_data();
    for (int i = 0; i < dim * dim; ++i) {
      one[i] = 1.0;
    }
  }
}

template <typename Dtype>
void WassersteinLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  // replace with CHECK_EQ(bottom[1]->channels(), 1) OR CHECK_EQ(bottom[1]->channels(), bottom[0]->channels())
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  int dim = bottom[0]->count()/bottom[0]->num();
  CHECK_EQ(groundm_.num(), dim) << "Wrong dimensions of distance matrix";
  CHECK_EQ(groundm_.count(), dim * dim) << "Wrong dimensions of distance matrix";    
}


template <typename Dtype>
void WassersteinLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* ypred = bottom[0]->cpu_data();
  Dtype* input_ylabel = bottom[1]->mutable_cpu_data();
  Dtype* ylabel = input_ylabel;

  const Dtype* K = K_.cpu_data();
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
  Dtype* v = v_.mutable_cpu_data();
  Dtype* u = u_.mutable_cpu_data();
  float val = 1.0;
  for (int i = 0; i < count; ++i) {
      u[i] = Dtype(val);
      v[i] = Dtype(val);
  }
  
  tmp_.ReshapeLike(u_);
  Dtype* tmp = tmp_.mutable_cpu_data();

  uint32_t scaling_iter = this->layer_param_.wasserstein_param().scaling_iter();
  for (int i = 0; i < scaling_iter; i++) {
    // v = ylabel ./ K^t u
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, dim, dim, Dtype(1.),
                   u, K, Dtype(0.), v);
    caffe_div(count, ylabel, v, v);

    // u = ypred ./ K v
    caffe_cpu_gemm(CblasNoTrans, CblasTrans, num, dim, dim, Dtype(1.),
                   v, K, Dtype(0.), u);
    caffe_div(count, ypred, u, u);
  }

  // Loss.
  const Dtype* KM = KM_.cpu_data();
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, dim, dim, Dtype(1.),
                 u, KM, Dtype(0.), tmp);
  Dtype loss = caffe_cpu_dot(count, v, tmp);

  // Entropy term.
  // (u.logu)^t K v 
  caffe_log(count, u, tmp);
  caffe_mul(count, u, tmp, tmp);

  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, dim, dim, Dtype(1.0/lambda),
                 tmp, K, Dtype(0.), tmp);
  loss += caffe_cpu_dot(count, tmp, v);

  // u^t K (v.logv) 
  caffe_log(count, v, tmp);
  caffe_mul(count, v, tmp, tmp);

  caffe_cpu_gemm(CblasNoTrans, CblasTrans, num, dim, dim, Dtype(1.0/lambda),
                 tmp, K, Dtype(0.), tmp);
  loss += caffe_cpu_dot(count, tmp, u);

  // u^t (K.logK) v
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, dim, dim, Dtype(1.0/lambda),
                 u, KlogK_.cpu_data(), Dtype(0.), tmp);
  loss += caffe_cpu_dot(count, tmp, v);

  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void WassersteinLossLayer<Dtype>::Backward_cpu(
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
    Dtype* alpha = bottom[0]->mutable_cpu_diff();
    caffe_log(bottom[0]->count(), u_.cpu_data(), alpha);
    caffe_scal(bottom[0]->count(), Dtype(1.0/(lambda*num)), alpha);

    if (this->layer_param_.wasserstein_param().shift_gradient()) {
      caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, dim, dim, Dtype(-1.0/dim), 
		     alpha, one_.cpu_data(), Dtype(1.), alpha);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(WassersteinLossLayer);
#endif

INSTANTIATE_CLASS(WassersteinLossLayer);
REGISTER_LAYER_CLASS(WassersteinLoss);

}  // namespace caffe
