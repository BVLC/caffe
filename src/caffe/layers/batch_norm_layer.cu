#include <algorithm>
#include <vector>

#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// multicast x[c] into y[.,c,...]
template <typename Dtype>
void BatchNormLayer<Dtype>::multicast_gpu(int N, int C, int S,
    const Dtype *x, Dtype *y ) {
  Blob<Dtype> temp_NC;
  vector<int> temp_size;
  temp_size.push_back(N*C);
  temp_NC.Reshape(temp_size);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N, C, 1,
      1., ones_N_.gpu_data(), x, 0., temp_NC.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N*C, S , 1,
      1., temp_NC.gpu_data(), ones_HW_.gpu_data(), 0., y);
}

// y[c] = sum x(.,c,...)
template <typename Dtype>
void BatchNormLayer<Dtype>::compute_sum_per_channel_gpu(int N, int C, int S,
    const Dtype *x, Dtype *y ) {
  // assume that x.shape(1)==C
  Blob<Dtype> temp_NC;
  vector<int> temp_size;
  temp_size.push_back(N*C);
  temp_NC.Reshape(temp_size);
  caffe_gpu_gemv<Dtype>(CblasNoTrans, N * C, S, 1., x, ones_HW_.gpu_data(),
      0., temp_NC.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasTrans, N, C, 1., temp_NC.gpu_data(),
      ones_N_.gpu_data(), 0., y);
}

// y[c] = mean x(.,c,...)
template <typename Dtype>
void BatchNormLayer<Dtype>::compute_mean_per_channel_gpu(int N, int C, int S,
    const Dtype *x, Dtype *y ) {
  Dtype F = 1.0 / (N*S);
  compute_sum_per_channel_gpu(N, C, S, x, y);
  caffe_gpu_scale(C, F, y, y);
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int N = bottom[0]->shape(0);
  int C = channels_;
  int S = bottom[0]->count(0) / (N*C);
  int top_size = top[0]->count();

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  if (use_global_stats_) {
    // use global mean/variance
    caffe_copy(C, this->blobs_[2]->gpu_data(), mean_.mutable_gpu_data());
    caffe_copy(C, this->blobs_[3]->gpu_data(), variance_.mutable_gpu_data());
  } else {
    compute_mean_per_channel_gpu(N, C, S, bottom_data,
        mean_.mutable_gpu_data());
  }

  //  Y = X- EX
  if (bottom[0] != top[0]) {
    caffe_copy(top_size, bottom_data, top_data);
  }
  multicast_gpu(N, C, S, mean_.gpu_data(), temp_.mutable_gpu_data());
  caffe_gpu_axpby(top_size, Dtype(-1.), temp_.gpu_data(),
      Dtype(1.), top_data);

  if (!use_global_stats_) {
     //  compute variance E (X-EX)^2
    caffe_gpu_powx(top_size, top_data, Dtype(2.), temp_.mutable_gpu_data());
    compute_mean_per_channel_gpu(N, C, S, temp_.gpu_data(),
        variance_.mutable_gpu_data());
    // int m = N*S;    // m = N*H*W
    // Dtype bias_corr = m > 1 ? Dtype(m)/(m-1) : 1;
    // bias_corr = 1.;
    // caffe_gpu_scale(C, bias_corr, variance_.gpu_data(),
    //    variance_.mutable_gpu_data());

    // clip variance
    if ((this->phase_ == TRAIN) && (iter_ <= BN_VARIANCE_CLIP_START))
      iter_++;
    if (iter_ > BN_VARIANCE_CLIP_START) {
      // clip from above
      // temp_C_[c] = average + gobal_var[c]
      Dtype y;
      caffe_gpu_asum(C, this->blobs_[3]->gpu_data(), &y);
      caffe_gpu_scale(C, Dtype(y/C), ones_C_.gpu_data(),
          temp_C_.mutable_gpu_data());
      caffe_gpu_axpby(C, Dtype(1.0), this->blobs_[3]->gpu_data(),
          Dtype(1.0), temp_C_.mutable_gpu_data());
      caffe_gpu_eltwise_min(C,
          Dtype(BN_VARIANCE_CLIP_CONST), temp_C_.gpu_data(),
          Dtype(1.0), variance_.mutable_gpu_data());
      // clip from below
      caffe_gpu_eltwise_max(C,
          Dtype((1.)/BN_VARIANCE_CLIP_CONST), this->blobs_[3]->gpu_data(),
          Dtype(1.0), variance_.mutable_gpu_data());
    }
    //  update global mean and variance
    if (iter_ > 1) {
      caffe_gpu_axpby(C,
        Dtype(1. - moving_average_fraction_), mean_.gpu_data(),
        Dtype(moving_average_fraction_), this->blobs_[2]->mutable_gpu_data());
      caffe_gpu_axpby(C,
        Dtype((1.- moving_average_fraction_)), variance_.gpu_data(),
        Dtype(moving_average_fraction_), this->blobs_[3]->mutable_gpu_data());
    } else {
      caffe_copy(C, mean_.gpu_data(), this->blobs_[2]->mutable_gpu_data());
      caffe_copy(C, variance_.gpu_data(), this->blobs_[3]->mutable_gpu_data());
    }
  }
  //  inv_var = (eps + variance)^(-0.5)
  caffe_gpu_add_scalar(C, eps_, variance_.mutable_gpu_data());
  caffe_gpu_powx(C, variance_.gpu_data(), Dtype(-0.5),
      inv_variance_.mutable_gpu_data());

  //  X_norm = (X-EX) * inv_var
  multicast_gpu(N, C, S, inv_variance_.gpu_data(), temp_.mutable_gpu_data());
  caffe_gpu_mul(top_size, top_data, temp_.gpu_data(), top_data);
  // copy x_norm for backward
  caffe_copy(top_size, top_data, x_norm_.mutable_gpu_data());

  //  -- STAGE 2:  Y = X_norm * scale[c] + shift[c]  -----------------

  //  Y = X_norm * scale[c]
  const Blob<Dtype> & scale_data = *(this->blobs_[0]);
  multicast_gpu(N, C, S, scale_data.gpu_data(), temp_.mutable_gpu_data());
  caffe_gpu_mul(top_size, top_data, temp_.gpu_data(), top_data);

  //  Y = Y + shift[c]
  const Blob<Dtype> & shift_data = *(this->blobs_[1]);
  multicast_gpu(N, C, S, shift_data.gpu_data(), temp_.mutable_gpu_data());
  caffe_gpu_add(top_size, top_data, temp_.mutable_gpu_data(), top_data);
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  int N = bottom[0]->shape(0);
  int C = channels_;
  int S = bottom[0]->count(0) / (N*C);
  int top_size = top[0]->count();

  //  --  STAGE 1: compute dE/d(scale) and dE/d(shift) ---------------

  const Dtype* top_diff = top[0]->gpu_diff();

  //  scale_diff: dE/d(scale)  =  sum(dE/dY .* X_norm)
  Dtype* scale_diff = this->blobs_[0]->mutable_gpu_diff();
  caffe_gpu_mul(top_size, top_diff, x_norm_.gpu_data(),
      temp_.mutable_gpu_diff());
  compute_sum_per_channel_gpu(N, C, S, temp_.gpu_diff(), scale_diff);

  //  shift_diff: dE/d(shift) = sum (dE/dY)
  Dtype* shift_diff = this->blobs_[1]->mutable_gpu_diff();
  compute_sum_per_channel_gpu(N, C, S, top_diff, shift_diff);

  // --  STAGE 2: backprop dE/d(x_norm) = dE/dY .* scale ------------

  //  dE/d(X_norm) = dE/dY * scale[c]
  const Dtype* scale_data = this->blobs_[0]->gpu_data();
  multicast_gpu(N, C, S, scale_data, temp_.mutable_gpu_data());
  caffe_gpu_mul(top_size, top_diff, temp_.gpu_data(),
      x_norm_.mutable_gpu_diff());

  // --  STAGE 3: backprop dE/dY --> dE/dX --------------------------

  // ATTENTION: from now on we will use notation Y:= X_norm
  const Dtype* top_data = x_norm_.gpu_data();
  top_diff = x_norm_.gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  //  temp = mean(dE/dY .* Y)
  caffe_gpu_mul(top_size, top_diff, top_data, temp_.mutable_gpu_diff());
  compute_mean_per_channel_gpu(N, C, S, temp_.gpu_diff(),
      temp_C_.mutable_gpu_diff());
  multicast_gpu(N, C, S, temp_C_.gpu_diff(), temp_.mutable_gpu_diff());

  // bottom = mean(dE/dY .* Y) .* Y
  caffe_gpu_mul(top_size, temp_.gpu_diff(), top_data, bottom_diff);

  // temp = mean(dE/dY)
  compute_mean_per_channel_gpu(N, C, S, top_diff, temp_C_.mutable_gpu_diff());
  multicast_gpu(N, C, S, temp_C_.gpu_diff(), temp_.mutable_gpu_diff());

  // bottom = mean(dE/dY) + mean(dE/dY .* Y) .* Y
  caffe_gpu_add(top_size, temp_.gpu_diff(), bottom_diff, bottom_diff);

  // bottom = dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
  caffe_gpu_axpby(top_size, Dtype(1.), top_diff, Dtype(-1.), bottom_diff);

  // dE/dX = dE/dX ./ sqrt(var(X) + eps)
  multicast_gpu(N, C, S, inv_variance_.gpu_data(), temp_.mutable_gpu_data());
  caffe_gpu_mul(top_size, bottom_diff, temp_.gpu_data(), bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(BatchNormLayer);

template void BatchNormLayer<float>::multicast_gpu(int N, int C,
    int S, const float *x, float *y);
template void BatchNormLayer<float>::compute_sum_per_channel_gpu(int N, int C,
    int S, const float *x, float *y);
template void BatchNormLayer<float>::compute_mean_per_channel_gpu(int N, int C,
    int S, const float *x, float *y);

template void BatchNormLayer<double>::multicast_gpu(int N, int C,
    int S, const double *x, double *y);
template void BatchNormLayer<double>::compute_sum_per_channel_gpu(int N, int C,
    int S, const double *x, double *y);
template void BatchNormLayer<double>::compute_mean_per_channel_gpu(int N, int C,
    int S, const double *x, double *y);

}  // namespace caffe
