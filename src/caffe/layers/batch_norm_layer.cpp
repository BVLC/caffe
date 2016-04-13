#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BatchNormParameter param = this->layer_param_.batch_norm_param();
  moving_average_fraction_ = param.moving_average_fraction();
  use_global_stats_ = this->phase_ == TEST;
  if (param.has_use_global_stats())
    use_global_stats_ = param.use_global_stats();
  if (bottom[0]->num_axes() == 1)
    channels_ = 1;
  else
    channels_ = bottom[0]->shape(1);
  eps_ = param.eps();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(5);

    vector<int> sz;
    sz.push_back(channels_);
    this->blobs_[0].reset(new Blob<Dtype>(sz));  // scale
    this->blobs_[1].reset(new Blob<Dtype>(sz));  // bias
    this->blobs_[2].reset(new Blob<Dtype>(sz));  // mean
    this->blobs_[3].reset(new Blob<Dtype>(sz));  // variance

    shared_ptr<Filler<Dtype> > scale_filler(
      GetFiller<Dtype>(this->layer_param_.batch_norm_param().scale_filler()));
    scale_filler->Fill(this->blobs_[0].get());
    shared_ptr<Filler<Dtype> > bias_filler(
      GetFiller<Dtype>(this->layer_param_.batch_norm_param().bias_filler()));
    bias_filler->Fill(this->blobs_[1].get());

    caffe_set(this->blobs_[2]->count(), Dtype(0.),
        this->blobs_[2]->mutable_cpu_data());
    caffe_set(this->blobs_[3]->count(), Dtype(0.),
        this->blobs_[3]->mutable_cpu_data());

    sz[0]=1;
    this->blobs_[4].reset(new Blob<Dtype>(sz));
    iter_ = 0;
  }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom[0]->num_axes() > 1)
    CHECK_EQ(bottom[0]->shape(1), channels_);
  top[0]->ReshapeLike(*bottom[0]);

  int N = bottom[0]->shape(0);
  int NC = N* channels_;
  int S = bottom[0]->count() / NC;  // S = H*W

  vector<int> sz;
  sz.push_back(channels_);
  mean_.Reshape(sz);
  variance_.Reshape(sz);
  inv_variance_.Reshape(sz);
  temp_C_.Reshape(sz);

  sz[0] = N;
  ones_N_.Reshape(sz);
  caffe_set(ones_N_.count(), Dtype(1.), ones_N_.mutable_cpu_data());
  sz[0] = channels_;
  ones_C_.Reshape(sz);
  caffe_set(ones_C_.count(), Dtype(1.), ones_C_.mutable_cpu_data());
  sz[0] = S;
  ones_HW_.Reshape(sz);
  caffe_set(ones_HW_.count(), Dtype(1.), ones_HW_.mutable_cpu_data());

  sz[0] = NC;
  temp_NC_.Reshape(sz);

  temp_.ReshapeLike(*bottom[0]);
  x_norm_.ReshapeLike(*bottom[0]);
}

//  multicast x[c] into y[.,c,...]
template <typename Dtype>
void BatchNormLayer<Dtype>::multicast_cpu(int N, int C, int S,
      const Dtype *x, Dtype *y ) {
  Blob<Dtype> temp_NC;
  vector<int> temp_size;
  temp_size.push_back(N*C);
  temp_NC.Reshape(temp_size);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N, C, 1,
      1., ones_N_.cpu_data(), x, 0., temp_NC.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N*C, S , 1,
      1., temp_NC.cpu_data(), ones_HW_.cpu_data(), 0., y);
}

//  y[c] = sum x(.,c,...)
template <typename Dtype>
void BatchNormLayer<Dtype>::compute_sum_per_channel_cpu(int N, int C, int S,
    const Dtype *x, Dtype *y ) {
  Blob<Dtype> temp_NC;
  vector<int> temp_size;
  temp_size.push_back(N*C);
  temp_NC.Reshape(temp_size);
  caffe_cpu_gemv<Dtype>(CblasNoTrans, N * C, S, 1., x, ones_HW_.cpu_data(),
      0., temp_NC.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, N, C , 1., temp_NC.cpu_data(),
      ones_N_.cpu_data(), 0., y);
}

// y[c] = mean x(.,c,...)
template <typename Dtype>
void BatchNormLayer<Dtype>::compute_mean_per_channel_cpu(int N, int C, int S,
    const Dtype *x, Dtype *y) {
  Dtype F = 1.0 / (N*S);
  compute_sum_per_channel_cpu(N, C, S, x, y);
  caffe_cpu_scale(C, F, y, y);
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int N = bottom[0]->shape(0);
  int C = channels_;
  int S = bottom[0]->count(0) / (N*C);
  int top_size = top[0]->count();

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  if (use_global_stats_) {
    // use global mean/variance
    caffe_copy(C, this->blobs_[2]->cpu_data(), mean_.mutable_cpu_data());
    caffe_copy(C, this->blobs_[3]->cpu_data(), variance_.mutable_cpu_data());
  } else {
    compute_mean_per_channel_cpu(N, C, S, bottom_data,
        mean_.mutable_cpu_data());
  }
  //  Y = X- EX
  if (bottom[0] != top[0]) {
    caffe_copy(top_size, bottom_data, top_data);
  }
  multicast_cpu(N, C, S, mean_.cpu_data(), temp_.mutable_cpu_data());
  caffe_cpu_axpby(top_size, Dtype(-1.), temp_.cpu_data(),
      Dtype(1.), top_data);
  if (!use_global_stats_) {
    // compute variance E (X-EX)^2
    caffe_powx(top_size, top_data, Dtype(2.), temp_.mutable_cpu_data());
    compute_mean_per_channel_cpu(N, C, S, temp_.cpu_data(),
        variance_.mutable_cpu_data());

    // int m = N*S;  // N*H*W
    // Dtype bias_corr = m > 1 ? Dtype(m)/(m-1) : 1;
    // bias_corr = 1.;
    // caffe_cpu_scale(C, bias_corr, variance_.cpu_data(),
    //    variance_.mutable_cpu_data());

    // clip variance
    if ((this->phase_ == TRAIN) && (iter_ <= BN_VARIANCE_CLIP_START))
      iter_++;
    if (iter_ > BN_VARIANCE_CLIP_START) {
      // clip from above
      // temp_C_[c] = average_var + gobal_var[c]
      Dtype y = caffe_cpu_asum(C, this->blobs_[3]->cpu_data());
      caffe_cpu_scale(C, Dtype(y/C), ones_C_.cpu_data(),
          temp_C_.mutable_cpu_data());
      caffe_cpu_axpby(C, Dtype(1.0), this->blobs_[3]->cpu_data(),
          Dtype(1.0), temp_C_.mutable_cpu_data());
      caffe_cpu_eltwise_min(C,
          Dtype(BN_VARIANCE_CLIP_CONST), temp_C_.cpu_data(),
          Dtype(1.0), variance_.mutable_cpu_data());
      // clip from below
      caffe_cpu_eltwise_max(C,
          Dtype((1.)/BN_VARIANCE_CLIP_CONST), this->blobs_[3]->cpu_data(),
          Dtype(1.0), variance_.mutable_cpu_data());
    }
    //  update global mean and variance
    if (iter_ > 1) {
      caffe_cpu_axpby(C,
          Dtype(1. - moving_average_fraction_), mean_.cpu_data(),
          Dtype(moving_average_fraction_), this->blobs_[2]->mutable_cpu_data());
      caffe_cpu_axpby(C,
          Dtype(1.- moving_average_fraction_), variance_.cpu_data(),
          Dtype(moving_average_fraction_), this->blobs_[3]->mutable_cpu_data());
    } else {
      caffe_copy(C, mean_.cpu_data(), this->blobs_[2]->mutable_cpu_data());
      caffe_copy(C, variance_.cpu_data(), this->blobs_[3]->mutable_cpu_data());
    }
  }
  //  inv_var= ( eps+ variance)^(-0.5)
  caffe_add_scalar(C, eps_, variance_.mutable_cpu_data());
  caffe_powx(C, variance_.cpu_data(), Dtype(-0.5),
      inv_variance_.mutable_cpu_data());
  // X_norm = (X-EX) * inv_var
  multicast_cpu(N, C, S, inv_variance_.cpu_data(), temp_.mutable_cpu_data());
  caffe_mul(top_size, top_data, temp_.cpu_data(), top_data);
  // copy x_norm for backward
  caffe_copy(top_size, top_data, x_norm_.mutable_cpu_data());

  // -- STAGE 2:  Y = X_norm * scale[c] + shift[c]  -----------------
  // Y = X_norm * scale[c]
  const Blob<Dtype> & scale_data = *(this->blobs_[0]);
  multicast_cpu(N, C, S, scale_data.cpu_data(), temp_.mutable_cpu_data());
  caffe_mul(top_size, top_data, temp_.cpu_data(), top_data);
  // Y = Y + shift[c]
  const Blob<Dtype> & shift_data = *(this->blobs_[1]);
  multicast_cpu(N, C, S, shift_data.cpu_data(), temp_.mutable_cpu_data());
  caffe_add(top_size, top_data, temp_.mutable_cpu_data(), top_data);
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  int N = bottom[0]->shape(0);
  int C = channels_;
  int S = bottom[0]->count(0) / (N*C);
  int top_size = top[0]->count();

  // --  STAGE 1: compute dE/d(scale) and dE/d(shift) ---------------

  const Dtype* top_diff = top[0]->cpu_diff();

  // scale_diff: dE/d(scale)  =  sum(dE/dY .* X_norm)
  Dtype* scale_diff = this->blobs_[0]->mutable_cpu_diff();
  caffe_mul(top_size, top_diff, x_norm_.cpu_data(), temp_.mutable_cpu_diff());
  compute_sum_per_channel_cpu(N, C, S, temp_.cpu_diff(), scale_diff);

  // shift_diff: dE/d(shift) = sum (dE/dY)
  Dtype* shift_diff = this->blobs_[1]->mutable_cpu_diff();
  compute_sum_per_channel_cpu(N, C, S, top_diff, shift_diff);

  // --  STAGE 2: backprop dE/d(x_norm) = dE/dY .* scale ------------

  // dE/d(X_norm) = dE/dY * scale[c]
  const Dtype* scale_data = this->blobs_[0]->cpu_data();
  multicast_cpu(N, C, S, scale_data, temp_.mutable_cpu_data());
  caffe_mul(top_size, top_diff, temp_.cpu_data(), x_norm_.mutable_cpu_diff());

  // --  STAGE 3: backprop dE/dY --> dE/dX --------------------------

  // ATTENTION: from now on we will use notation Y:= X_norm
  // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
  //
  // dE(Y)/dX =  (dE/dY - mean(dE/dY) - mean(dE/dY .* Y) .* Y)
  //             ./ sqrt(var(X) + eps)
  // where
  // .* and ./ are element-wise product and division,
  // mean, var, sum are computed along all dimensions except the channels.

  const Dtype* top_data = x_norm_.cpu_data();
  top_diff = x_norm_.cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  // temp = mean(dE/dY .* Y)
  caffe_mul(top_size, top_diff, top_data, temp_.mutable_cpu_diff());
  compute_mean_per_channel_cpu(N, C, S, temp_.cpu_diff(),
      temp_C_.mutable_cpu_diff());
  multicast_cpu(N, C, S, temp_C_.cpu_diff(), temp_.mutable_cpu_diff());

  // bottom = mean(dE/dY .* Y) .* Y
  caffe_mul(top_size, temp_.cpu_diff(), top_data, bottom_diff);

  // temp = mean(dE/dY)
  compute_mean_per_channel_cpu(N, C, S, top_diff, temp_C_.mutable_cpu_diff());
  multicast_cpu(N, C, S, temp_C_.cpu_diff(), temp_.mutable_cpu_diff());

  // bottom = mean(dE/dY) + mean(dE/dY .* Y) .* Y
  caffe_add(top_size, temp_.cpu_diff(), bottom_diff, bottom_diff);

  // bottom = dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
  caffe_cpu_axpby(top_size, Dtype(1.), top_diff, Dtype(-1.), bottom_diff);

  // dE/dX = dE/dX ./ sqrt(var(X) + eps)
  multicast_cpu(N, C, S, inv_variance_.cpu_data(), temp_.mutable_cpu_data());
  caffe_mul(top_size, bottom_diff, temp_.cpu_data(), bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(BatchNormLayer);
#endif

INSTANTIATE_CLASS(BatchNormLayer);

}  // namespace caffe
