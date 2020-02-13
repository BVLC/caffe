#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/local_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LocalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1) << "Local Layer takes a single blob as input.";
  CHECK_EQ(top.size(), 1) << "Local Layer takes a single blob as output.";

  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);

  // BaseConvolutionLayer can handle more than 2D, with arbitrary shape.
  // However, LocalLayer cannot (yet). So here we make sure that only 2D
  // parameters have been given.
  CHECK_EQ(this->num_spatial_axes_, 2) <<
    "Local Layer can only be used for 2D convolution.";
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  height_out_ = (height_ + 2 * pad_data[0] - kernel_shape_data[0]) /
                stride_data[0] + 1;
  width_out_ = (width_ + 2 * pad_data[1] - kernel_shape_data[1]) /
                stride_data[1] + 1;

  M_ = this->num_output_;
  K_ = this->channels_ * kernel_shape_data[0] * kernel_shape_data[1];
  N_ = height_out_ * width_out_;

  CHECK_GT(this->num_output_, 0);
  CHECK_GE(height_, kernel_shape_data[0]) << "height smaller than kernel size";
  CHECK_GE(width_, kernel_shape_data[1]) << "width smaller than kernel size";
  // Set the parameters

  // Check if we need to set up the weights
  if (this->bias_term_) {
    this->blobs_.resize(2);
  } else {
    this->blobs_.resize(1);
  }
  // Intialize the weight
  this->blobs_[0].reset(new Blob<Dtype>(
        this->num_output_, 1, K_, N_));
  // fill the weights
  shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
  weight_filler->Fill(this->blobs_[0].get());
  // If necessary, initialize and fill the bias term
  if (this->bias_term_) {
    this->blobs_[1].reset(new Blob<Dtype>(1, 1, M_, N_));
    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
    bias_filler->Fill(this->blobs_[1].get());
  }
}

template <typename Dtype>
void LocalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
  const int first_spatial_axis = this->channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + this->num_spatial_axes_)
      << "bottom num_axes may not change.";
  this->num_ = bottom[0]->count(0, this->channel_axis_);
  CHECK_EQ(bottom[0]->shape(this->channel_axis_), this->channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
  }
  // Shape the tops.
  this->bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + this->channel_axis_);
  top_shape.push_back(this->num_output_);
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    top_shape.push_back(this->output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  this->bottom_dim_ = bottom[0]->count(this->channel_axis_);
  this->top_dim_ = top[0]->count(this->channel_axis_);
  this->out_spatial_dim_ = top[0]->count(first_spatial_axis);
  CHECK_EQ(bottom[0]->channels(), this->channels_) <<
    "Input size incompatible with weights.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(this->num_, bottom[bottom_id]->num()) <<
      "Inputs must have same num.";
    CHECK_EQ(this->channels_, bottom[bottom_id]->channels())
      << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
      << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
      << "Inputs must have same width.";
  }
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  // Shape the tops.
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(this->num_, this->num_output_,
      height_out_, width_out_);
  }

  // The im2col result buffer would only hold one image at a time to avoid
  // overly large memory usage.
  col_buffer_.Reshape(
      1, this->channels_ * kernel_shape_data[0] * kernel_shape_data[1],
      height_out_, width_out_);


  E_.Reshape(1, 1, 1, K_);
  caffe_set(E_.count(), Dtype(1), E_.mutable_cpu_data());
  intermediate_.Reshape(1, 1, K_, N_);
  intermediate_backward_.Reshape(1, 1, 1, N_);
  xt_.Reshape(1, 1, K_, N_);
}

template <typename Dtype>
void LocalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  Dtype* x_data = col_buffer_.mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();

  for (int n = 0; n < this->num_; n++) {
    im2col_cpu(bottom_data + bottom[0]->offset(n), this->channels_, height_,
        width_, kernel_shape_data[0], kernel_shape_data[1],
        pad_data[0], pad_data[1], stride_data[0], stride_data[1],
        dilation_data[0], dilation_data[1], x_data);

    for (int m = 0; m < this->num_output_; m++) {
      caffe_mul(K_*N_, x_data, weight+this->blobs_[0]->offset(m),
          intermediate_.mutable_cpu_data());

      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, N_, K_,
          (Dtype)1., E_.cpu_data(),
          intermediate_.cpu_data(),
          (Dtype)0., top_data + top[0]->offset(n, m));
    }

    if (this->bias_term_) {
      caffe_add(M_ * N_, this->blobs_[1]->cpu_data(),
          top_data + top[0]->offset(n),
          top_data + top[0]->offset(n));
    }
  }
}

template <typename Dtype>
void LocalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* x_data = col_buffer_.mutable_cpu_data();
  Dtype* x_diff = col_buffer_.mutable_cpu_diff();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* bias_diff = NULL;
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();

  Dtype* xt_data = xt_.mutable_cpu_data();

  if (this->bias_term_) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();
    caffe_set(this->blobs_[1]->count(), Dtype(0.0), bias_diff);
    for (int n = 0; n < this->num_; ++n) {
      caffe_add(M_ * N_, bias_diff,
          top_diff + top[0]->offset(n),
          bias_diff);
    }
  }

  caffe_set(this->blobs_[0]->count(), Dtype(0.0), weight_diff);
  for (int n = 0; n < this->num_; n++) {
    im2col_cpu(bottom_data + bottom[0]->offset(n), this->channels_, height_,
        width_, kernel_shape_data[0], kernel_shape_data[1],
        pad_data[0], pad_data[1], stride_data[0], stride_data[1],
        dilation_data[0], dilation_data[1], x_data);

    // gradient wrt weight
    for (int m = 0; m < this->num_output_; m++) {
      Dtype* filter_weight_diff = weight_diff+this->blobs_[0]->offset(m);
      for (int k = 0; k < K_; k++) {
        caffe_mul(N_, top_diff+top[0]->offset(n, m),
            x_data+col_buffer_.offset(0, k), xt_data+xt_.offset(0, 0, k));
      }
      caffe_cpu_axpby(K_*N_, Dtype(1.0), xt_data,
          Dtype(1.0), filter_weight_diff);
    }

    // gradient wrt bottom data
    if (propagate_down[0]) {
      caffe_set(col_buffer_.count(), Dtype(0.0), x_diff);
      for (int m = 0; m < this->num_output_; m++) {
        for (int k = 0; k < K_; k++) {
          caffe_mul(N_, top_diff+top[0]->offset(n, m),
              weight+this->blobs_[0]->offset(m, 0, k),
              intermediate_backward_.mutable_cpu_data());

          caffe_cpu_axpby(N_, Dtype(1.0),
              intermediate_backward_.cpu_data(), Dtype(1.0),
              x_diff+col_buffer_.offset(0, k));
        }
      }

      // col2im back to the data
      col2im_cpu(x_diff, this->channels_, height_,
          width_, kernel_shape_data[0], kernel_shape_data[1],
          pad_data[0], pad_data[1], stride_data[0], stride_data[1],
          dilation_data[0], dilation_data[1], bottom_diff+bottom[0]->offset(n));
    }
  }
}

template <typename Dtype>
void LocalLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

#ifdef CPU_ONLY
STUB_GPU(LocalLayer);
#endif

INSTANTIATE_CLASS(LocalLayer);
REGISTER_LAYER_CLASS(Local);

}  // namespace caffe
