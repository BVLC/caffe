#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/local_layer.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {

template <typename Dtype>
void LocalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1) << "Conv Layer takes a single blob as input.";
  CHECK_EQ(top.size(), 1) << "Conv Layer takes a single blob as output.";

  kernel_size_ = this->layer_param_.local_param().kernel_size();
  stride_ = this->layer_param_.local_param().stride();
  pad_ = this->layer_param_.local_param().pad();
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  num_output_ = this->layer_param_.local_param().num_output();

  height_out_ = (height_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  width_out_ = (width_ + 2 * pad_ - kernel_size_) / stride_ + 1;

  M_ = num_output_;
  K_ = channels_ * kernel_size_ * kernel_size_;
  N_ = height_out_ * width_out_;

  CHECK_GT(num_output_, 0);
  CHECK_GE(height_, kernel_size_) << "height smaller than kernel size";
  CHECK_GE(width_, kernel_size_) << "width smaller than kernel size";
  // Set the parameters
  bias_term_ = this->layer_param_.local_param().bias_term();

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    this->blobs_[0].reset(new Blob<Dtype>(
          num_output_, 1, K_, N_));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
          this->layer_param_.local_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, M_, N_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
            this->layer_param_.local_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
}

template <typename Dtype>
void LocalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
    " weights.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
      << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
      << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
      << "Inputs must have same width.";
  }

  // Shape the tops.
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  }

  // The im2col result buffer would only hold one image at a time to avoid
  // overly large memory usage.
  col_buffer_.Reshape(
      1, channels_ * kernel_size_ * kernel_size_, height_out_, width_out_);

  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  }
}

template <typename Dtype>
void LocalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  Dtype* x_data = col_buffer_.mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  Blob<Dtype> E;
  E.Reshape(1, 1, 1, K_);
  FillerParameter filler_param;
  filler_param.set_value(1);
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(&E);

  Blob<Dtype> intermediate;
  intermediate.Reshape(1, 1, K_, N_);
  for (int n = 0; n < num_; n++) {
    im2col_cpu(bottom_data + bottom[0]->offset(n), channels_, height_,
        width_, kernel_size_, kernel_size_,
        pad_, pad_, stride_, stride_,
        dilation_, dilation_, x_data);

    for (int m = 0; m < num_output_; m++) {
      caffe_mul(K_*N_, x_data, weight+this->blobs_[0]->offset(m),
          intermediate.mutable_cpu_data());

      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, N_, K_,
          (Dtype)1., E.cpu_data(),
          intermediate.cpu_data(),
          (Dtype)0., top_data + top[0]->offset(n, m));
    }

    if (bias_term_) {
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

  Blob<Dtype> intermediate;
  intermediate.Reshape(1, 1, 1, N_);

  Blob<Dtype> xt;
  xt.Reshape(1, 1, K_, N_);
  Dtype* xt_data = xt.mutable_cpu_data();

  if (bias_term_) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();
    caffe_set(this->blobs_[1]->count(), Dtype(0.0), bias_diff);
    for (int n = 0; n < num_; ++n) {
      caffe_add(M_ * N_, bias_diff,
          top_diff + top[0]->offset(n),
          bias_diff);
    }
  }

  caffe_set(this->blobs_[0]->count(), Dtype(0.0), weight_diff);
  for (int n = 0; n < num_; n++) {
    im2col_cpu(bottom_data + bottom[0]->offset(n), channels_, height_,
        width_, kernel_size_, kernel_size_,
        pad_, pad_, stride_, stride_,
        dilation_, dilation_, x_data);

    // gradient wrt weight
    for (int m = 0; m < num_output_; m++) {
      Dtype* filter_weight_diff = weight_diff+this->blobs_[0]->offset(m);
      for (int k = 0; k < K_; k++) {
        caffe_mul(N_, top_diff+top[0]->offset(n, m),
            x_data+col_buffer_.offset(0, k), xt_data+xt.offset(0, 0, k));
      }
      caffe_cpu_axpby(K_*N_, Dtype(1.0), xt_data,
          Dtype(1.0), filter_weight_diff);
    }

    // gradient wrt bottom data
    if (propagate_down[0]) {
      caffe_set(col_buffer_.count(), Dtype(0.0), x_diff);
      for (int m = 0; m < num_output_; m++) {
        for (int k = 0; k < K_; k++) {
          caffe_mul(N_, top_diff+top[0]->offset(n, m),
              weight+this->blobs_[0]->offset(m, 0, k),
              intermediate.mutable_cpu_data());

          caffe_cpu_axpby(N_, Dtype(1.0),
              intermediate.cpu_data(), Dtype(1.0),
              x_diff+col_buffer_.offset(0, k));
        }
      }

      // col2im back to the data
      col2im_cpu(x_diff, channels_, height_,
          width_, kernel_size_, kernel_size_,
          pad_, pad_, stride_, stride_,
          dilation_, dilation_, bottom_diff + bottom[0]->offset(n));
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(LocalLayer);
#endif

INSTANTIATE_CLASS(LocalLayer);
REGISTER_LAYER_CLASS(Local);

}  // namespace caffe
