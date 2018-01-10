#include <vector>

#include "caffe/layers/smooth_l1_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  SmoothL1LossParameter loss_param = this->layer_param_.smooth_l1_loss_param();
  sigma2_ = loss_param.sigma() * loss_param.sigma();
  has_weights_ = (bottom.size() >= 3);
  if (has_weights_) {
    CHECK_EQ(bottom.size(), 4) << "If weights are used, must specify both "
      "inside and outside weights";
  }
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  if (has_weights_) {
    CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[2]->height());
    CHECK_EQ(bottom[0]->width(), bottom[2]->width());
    CHECK_EQ(bottom[0]->channels(), bottom[3]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[3]->height());
    CHECK_EQ(bottom[0]->width(), bottom[3]->width());
  }
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  errors_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  // vector of ones used to sum
  ones_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  for (int i = 0; i < bottom[0]->count(); ++i) {
    ones_.mutable_cpu_data()[i] = Dtype(1);
  }
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
					   const vector<Blob<Dtype>*>& top) {
  // f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
  //        |x| - 0.5 / sigma / sigma    otherwise
  int count = bottom[0]->count();
  caffe_sub(
	    count,
	    bottom[0]->cpu_data(),
	    bottom[1]->cpu_data(),
	    diff_.mutable_cpu_data());
  if (has_weights_) {
    caffe_mul(
	      count,
	      bottom[2]->cpu_data(),
	      diff_.cpu_data(),
	      diff_.mutable_cpu_data());  // d := w * (b0 - b1)
  }
  const Dtype* diff_data = diff_.cpu_data();
  Dtype* error_data = errors_.mutable_cpu_data();
  for (int i = 0; i < count; ++i) {
    Dtype val = diff_data[i];
    Dtype abs_val = fabs(val);
    if (abs_val < 1.) {
      error_data[i] = 0.5 * val * val * sigma2_;
    } else {
      error_data[i] = abs_val - 0.5 / sigma2_;
    }
  }
  top[0]->mutable_cpu_data()[0] =
    caffe_cpu_asum(count, errors_.cpu_data()) / bottom[0]->num();
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = diff_.count();
  Dtype* diff_data = diff_.mutable_cpu_data();
  for (int i = 0; i < count; ++i) {
    Dtype val = diff_data[i];
    // f'(x) = x         if |x| < 1 / sigma / sigma
    //       = sign(x)   otherwise
    if (fabs(val) < 1. / sigma2_) {
      diff_data[i] = sigma2_ * val;
    } else {
      diff_data[i] = (Dtype(0) < val) - (val < Dtype(0));
    }
  }
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
		      bottom[i]->count(),               // count
		      alpha,                            // alpha
		      diff_.cpu_data(),                 // a
		      Dtype(0),                         // beta
		      bottom[i]->mutable_cpu_diff());   // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SmoothL1LossLayer);
#endif

INSTANTIATE_CLASS(SmoothL1LossLayer);
REGISTER_LAYER_CLASS(SmoothL1Loss);

}  // namespace caffe
