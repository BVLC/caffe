#include <vector>

#include "caffe/layers/log_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LogLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  const Dtype base = this->layer_param_.log_param().base();
  if (base != Dtype(-1)) {
    CHECK_GT(base, 0) << "base must be strictly positive.";
  }
  // If base == -1, interpret the base as e and set log_base = 1 exactly.
  // Otherwise, calculate its log explicitly.
  const Dtype log_base = (base == Dtype(-1)) ? Dtype(1) : log(base);
  CHECK(!isnan(log_base))
      << "NaN result: log(base) = log(" << base << ") = " << log_base;
  CHECK(!isinf(log_base))
      << "Inf result: log(base) = log(" << base << ") = " << log_base;
  base_scale_ = Dtype(1) / log_base;
  CHECK(!isnan(base_scale_))
      << "NaN result: 1/log(base) = 1/log(" << base << ") = " << base_scale_;
  CHECK(!isinf(base_scale_))
      << "Inf result: 1/log(base) = 1/log(" << base << ") = " << base_scale_;
  input_scale_ = this->layer_param_.log_param().scale();
  input_shift_ = this->layer_param_.log_param().shift();
  backward_num_scale_ = input_scale_ / log_base;
}

template <typename Dtype>
void LogLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  if (input_scale_ == Dtype(1) && input_shift_ == Dtype(0)) {
    caffe_log(count, bottom_data, top_data);
  } else {
    caffe_copy(count, bottom_data, top_data);
    if (input_scale_ != Dtype(1)) {
      caffe_scal(count, input_scale_, top_data);
    }
    if (input_shift_ != Dtype(0)) {
      caffe_add_scalar(count, input_shift_, top_data);
    }
    caffe_log(count, top_data, top_data);
  }
  if (base_scale_ != Dtype(1)) {
    caffe_scal(count, base_scale_, top_data);
  }
}

#ifdef CPU_ONLY
STUB_GPU(LogLayer);
#endif

INSTANTIATE_CLASS(LogLayer);
REGISTER_LAYER_CLASS(Log);

}  // namespace caffe
