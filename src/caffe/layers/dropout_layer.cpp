// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <limits>
#include <vector>

#include "caffe/layers/dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void DropoutLayer<Dtype, MItype, MOtype>::LayerSetUp(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  NeuronLayer<Dtype, MItype, MOtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ =
      static_cast<uint_tp>(static_cast<long double>
          (std::numeric_limits<uint_tp>::max())
          * static_cast<long double>(threshold_));
  this->InitializeQuantizers(bottom, top);
}

template<typename Dtype, typename MItype, typename MOtype>
void DropoutLayer<Dtype, MItype, MOtype>::Reshape(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  NeuronLayer<Dtype, MItype, MOtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  // ReshapeLike does not work because rand_vec_ is of Dtype uint
  rand_vec_.Reshape(bottom[0]->shape());

  if (Caffe::mode() == Caffe::GPU && this->device_program_.get() == nullptr) {
    this->GenerateProgram();
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void DropoutLayer<Dtype, MItype, MOtype>::Forward_cpu(
                            const vector<Blob<MItype>*>& bottom,
                            const vector<Blob<MOtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  uint_tp* mask = rand_vec_.mutable_cpu_data();
  const int_tp count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    // Create random numbers
    caffe_rng_bernoulli(count, 1. - threshold_, mask);
    for (int_tp i = 0; i < count; ++i) {
      top_data[i] = bottom_data[i] * mask[i] * scale_;
    }
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void DropoutLayer<Dtype, MItype, MOtype>::Backward_cpu(
                            const vector<Blob<MOtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<MItype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (this->phase_ == TRAIN) {
      const uint_tp* mask = rand_vec_.cpu_data();
      const int_tp count = bottom[0]->count();
      for (int_tp i = 0; i < count; ++i) {
        bottom_diff[i] = top_diff[i] * mask[i] * scale_;
      }
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(DropoutLayer);
#endif

INSTANTIATE_CLASS_3T_GUARDED(DropoutLayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(DropoutLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(DropoutLayer, (double), (double), (double));
INSTANTIATE_CLASS_3T_GUARDED(DropoutLayer, (uint8_t), (uint8_t), (uint8_t));
INSTANTIATE_CLASS_3T_GUARDED(DropoutLayer, (uint16_t), (uint16_t), (uint16_t));
INSTANTIATE_CLASS_3T_GUARDED(DropoutLayer, (uint32_t), (uint32_t), (uint32_t));
INSTANTIATE_CLASS_3T_GUARDED(DropoutLayer, (uint64_t), (uint64_t), (uint64_t));

REGISTER_LAYER_CLASS(Dropout);
REGISTER_LAYER_CLASS_INST(Dropout, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(Dropout, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(Dropout, (double), (double), (double));
REGISTER_LAYER_CLASS_INST(Dropout, (uint8_t), (uint8_t), (uint8_t));
REGISTER_LAYER_CLASS_INST(Dropout, (uint16_t), (uint16_t), (uint16_t));
REGISTER_LAYER_CLASS_INST(Dropout, (uint32_t), (uint32_t), (uint32_t));
REGISTER_LAYER_CLASS_INST(Dropout, (uint64_t), (uint64_t), (uint64_t));

}  // namespace caffe
