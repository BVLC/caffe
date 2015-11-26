#ifdef USE_OCL
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

extern "C" const char _cl_dropout_layer_start;
extern "C" const char _cl_dropout_layer_end;

namespace caffe {

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    unsigned int* mask =
        static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
    caffe_gpu_rng_uniform(count, mask);

    ClState& state = Caffe::cl_state();
    state.submit_program("dropout",
        &_cl_dropout_layer_start,
        &_cl_dropout_layer_end);

    ClKernel kernel = state.get_kernel("DropoutForward");
    cl_uint argIdx = 0;
    kernel.set_arg(argIdx++, count);
    kernel.set_arg(argIdx++, bottom_data);
    kernel.set_arg(argIdx++, mask);
    kernel.set_arg(argIdx++, uint_thres_);
    kernel.set_arg(argIdx++, scale_);
    kernel.set_arg(argIdx++, top_data);
    kernel.enqueue(count);
  } else {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    if (this->phase_ == TRAIN) {
      const unsigned int* mask =
          static_cast<const unsigned int*>(rand_vec_.gpu_data());
      const int count = bottom[0]->count();

      ClState& state = Caffe::cl_state();
      state.submit_program("dropout", &_cl_dropout_layer_start,
          &_cl_dropout_layer_end);

      ClKernel kernel = state.get_kernel("DropoutBackward");
      cl_uint argIdx = 0;
      kernel.set_arg(argIdx++, count);
      kernel.set_arg(argIdx++, top_diff);
      kernel.set_arg(argIdx++, mask);
      kernel.set_arg(argIdx++, uint_thres_);
      kernel.set_arg(argIdx++, scale_);
      kernel.set_arg(argIdx++, bottom_diff);
      kernel.enqueue(count);
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(DropoutLayer);

}  // namespace caffe
#endif  // USE_OCL
