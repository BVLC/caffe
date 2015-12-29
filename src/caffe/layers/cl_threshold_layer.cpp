#ifdef USE_OCL
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/threshold_layer.hpp"

extern "C" const char _cl_threshold_layer_start;
extern "C" const char _cl_threshold_layer_end;

namespace caffe {

template <typename Dtype>
void ThresholdLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();

  cl_uint argIdx = 0;
  ClState& state = Caffe::cl_state();
  state.submit_program("threshold", &_cl_threshold_layer_start,
      &_cl_threshold_layer_end);

  ClKernel kernel = state.get_kernel("ThresholdForward");
  kernel.set_arg(argIdx++, count);
  kernel.set_arg(argIdx++, threshold_);
  kernel.set_arg(argIdx++, bottom_data);
  kernel.set_arg(argIdx++, top_data);
  kernel.enqueue(count);
}


INSTANTIATE_LAYER_GPU_FORWARD(ThresholdLayer);


}  // namespace caffe
#endif  // USE_OCL
