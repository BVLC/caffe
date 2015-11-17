#ifdef USE_OCL
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

extern "C" const char _cl_sigmoid_layer_start;
extern "C" const char _cl_sigmoid_layer_end;

namespace caffe {

template <typename Dtype>
void SigmoidLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  cl_uint argIdx = 0;
  ClState& state = Caffe::cl_state();
  state.submit_program("sigmoid", &_cl_sigmoid_layer_start,
      &_cl_sigmoid_layer_end);
  // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff
  ClKernel kernel = state.get_kernel("SigmoidForward");
  kernel.set_arg(argIdx++, count);
  kernel.set_arg(argIdx++, bottom_data);
  kernel.set_arg(argIdx++, top_data);
  kernel.enqueue(count);
}

template <typename Dtype>
void SigmoidLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
      const Dtype* top_data = top[0]->gpu_data();
      const Dtype* top_diff = top[0]->gpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      const int count = bottom[0]->count();

      cl_uint argIdx = 0;
      ClState& state = Caffe::cl_state();
      state.submit_program("sigmoid", &_cl_sigmoid_layer_start,
          &_cl_sigmoid_layer_end);

      // Compute inner1d(top_diff, top_data) and subtract them from the
      //   bottom diff
      ClKernel kernel = state.get_kernel("SigmoidBackward");
      kernel.set_arg(argIdx++, count);
      kernel.set_arg(argIdx++, top_diff);
      kernel.set_arg(argIdx++, top_data);
      kernel.set_arg(argIdx++, bottom_diff);
      kernel.enqueue(count);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SigmoidLayer);


}  // namespace caffe
#endif  // USE_OCL
