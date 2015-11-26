#ifdef USE_OCL
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

extern "C" const char _cl_relu_layer_start;
extern "C" const char _cl_relu_layer_end;

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const Dtype negative_slope = this->layer_param_.relu_param().negative_slope();

  ClState& state = Caffe::cl_state();
  state.submit_program("relu", &_cl_relu_layer_start, &_cl_relu_layer_end);
  ClKernel kernel = state.get_kernel("ReLUForward");
  cl_uint argIdx = 0;
  kernel.set_arg(argIdx++, count);
  kernel.set_arg(argIdx++, bottom_data);
  kernel.set_arg(argIdx++, top_data);
  kernel.set_arg(argIdx++, negative_slope);
  kernel.enqueue(count);
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    const Dtype negative_slope =
        this->layer_param_.relu_param().negative_slope();

    ClState& state = Caffe::cl_state();
    state.submit_program("relu", &_cl_relu_layer_start, &_cl_relu_layer_end);
    ClKernel kernel = state.get_kernel("ReLUBackward");
    cl_uint argIdx = 0;
    kernel.set_arg(argIdx++, count);
    kernel.set_arg(argIdx++, top_diff);
    kernel.set_arg(argIdx++, bottom_data);
    kernel.set_arg(argIdx++, bottom_diff);
    kernel.set_arg(argIdx++, negative_slope);
    kernel.enqueue(count);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ReLULayer);


}  // namespace caffe
#endif  // USE_OCL
