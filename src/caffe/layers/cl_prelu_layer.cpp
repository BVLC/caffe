#ifdef USE_OCL
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/prelu_layer.hpp"

extern "C" const char _cl_prelu_layer_start;
extern "C" const char _cl_prelu_layer_end;

namespace caffe {

template <typename Dtype>
void PReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const Dtype* slope_data = this->blobs_[0]->gpu_data();
  const int div_factor = channel_shared_ ? channels : 1;

  // For in-place computation
  if (top[0] == bottom[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
  }

  ClState& state = Caffe::cl_state();
  state.submit_program("prelu", &_cl_prelu_layer_start, &_cl_prelu_layer_end);
  ClKernel kernel = state.get_kernel("PReLUForward");
  cl_uint argIdx = 0;
  kernel.set_arg(argIdx++, count);
  kernel.set_arg(argIdx++, channels);
  kernel.set_arg(argIdx++, dim);
  kernel.set_arg(argIdx++, bottom_data);
  kernel.set_arg(argIdx++, top_data);
  kernel.set_arg(argIdx++, slope_data);
  kernel.set_arg(argIdx++, div_factor);
  kernel.enqueue(count);
}

template <typename Dtype>
void PReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  ClState& state = Caffe::cl_state();
  state.submit_program("prelu", &_cl_prelu_layer_start, &_cl_prelu_layer_end);

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.gpu_data();
  }

  // Propagate to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computation), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[0]) {
    Dtype* slope_diff = this->blobs_[0]->mutable_gpu_diff();
    // slope_diff is set as 0, then accumulated over batches
    caffe_gpu_set<Dtype>(this->blobs_[0]->count(), Dtype(0), slope_diff);
    int cdim = channels * dim;
    Dtype dsum = 0.;
    ClKernel kernel = state.get_kernel("PReLUParamBackward");
    for (int n = 0; n < bottom[0]->num(); ++n) {
      ClMemOff<Dtype> buf_top_diff = state.get_buffer_mem(
          top_diff + top[0]->offset(n));
      ClMemOff<Dtype> buf_bottom_data = state.get_buffer_mem(
          bottom_data + bottom[0]->offset(n));
      cl_uint argIdx = 0;
      kernel.set_arg(argIdx++, cdim);
      kernel.set_arg_mem(argIdx++, buf_top_diff.memobj);
      kernel.set_arg(argIdx++, static_cast<int>(buf_top_diff.offset));
      kernel.set_arg_mem(argIdx++, buf_bottom_data.memobj);
      kernel.set_arg(argIdx++, static_cast<int>(buf_bottom_data.offset));
      kernel.set_arg(argIdx++, multiplier_.mutable_gpu_diff());
      kernel.enqueue(count);

      if (channel_shared_) {
        Dtype d;
        caffe_gpu_dot<Dtype>(channels * dim, multiplier_.gpu_diff(),
            multiplier_.gpu_data(), &d);
        dsum += d;
      } else {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
            multiplier_.gpu_diff(), multiplier_.gpu_data(), 1.,
            slope_diff);
      }
    }
    if (channel_shared_) {
      caffe_gpu_set(this->blobs_[0]->count(), Dtype(dsum), slope_diff);
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* slope_data = this->blobs_[0]->gpu_data();
    int div_factor = channel_shared_ ? channels : 1;

    ClKernel kernel = state.get_kernel("PReLUBackward");
    cl_uint argIdx = 0;
    kernel.set_arg(argIdx++, count);
    kernel.set_arg(argIdx++, channels);
    kernel.set_arg(argIdx++, dim);
    kernel.set_arg(argIdx++, top_diff);
    kernel.set_arg(argIdx++, bottom_data);
    kernel.set_arg(argIdx++, bottom_diff);
    kernel.set_arg(argIdx++, slope_data);
    kernel.set_arg(argIdx++, div_factor);
    kernel.enqueue(count);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PReLULayer);


}  // namespace caffe
#endif  // USE_OCL
