#ifdef USE_OCL
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/pooling_layer.hpp"

extern "C" const char _cl_pooling_layer_start;
extern "C" const char _cl_pooling_layer_end;

namespace caffe {

template <typename Dtype>
void PoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  int num = bottom[0]->num();
  cl_uint argIdx = 0;
  ClState& state = Caffe::cl_state();
  state.submit_program("pooling", &_cl_pooling_layer_start,
      &_cl_pooling_layer_end);

  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    {
      // We'll output the mask to top[1] if it's of size >1.
      const bool use_top_mask = top.size() > 1;
      int* mask = NULL;
      Dtype* top_mask = NULL;
      if (use_top_mask) {
        top_mask = top[1]->mutable_gpu_data();
      } else {
        mask = max_idx_.mutable_gpu_data();
      }

      ClKernel kernel = state.get_kernel("MaxPoolForward");
      kernel.set_arg(argIdx++, count);
      kernel.set_arg(argIdx++, bottom_data);
      kernel.set_arg(argIdx++, num);
      kernel.set_arg(argIdx++, channels_);
      kernel.set_arg(argIdx++, height_);
      kernel.set_arg(argIdx++, width_);
      kernel.set_arg(argIdx++, pooled_height_);
      kernel.set_arg(argIdx++, pooled_width_);
      kernel.set_arg(argIdx++, kernel_h_);
      kernel.set_arg(argIdx++, kernel_w_);
      kernel.set_arg(argIdx++, stride_h_);
      kernel.set_arg(argIdx++, stride_w_);
      kernel.set_arg(argIdx++, pad_h_);
      kernel.set_arg(argIdx++, pad_w_);
      kernel.set_arg(argIdx++, top_data);
      kernel.set_arg(argIdx++, mask);
      kernel.set_arg(argIdx++, top_mask);
      kernel.enqueue(count);
      break;
    }
  case PoolingParameter_PoolMethod_AVE:
    {
      ClKernel kernel = state.get_kernel("AvePoolForward");
      kernel.set_arg(argIdx++, count);
      kernel.set_arg(argIdx++, bottom_data);
      kernel.set_arg(argIdx++, num);
      kernel.set_arg(argIdx++, channels_);
      kernel.set_arg(argIdx++, height_);
      kernel.set_arg(argIdx++, width_);
      kernel.set_arg(argIdx++, pooled_height_);
      kernel.set_arg(argIdx++, pooled_width_);
      kernel.set_arg(argIdx++, kernel_h_);
      kernel.set_arg(argIdx++, kernel_w_);
      kernel.set_arg(argIdx++, stride_h_);
      kernel.set_arg(argIdx++, stride_w_);
      kernel.set_arg(argIdx++, pad_h_);
      kernel.set_arg(argIdx++, pad_w_);
      kernel.set_arg(argIdx++, top_data);
      kernel.enqueue(count);
      break;
    }
  case PoolingParameter_PoolMethod_STOCHASTIC:
    {
      if (this->phase_ == TRAIN) {
        // We need to create the random index as well.
        caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1),
                              rand_idx_.mutable_gpu_data());

        ClKernel kernel = state.get_kernel("StoPoolForwardTrain");
        kernel.set_arg(argIdx++, count);
        kernel.set_arg(argIdx++, bottom_data);
        kernel.set_arg(argIdx++, num);
        kernel.set_arg(argIdx++, channels_);
        kernel.set_arg(argIdx++, height_);
        kernel.set_arg(argIdx++, width_);
        kernel.set_arg(argIdx++, pooled_height_);
        kernel.set_arg(argIdx++, pooled_width_);
        kernel.set_arg(argIdx++, kernel_h_);
        kernel.set_arg(argIdx++, kernel_w_);
        kernel.set_arg(argIdx++, stride_h_);
        kernel.set_arg(argIdx++, stride_w_);
        kernel.set_arg(argIdx++, rand_idx_.mutable_gpu_data());
        kernel.set_arg(argIdx++, top_data);
        kernel.enqueue(count);
      } else {
        ClKernel kernel = state.get_kernel("StoPoolForwardTest");
        kernel.set_arg(argIdx++, count);
        kernel.set_arg(argIdx++, bottom_data);
        kernel.set_arg(argIdx++, num);
        kernel.set_arg(argIdx++, channels_);
        kernel.set_arg(argIdx++, height_);
        kernel.set_arg(argIdx++, width_);
        kernel.set_arg(argIdx++, pooled_height_);
        kernel.set_arg(argIdx++, pooled_width_);
        kernel.set_arg(argIdx++, kernel_h_);
        kernel.set_arg(argIdx++, kernel_w_);
        kernel.set_arg(argIdx++, stride_h_);
        kernel.set_arg(argIdx++, stride_w_);
        kernel.set_arg(argIdx++, top_data);
        kernel.enqueue(count);
      }
      break;
    }
  default:
    {
      LOG(FATAL) << "Unknown pooling method.";
    }
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  int num = top[0]->num();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  cl_uint argIdx = 0;
  ClState& state = Caffe::cl_state();
  state.submit_program("pooling", &_cl_pooling_layer_start,
      &_cl_pooling_layer_end);

  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    {
      // We'll output the mask to top[1] if it's of size >1.
      const bool use_top_mask = top.size() > 1;
      const int* mask = NULL;
      const Dtype* top_mask = NULL;
      if (use_top_mask) {
        top_mask = top[1]->gpu_data();
      } else {
        mask = max_idx_.gpu_data();
      }

      ClKernel kernel = state.get_kernel("MaxPoolBackward");
      kernel.set_arg(argIdx++, count);
      kernel.set_arg(argIdx++, top_diff);
      kernel.set_arg(argIdx++, mask);
      kernel.set_arg(argIdx++, top_mask);
      kernel.set_arg(argIdx++, num);
      kernel.set_arg(argIdx++, channels_);
      kernel.set_arg(argIdx++, height_);
      kernel.set_arg(argIdx++, width_);
      kernel.set_arg(argIdx++, pooled_height_);
      kernel.set_arg(argIdx++, pooled_width_);
      kernel.set_arg(argIdx++, kernel_h_);
      kernel.set_arg(argIdx++, kernel_w_);
      kernel.set_arg(argIdx++, stride_h_);
      kernel.set_arg(argIdx++, stride_w_);
      kernel.set_arg(argIdx++, pad_h_);
      kernel.set_arg(argIdx++, pad_w_);
      kernel.set_arg(argIdx++, bottom_diff);
      kernel.enqueue(count);
      break;
    }
  case PoolingParameter_PoolMethod_AVE:
    {
      ClKernel kernel = state.get_kernel("AvePoolBackward");
      kernel.set_arg(argIdx++, count);
      kernel.set_arg(argIdx++, top_diff);
      kernel.set_arg(argIdx++, num);
      kernel.set_arg(argIdx++, channels_);
      kernel.set_arg(argIdx++, height_);
      kernel.set_arg(argIdx++, width_);
      kernel.set_arg(argIdx++, pooled_height_);
      kernel.set_arg(argIdx++, pooled_width_);
      kernel.set_arg(argIdx++, kernel_h_);
      kernel.set_arg(argIdx++, kernel_w_);
      kernel.set_arg(argIdx++, stride_h_);
      kernel.set_arg(argIdx++, stride_w_);
      kernel.set_arg(argIdx++, pad_h_);
      kernel.set_arg(argIdx++, pad_w_);
      kernel.set_arg(argIdx++, bottom_diff);
      kernel.enqueue(count);
      break;
    }
  case PoolingParameter_PoolMethod_STOCHASTIC:
    {
      ClKernel kernel = state.get_kernel("StoPoolBackward");
      kernel.set_arg(argIdx++, count);
      kernel.set_arg(argIdx++, rand_idx_.gpu_data());
      kernel.set_arg(argIdx++, top_diff);
      kernel.set_arg(argIdx++, num);
      kernel.set_arg(argIdx++, channels_);
      kernel.set_arg(argIdx++, height_);
      kernel.set_arg(argIdx++, width_);
      kernel.set_arg(argIdx++, pooled_height_);
      kernel.set_arg(argIdx++, pooled_width_);
      kernel.set_arg(argIdx++, kernel_h_);
      kernel.set_arg(argIdx++, kernel_w_);
      kernel.set_arg(argIdx++, stride_h_);
      kernel.set_arg(argIdx++, stride_w_);
      kernel.set_arg(argIdx++, bottom_diff);
      kernel.enqueue(count);
      break;
    }
  default:
    {
      LOG(FATAL) << "Unknown pooling method.";
    }
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(PoolingLayer);


}  // namespace caffe
#endif  // USE_OCL
