#ifdef USE_OCL
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/eltwise_layer.hpp"

extern "C" const char _cl_eltwise_layer_start;
extern "C" const char _cl_eltwise_layer_end;

namespace caffe {

template <typename Dtype>
void EltwiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int* mask = NULL;
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();
  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
    caffe_gpu_mul(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
        top_data);
    for (int i = 2; i < bottom.size(); ++i) {
      caffe_gpu_mul(count, top_data, bottom[i]->gpu_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    caffe_gpu_set(count, Dtype(0.), top_data);
    // TODO(shelhamer) does cuBLAS optimize to sum for coeff = 1?
    for (int i = 0; i < bottom.size(); ++i) {
      caffe_gpu_axpy(count, coeffs_[i], bottom[i]->gpu_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_MAX:
  {
    mask = max_idx_.mutable_gpu_data();
    cl_uint argIdx = 0;
    ClState& state = Caffe::cl_state();
    state.submit_program("eltwise", &_cl_eltwise_layer_start,
        &_cl_eltwise_layer_end);

    const Dtype* bottom_data0 = bottom[0]->gpu_data();
    const Dtype* bottom_data1 = bottom[1]->gpu_data();
    int blob_idx = 0;

    ClKernel kernel = state.get_kernel("MaxForward");

    // Compute inner1d(top_diff, top_data) and subtract them from the
    //   bottom diff
    kernel.set_arg(argIdx++, count);
    kernel.set_arg(argIdx++, bottom_data0);
    kernel.set_arg(argIdx++, bottom_data1);
    kernel.set_arg(argIdx++, blob_idx);
    kernel.set_arg(argIdx++, top_data);
    kernel.set_arg(argIdx++, mask);
    kernel.enqueue(count);

    for (int i = 2; i < bottom.size(); ++i) {
      argIdx = 0;
      const Dtype* bottom_datai = bottom[i]->gpu_data();
      blob_idx = i-1;

      // Compute inner1d(top_diff, top_data) and subtract them from the
      //   bottom diff
      kernel.set_arg(argIdx++, count);
      kernel.set_arg(argIdx++, top_data);
      kernel.set_arg(argIdx++, bottom_datai);
      kernel.set_arg(argIdx++, blob_idx);
      kernel.set_arg(argIdx++, top_data);
      kernel.set_arg(argIdx++, mask);
      kernel.enqueue(count);
    }
    break;
  }
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
}

template <typename Dtype>
void EltwiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int* mask = NULL;
  const int count = top[0]->count();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      switch (op_) {
      case EltwiseParameter_EltwiseOp_PROD:
        if (stable_prod_grad_) {
          bool initialized = false;
          for (int j = 0; j < bottom.size(); ++j) {
            if (i == j) { continue; }
            if (!initialized) {
              caffe_copy(count, bottom[j]->gpu_data(), bottom_diff);
              initialized = true;
            } else {
              caffe_gpu_mul(count, bottom[j]->gpu_data(), bottom_diff,
                            bottom_diff);
            }
          }
        } else {
          caffe_gpu_div(count, top_data, bottom_data, bottom_diff);
        }
        caffe_gpu_mul(count, bottom_diff, top_diff, bottom_diff);
        break;
      case EltwiseParameter_EltwiseOp_SUM:
        if (coeffs_[i] == Dtype(1.)) {
          caffe_copy(count, top_diff, bottom_diff);
        } else {
          caffe_gpu_scale(count, coeffs_[i], top_diff, bottom_diff);
        }
        break;
      case EltwiseParameter_EltwiseOp_MAX:
      {
        mask = max_idx_.gpu_data();
        cl_uint argIdx = 0;
        ClState& state = Caffe::cl_state();
        state.submit_program("eltwise", &_cl_eltwise_layer_start,
            &_cl_eltwise_layer_end);

        // Compute inner1d(top_diff, top_data) and subtract them from the
        //   bottom diff
        ClKernel kernel = state.get_kernel("MaxBackward");
        kernel.set_arg(argIdx++, count);
        kernel.set_arg(argIdx++, top_diff);
        kernel.set_arg(argIdx++, i);
        kernel.set_arg(argIdx++, mask);
        kernel.set_arg(argIdx++, bottom_diff);
        kernel.enqueue(count);
        break;
      }
      default:
        LOG(FATAL) << "Unknown elementwise operation.";
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EltwiseLayer);

}  // namespace caffe
#endif  // USE_OCL
