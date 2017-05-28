#include <cfloat>
#include <vector>

#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

#ifdef USE_CUDA
template<typename Dtype>
__global__ void MaxForward(const int_tp nthreads, const Dtype* bottom_data_a,
                           const Dtype* bottom_data_b, const int_tp blob_idx,
                           Dtype* top_data, int_tp* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype maxval = -FLT_MAX;
    int_tp maxidx = -1;
    if (bottom_data_a[index] > bottom_data_b[index]) {
      // only update for very first bottom_data blob (blob_idx == 0)
      if (blob_idx == 0) {
        maxval = bottom_data_a[index];
        top_data[index] = maxval;
        maxidx = blob_idx;
        mask[index] = maxidx;
      }
    } else {
      maxval = bottom_data_b[index];
      top_data[index] = maxval;
      maxidx = blob_idx + 1;
      mask[index] = maxidx;
    }
  }
}
#endif  // USE_CUDA

template<typename Dtype>
void EltwiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  int_tp* mask = NULL;
  const int_tp count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    switch (op_) {
      case EltwiseParameter_EltwiseOp_PROD:
        caffe_gpu_mul(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
                      top_data);
        for (int_tp i = 2; i < bottom.size(); ++i) {
          caffe_gpu_mul(count, top_data, bottom[i]->gpu_data(), top_data);
        }
        break;
      case EltwiseParameter_EltwiseOp_SUM:
        caffe_gpu_set(count, Dtype(0.), top_data);
        // TODO(shelhamer) does cuBLAS optimize to sum for coeff = 1?
        for (int_tp i = 0; i < bottom.size(); ++i) {
          caffe_gpu_axpy(count, coeffs_[i], bottom[i]->gpu_data(), top_data);
        }
        break;
      case EltwiseParameter_EltwiseOp_MAX:
        mask = max_idx_.mutable_gpu_data();
        // NOLINT_NEXT_LINE(whitespace/operators)
        MaxForward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
                                      CAFFE_CUDA_NUM_THREADS)(
            count, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
            0, top_data, mask);
        for (int_tp i = 2; i < bottom.size(); ++i) {
          // NOLINT_NEXT_LINE(whitespace/operators)
          MaxForward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
                                        CAFFE_CUDA_NUM_THREADS)(
              count, top_data, bottom[i]->gpu_data(), i-1, top_data, mask);
        }
        break;
      default: {
        LOG(FATAL)<< "Unknown elementwise operation.";
      }
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();

    switch (op_) {
      case EltwiseParameter_EltwiseOp_PROD: {
        greentea_gpu_mul<Dtype>(this->device_->id(),
                                count, (cl_mem)(bottom[0]->gpu_data()), 0,
                                (cl_mem)(bottom[1]->gpu_data()), 0,
            (cl_mem)top_data, 0);
        for (int_tp i = 2; i < bottom.size(); ++i) {
          greentea_gpu_mul<Dtype>(this->device_->id(),
                                  count, (cl_mem)top_data, 0,
                                  (cl_mem)(bottom[i]->gpu_data()), 0,
                                  (cl_mem)top_data, 0);
        }
      }
      break;
      case EltwiseParameter_EltwiseOp_SUM: {
        greentea_gpu_set<Dtype>(this->device_->id(), count, 0,
                                (cl_mem)top_data, 0);
        for (int_tp i = 0; i < bottom.size(); ++i) {
          greentea_gpu_axpy<Dtype>(this->device_->id(),
                                   count, coeffs_[i],
                                   (cl_mem)(bottom[i]->gpu_data()),
                                   0, (cl_mem)top_data, 0);
        }
      }
      break;
      case EltwiseParameter_EltwiseOp_MAX: {
        mask = max_idx_.mutable_gpu_data();

        viennacl::ocl::kernel &oclk_max_forward = program.get_kernel(
            CL_KERNEL_SELECT("eltwise_max_forward"));

        viennacl::ocl::enqueue(
            oclk_max_forward(count,
                WrapHandle((cl_mem)(bottom[0]->gpu_data()), &ctx),
                WrapHandle((cl_mem)(bottom[1]->gpu_data()), &ctx), (int_tp)0,
                WrapHandle((cl_mem)top_data, &ctx),
                WrapHandle((cl_mem)mask, &ctx)),
            ctx.get_queue());

        for (int_tp i = 2; i < bottom.size(); ++i) {
          viennacl::ocl::enqueue(
              oclk_max_forward(count, WrapHandle((cl_mem)(top_data), &ctx),
                  WrapHandle((cl_mem)(bottom[i]->gpu_data()), &ctx), i-1,
                  WrapHandle((cl_mem)top_data, &ctx),
                  WrapHandle((cl_mem)mask, &ctx)),
              ctx.get_queue());
        }
      }
      break;
      default: {
        LOG(FATAL)<< "Unknown elementwise operation.";
      }
    }
#endif  // USE_GREENTEA
  }
}

#ifdef USE_CUDA
template<typename Dtype>
__global__ void MaxBackward(const int_tp nthreads, const Dtype* top_diff,
                            const int_tp blob_idx, const int_tp* mask,
                            Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype gradient = 0;
    if (mask[index] == blob_idx) {
      gradient += top_diff[index];
    }
    bottom_diff[index] = gradient;
  }
}
#endif  // USE_CUDA

template<typename Dtype>
void EltwiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom) {
  const int_tp* mask = NULL;
  const int_tp count = top[0]->count();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    for (int_tp i = 0; i < bottom.size(); ++i) {
      if (propagate_down[i]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        switch (op_) {
          case EltwiseParameter_EltwiseOp_PROD:
            if (stable_prod_grad_) {
              bool initialized = false;
              for (int_tp j = 0; j < bottom.size(); ++j) {
                if (i == j) {
                  continue;
                }
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
            mask = max_idx_.gpu_data();
            MaxBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            CUDA_KERNEL(CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS)(
                count, top_diff, i, mask, bottom_diff);
            break;
          default: {
            LOG(FATAL)<< "Unknown elementwise operation.";
          }
        }
      }
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();

    for (int_tp i = 0; i < bottom.size(); ++i) {
      if (propagate_down[i]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        switch (op_) {
          case EltwiseParameter_EltwiseOp_PROD: {
            if (stable_prod_grad_) {
              bool initialized = false;
              for (int_tp j = 0; j < bottom.size(); ++j) {
                if (i == j) {
                  continue;
                }
                if (!initialized) {
                  greentea_copy<Dtype>(count,
                      (cl_mem)(bottom[j]->gpu_data()), 0,
                      (cl_mem)(bottom_diff), 0, &ctx);
                  initialized = true;
                } else {
                  greentea_gpu_mul<Dtype>(this->device_->id(), count,
                      (cl_mem)bottom[j]->gpu_data(), 0,
                      (cl_mem)bottom_diff, 0,
                      (cl_mem)bottom_diff, 0);
                }
              }
            } else {
              greentea_gpu_div<Dtype>(this->device_->id(),
                                      count, (cl_mem)top_data, 0,
                  (cl_mem)bottom_data, 0, (cl_mem)bottom_diff, 0);
            }
            greentea_gpu_mul<Dtype>(this->device_->id(),
                                    count, (cl_mem)bottom_diff, 0,
                (cl_mem)top_diff, 0, (cl_mem)bottom_diff, 0);
          }
          break;
          case EltwiseParameter_EltwiseOp_SUM: {
            if (coeffs_[i] == Dtype(1.)) {
              greentea_copy<Dtype>(count, (cl_mem)top_diff,
                                   0, (cl_mem)bottom_diff, 0, &ctx);
            } else {
              greentea_gpu_scale<Dtype>(this->device_->id(),
                  count, coeffs_[i], (cl_mem)top_diff,
                  0, (cl_mem)bottom_diff, 0);
            }
          }
          break;
          case EltwiseParameter_EltwiseOp_MAX: {
            mask = max_idx_.gpu_data();

            viennacl::ocl::kernel &oclk_max_backward = program.get_kernel(
                CL_KERNEL_SELECT("eltwise_max_backward"));

            viennacl::ocl::enqueue(
                oclk_max_backward(count, WrapHandle((cl_mem)top_diff, &ctx), i,
                    WrapHandle((cl_mem)mask, &ctx),
                    WrapHandle((cl_mem)bottom_diff, &ctx)),
                ctx.get_queue());
          }
          break;
          default: {
            LOG(FATAL)<< "Unknown elementwise operation.";
          }
        }
      }
    }
#endif
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EltwiseLayer);

}  // namespace caffe
