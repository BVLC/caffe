#include <cfloat>
#include <vector>

#include "caffe/layers/scale_layer.hpp"
#include "caffe/util/math_functions.hpp"


#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

#ifdef USE_CUDA
template <typename Dtype>
__global__ void ScaleForward(const int_tp n, const Dtype* in,
    const Dtype* scale, const int_tp scale_dim, const int_tp inner_dim,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int_tp scale_index = (index / inner_dim) % scale_dim;
    out[index] = in[index] * scale[scale_index];
  }
}

template <typename Dtype>
__global__ void ScaleBiasForward(const int_tp n, const Dtype* in,
    const Dtype* scale, const Dtype* bias,
    const int_tp scale_dim, const int_tp inner_dim, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int_tp scale_index = (index / inner_dim) % scale_dim;
    out[index] = in[index] * scale[scale_index] + bias[scale_index];
  }
}
#endif  // USE_CUDA

template<typename Dtype>
void ScaleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  const int_tp count = top[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    if (bottom[0] == top[0]) {
      caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(),
                     temp_.mutable_gpu_data());
    }
    const Dtype* scale_data = (
        (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    if (bias_layer_) {
      const Dtype* bias_data = this->blobs_[bias_param_id_]->gpu_data();
      ScaleBiasForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      CUDA_KERNEL(CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS)(
          count, bottom_data, scale_data, bias_data, scale_dim_, inner_dim_,
          top_data);
    } else {
      ScaleForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      CUDA_KERNEL(CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS)(
          count, bottom_data, scale_data, scale_dim_, inner_dim_, top_data);
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();

    if (bottom[0] == top[0]) {
      greentea_copy<Dtype>(bottom[0]->count(), (cl_mem) (bottom[0]->gpu_data()),
                           0, (cl_mem) (temp_.mutable_gpu_data()), 0, &ctx);
    }
    const Dtype* scale_data = (
        (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    if (bias_layer_) {
      const Dtype* bias_data = this->blobs_[bias_param_id_]->gpu_data();
      viennacl::ocl::kernel &oclk_scale_bias_forward = program.get_kernel(
          CL_KERNEL_SELECT("scale_bias_forward"));
      viennacl::ocl::enqueue(
          oclk_scale_bias_forward(count, WrapHandle((cl_mem) bottom_data, &ctx),
                                  WrapHandle((cl_mem) scale_data, &ctx),
                                  WrapHandle((cl_mem) bias_data, &ctx),
                                  scale_dim_, inner_dim_,
                                  WrapHandle((cl_mem) top_data, &ctx)),
          ctx.get_queue());
    } else {
      viennacl::ocl::kernel &oclk_scale_forward = program.get_kernel(
          CL_KERNEL_SELECT("scale_forward"));
      viennacl::ocl::enqueue(
          oclk_scale_forward(count, WrapHandle((cl_mem)bottom_data, &ctx),
                             WrapHandle((cl_mem)scale_data, &ctx), scale_dim_,
                             inner_dim_, WrapHandle((cl_mem)top_data, &ctx)),
          ctx.get_queue());
    }
#endif  // USE_GREENTEA
  }
}

template<typename Dtype>
void ScaleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                     const vector<bool>& propagate_down,
                                     const vector<Blob<Dtype>*>& bottom) {
  if (bias_layer_
      && this->param_propagate_down_[this->param_propagate_down_.size() - 1]) {
    bias_layer_->Backward(top, bias_propagate_down_, bias_bottom_vec_);
  }
  const bool scale_param = (bottom.size() == 1);
  Blob<Dtype>* scale = scale_param ? this->blobs_[0].get() : bottom[1];

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    if ((!scale_param && propagate_down[1])
        || (scale_param && this->param_propagate_down_[0])) {
      const Dtype* top_diff = top[0]->gpu_diff();
      const bool in_place = (bottom[0] == top[0]);
      const Dtype* bottom_data = (in_place ? &temp_ : bottom[0])->gpu_data();
      const bool is_eltwise = (bottom[0]->count() == scale->count());
      Dtype* product = (
          is_eltwise ?
              scale->mutable_gpu_diff() :
              (in_place ?
                  temp_.mutable_gpu_data() : bottom[0]->mutable_gpu_diff()));
      caffe_gpu_mul(top[0]->count(), top_diff, bottom_data, product);
      if (!is_eltwise) {
        Dtype* sum_result = NULL;
        if (inner_dim_ == 1) {
          sum_result = product;
        } else if (sum_result_.count() == 1) {
          const Dtype* sum_mult = sum_multiplier_.gpu_data();
          Dtype* scale_diff = scale->mutable_cpu_diff();
          if (scale_param) {
            Dtype result;
            caffe_gpu_dot(inner_dim_, product, sum_mult, &result);
            *scale_diff += result;
          } else {
            caffe_gpu_dot(inner_dim_, product, sum_mult, scale_diff);
          }
        } else {
          const Dtype* sum_mult = sum_multiplier_.gpu_data();
          sum_result =
              (outer_dim_ == 1) ?
                  scale->mutable_gpu_diff() : sum_result_.mutable_gpu_data();
          caffe_gpu_gemv(CblasNoTrans, sum_result_.count(), inner_dim_,
                         Dtype(1), product, sum_mult, Dtype(0), sum_result);
        }
        if (outer_dim_ != 1) {
          const Dtype* sum_mult = sum_multiplier_.gpu_data();
          if (scale_dim_ == 1) {
            Dtype* scale_diff = scale->mutable_cpu_diff();
            if (scale_param) {
              Dtype result;
              caffe_gpu_dot(outer_dim_, sum_mult, sum_result, &result);
              *scale_diff += result;
            } else {
              caffe_gpu_dot(outer_dim_, sum_mult, sum_result, scale_diff);
            }
          } else {
            Dtype* scale_diff = scale->mutable_gpu_diff();
            caffe_gpu_gemv(CblasTrans, outer_dim_, scale_dim_, Dtype(1),
                           sum_result, sum_mult, Dtype(scale_param),
                           scale_diff);
          }
        }
      }
    }
    if (propagate_down[0]) {
      const int_tp count = top[0]->count();
      const Dtype* top_diff = top[0]->gpu_diff();
      const Dtype* scale_data = scale->gpu_data();
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      ScaleForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      CUDA_KERNEL(CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS)(
          count, top_diff, scale_data, scale_dim_, inner_dim_, bottom_diff);
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();

    if ((!scale_param && propagate_down[1])
        || (scale_param && this->param_propagate_down_[0])) {
      const Dtype* top_diff = top[0]->gpu_diff();
      const bool in_place = (bottom[0] == top[0]);
      const Dtype* bottom_data = (in_place ? &temp_ : bottom[0])->gpu_data();
      const bool is_eltwise = (bottom[0]->count() == scale->count());
      Dtype* product = (
          is_eltwise ?
              scale->mutable_gpu_diff() :
              (in_place ?
                  temp_.mutable_gpu_data() : bottom[0]->mutable_gpu_diff()));
      greentea_gpu_mul<Dtype>(this->device_->id(), top[0]->count(),
                              (cl_mem) top_diff, 0, (cl_mem) bottom_data, 0,
                              (cl_mem) product, 0);
      if (!is_eltwise) {
        Dtype* sum_result = NULL;
        if (inner_dim_ == 1) {
          sum_result = product;
        } else if (sum_result_.count() == 1) {
          const Dtype* sum_mult = sum_multiplier_.gpu_data();
          Dtype* scale_diff = scale->mutable_cpu_diff();
          if (scale_param) {
            Dtype result;
            greentea_gpu_dot<Dtype>(this->device_->id(), inner_dim_,
                                    (cl_mem) product, 0, (cl_mem) sum_mult, 0,
                                    &result);
            *scale_diff += result;
          } else {
            greentea_gpu_dot<Dtype>(this->device_->id(), inner_dim_,
                                    (cl_mem) product, 0, (cl_mem) sum_mult, 0,
                                    scale_diff);
          }
        } else {
          const Dtype* sum_mult = sum_multiplier_.gpu_data();
          sum_result =
              (outer_dim_ == 1) ?
                  scale->mutable_gpu_diff() : sum_result_.mutable_gpu_data();
          greentea_gpu_gemv<Dtype>(this->device_->id(), CblasNoTrans,
                                   sum_result_.count(), inner_dim_, Dtype(1),
                                   (cl_mem) product, 0, (cl_mem) sum_mult, 0,
                                   Dtype(0), (cl_mem) sum_result, 0);
        }
        if (outer_dim_ != 1) {
          const Dtype* sum_mult = sum_multiplier_.gpu_data();
          if (scale_dim_ == 1) {
            Dtype* scale_diff = scale->mutable_cpu_diff();
            if (scale_param) {
              Dtype result;
              greentea_gpu_dot<Dtype>(this->device_->id(), outer_dim_,
                                      (cl_mem) sum_mult, 0, (cl_mem) sum_result,
                                      0, &result);
              *scale_diff += result;
            } else {
              greentea_gpu_dot<Dtype>(this->device_->id(), outer_dim_,
                                      (cl_mem) sum_mult, 0, (cl_mem) sum_result,
                                      0, scale_diff);
            }
          } else {
            Dtype* scale_diff = scale->mutable_gpu_diff();
            greentea_gpu_gemv<Dtype>(this->device_->id(), CblasTrans,
                                     outer_dim_, scale_dim_, Dtype(1),
                                     (cl_mem) sum_result, 0, (cl_mem) sum_mult,
                                     0, Dtype(scale_param), (cl_mem) scale_diff,
                                     0);
          }
        }
      }
    }
    if (propagate_down[0]) {
      const int_tp count = top[0]->count();
      const Dtype* top_diff = top[0]->gpu_diff();
      const Dtype* scale_data = scale->gpu_data();
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      viennacl::ocl::kernel &oclk_scale_forward = program.get_kernel(
          CL_KERNEL_SELECT("scale_forward"));
      viennacl::ocl::enqueue(
          oclk_scale_forward(count, WrapHandle((cl_mem) top_diff, &ctx),
                             WrapHandle((cl_mem) scale_data, &ctx), scale_dim_,
                             inner_dim_,
                             WrapHandle((cl_mem) bottom_diff, &ctx)),
          ctx.get_queue());
    }
#endif  // USE_GREENTEA
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ScaleLayer);

}  // namespace caffe
