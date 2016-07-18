#include <cfloat>
#include <vector>

#include "caffe/layers/scale_layer.hpp"
#include "caffe/util/math_functions.hpp"


#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
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

  if (bottom[0] == top[0]) {
    caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(),
                   temp_.mutable_gpu_data());
  }
  const Dtype* scale_data = (
      (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  ClState& clState = Caffe::cl_state();
  ClMemOff<Dtype> buf_scale = clState.get_buffer_mem(scale_data);
  ClMemOff<Dtype> buf_bottom = clState.get_buffer_mem(bottom_data);
  ClMemOff<Dtype> buf_top = clState.get_buffer_mem(top_data);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
  viennacl::ocl::program &program = this->device_->program();

  if (bias_layer_) {
    const Dtype* bias_data = this->blobs_[bias_param_id_]->gpu_data();
    viennacl::ocl::kernel &oclk_scale_bias_forward = program.get_kernel(
        CL_KERNEL_SELECT("scale_bias_forward"));

    ClMemOff<Dtype> buf_bias = clState.get_buffer_mem(bias_data);
    viennacl::ocl::enqueue(
        oclk_scale_bias_forward(count, WrapHandle(buf_bottom.memobj, &ctx),
                                WrapHandle(buf_scale.memobj, &ctx),
                                WrapHandle(buf_bias.memobj, &ctx),
                                scale_dim_, inner_dim_,
                                WrapHandle(buf_top.memobj, &ctx)),
        ctx.get_queue());
  } else {
    viennacl::ocl::kernel &oclk_scale_forward = program.get_kernel(
        CL_KERNEL_SELECT("scale_forward"));
    viennacl::ocl::enqueue(
        oclk_scale_forward(count, WrapHandle(buf_bottom.memobj, &ctx),
                           WrapHandle(buf_scale.memobj, &ctx), scale_dim_,
                           inner_dim_, WrapHandle(buf_top.memobj, &ctx)),
        ctx.get_queue());
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
        caffe_gpu_gemv<Dtype>(CblasNoTrans, sum_result_.count(), inner_dim_,
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
          caffe_gpu_gemv<Dtype>(CblasTrans, outer_dim_, scale_dim_, Dtype(1),
                         sum_result, sum_mult, Dtype(scale_param), scale_diff);
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

    ClState& clState = Caffe::cl_state();
    ClMemOff<Dtype> buf_bottom_diff = clState.get_buffer_mem(bottom_diff);
    ClMemOff<Dtype> buf_top_diff = clState.get_buffer_mem(top_diff);
    ClMemOff<Dtype> buf_scale = clState.get_buffer_mem(scale_data);

    viennacl::ocl::enqueue(
        oclk_scale_forward(count, WrapHandle(buf_top_diff.memobj, &ctx),
                           WrapHandle(buf_scale.memobj, &ctx), scale_dim_,
                           inner_dim_,
                           WrapHandle(buf_bottom_diff.memobj, &ctx)),
        ctx.get_queue());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ScaleLayer);

}  // namespace caffe
