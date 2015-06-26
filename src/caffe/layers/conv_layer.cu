#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_im2col.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
#ifdef USE_GREENTEA
      if (this->device_context_->backend() == BACKEND_OpenCL) {
        viennacl::ocl::context &ctx =
            viennacl::ocl::get_context(this->device_context_->id());
         // ctx.switch_queue(n % GREENTEA_QUEUE_COUNT);
      }
#endif  // USE_GREENTEA

      this->forward_gpu_gemm(bottom_data, bottom[i]->offset(n), weight,
          top_data, top[i]->offset(n));
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data, top[i]->offset(n), bias);
      }
    }
#ifdef USE_GREENTEA
      if (this->device_context_->backend() == BACKEND_OpenCL) {
        viennacl::ocl::context &ctx =
            viennacl::ocl::get_context(this->device_context_->id());
        FinishQueues(&ctx);
      }
#endif  // USE_GREENTEA
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
#ifdef USE_GREENTEA
        if (this->device_context_->backend() == BACKEND_OpenCL) {
          viennacl::ocl::context &ctx =
              viennacl::ocl::get_context(this->device_context_->id());
          // ctx.switch_queue(n % GREENTEA_QUEUE_COUNT);
        }
#endif  // USE_GREENTEA

        this->backward_gpu_bias(bias_diff, top_diff, top[i]->offset(n));

#ifdef USE_GREENTEA
        if (this->device_context_->backend() == BACKEND_OpenCL) {
          viennacl::ocl::context &ctx =
              viennacl::ocl::get_context(this->device_context_->id());
          FinishQueues(&ctx);
        }
#endif  // USE_GREENTEA
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
#ifdef USE_GREENTEA
        if (this->device_context_->backend() == BACKEND_OpenCL) {
          viennacl::ocl::context &ctx =
              viennacl::ocl::get_context(this->device_context_->id());
           // ctx.switch_queue(n % GREENTEA_QUEUE_COUNT);
        }
#endif  // USE_GREENTEA

        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data, bottom[i]->offset(n),
              top_diff, top[i]->offset(n), weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff, top[i]->offset(n), weight,
              bottom_diff, bottom[i]->offset(n));
        }

#ifdef USE_GREENTEA
        if (this->device_context_->backend() == BACKEND_OpenCL) {
          viennacl::ocl::context &ctx =
              viennacl::ocl::get_context(this->device_context_->id());
          FinishQueues(&ctx);
        }
#endif  // USE_GREENTEA
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
