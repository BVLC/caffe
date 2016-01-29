#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea_im2col.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

#ifdef USE_CUDA
template<typename Dtype>
__global__ void SoftmaxLossForwardGPU(const int_tp nthreads,
                                      const Dtype* prob_data,
                                      const Dtype* label, Dtype* loss,
                                      const int_tp num, const int_tp dim,
                                      const int_tp spatial_dim,
                                      const bool has_ignore_label_,
                                      const int_tp ignore_label_,
                                      Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int_tp n = index / spatial_dim;
    const int_tp s = index % spatial_dim;
    const int_tp label_value = static_cast<int_tp>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -log(
          max(prob_data[n * dim + label_value * spatial_dim + s],
              Dtype(FLT_MIN)));
      counts[index] = 1;
    }
  }
}
#endif  // USE_CUDA

template<typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* label = bottom[1]->gpu_data();
    const int_tp dim = prob_.count() / outer_num_;
    const int_tp nthreads = outer_num_ * inner_num_;
    // Since this memory is not used for anything until it is overwritten
    // on the backward pass, we use it here to avoid having to allocate new GPU
    // memory to accumulate intermediate results in the kernel.
    Dtype* loss_data = bottom[0]->mutable_gpu_diff();
    // Similarly, this memory is never used elsewhere, and thus we can use it
    // to avoid having to allocate additional GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxLossForwardGPU<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS)(nthreads, prob_data,
            label, loss_data, outer_num_,
            dim, inner_num_, has_ignore_label_, ignore_label_, counts);
    Dtype loss;
    caffe_gpu_asum(nthreads, loss_data, &loss);
    Dtype valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if (normalization_ == LossParameter_NormalizationMode_VALID
        && has_ignore_label_) {
      caffe_gpu_asum(nthreads, counts, &valid_count);
    }
    top[0]->mutable_cpu_data()[0] = loss
        / get_normalizer(normalization_, valid_count);
    if (top.size() >= 2) {
      top[1]->ShareData(prob_);
    }

#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();

    cl_mem prob_data = (cl_mem) (prob_.gpu_data());
    cl_mem label = (cl_mem) (bottom[1]->gpu_data());
    const int_tp dim = prob_.count() / outer_num_;
    const int_tp nthreads = outer_num_ * inner_num_;
    cl_mem loss_data = (cl_mem) (bottom[0]->mutable_gpu_diff());
    cl_mem counts = (cl_mem) (prob_.mutable_gpu_diff());

    viennacl::ocl::kernel &oclk_softmax_loss_forward = program.get_kernel(
        CL_KERNEL_SELECT("softmax_loss_forward"));
    viennacl::ocl::enqueue(
        oclk_softmax_loss_forward(nthreads, WrapHandle(prob_data, &ctx),
                                  WrapHandle(label, &ctx),
                                  WrapHandle(loss_data, &ctx), outer_num_, dim,
                                  inner_num_, has_ignore_label_ ? 1 : 0,
                                  ignore_label_, WrapHandle(counts, &ctx)),
        ctx.get_queue());

    Dtype loss;
    greentea_gpu_asum<Dtype>(this->device_->id(), nthreads, loss_data, 0,
                             &loss);
    Dtype valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if (normalization_ == LossParameter_NormalizationMode_VALID
        && has_ignore_label_) {
      greentea_gpu_asum<Dtype>(this->device_->id(), nthreads, counts, 0,
                               &valid_count);
    }
    top[0]->mutable_cpu_data()[0] = loss
        / get_normalizer(normalization_, valid_count);
    if (top.size() >= 2) {
      top[1]->ShareData(prob_);
    }
#endif  // USE_GREENTEA
  }
}

#ifdef USE_CUDA
template<typename Dtype>
__global__ void SoftmaxLossBackwardGPU(const int_tp nthreads, const Dtype* top,
                                       const Dtype* label, Dtype* bottom_diff,
                                       const int_tp num, const int_tp dim,
                                       const int_tp spatial_dim,
                                       const bool has_ignore_label_,
                                       const int_tp ignore_label_,
                                       Dtype* counts) {
  const int_tp channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int_tp n = index / spatial_dim;
    const int_tp s = index % spatial_dim;
    const int_tp label_value = static_cast<int_tp>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int_tp c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}
#endif  // USE_CUDA

template<typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) <<
        this->type() << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      const Dtype* prob_data = prob_.gpu_data();
      const Dtype* top_data = top[0]->gpu_data();
      caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
      const Dtype* label = bottom[1]->gpu_data();
      const int_tp dim = prob_.count() / outer_num_;
      const int_tp nthreads = outer_num_ * inner_num_;
      // Since this memory is never used for anything else,
      // we use to to avoid allocating new GPU memory.
      Dtype* counts = prob_.mutable_gpu_diff();
      // NOLINT_NEXT_LINE(whitespace/operators)
      SoftmaxLossBackwardGPU<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS) (nthreads, top_data, label, bottom_diff,
          outer_num_, dim, inner_num_, has_ignore_label_,
          ignore_label_, counts);

      Dtype valid_count = -1;
      if (normalization_ == LossParameter_NormalizationMode_VALID &&
          has_ignore_label_) {
        caffe_gpu_asum(nthreads, counts, &valid_count);
      }
      const Dtype loss_weight = top[0]->cpu_diff()[0] /
      get_normalizer(normalization_, valid_count);
      caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);
#endif  // USE_CUDA
    } else {
#ifdef USE_GREENTEA
      viennacl::ocl::context &ctx = viennacl::ocl::get_context(
          this->device_->id());
      viennacl::ocl::program &program = this->device_->program();

      cl_mem bottom_diff = (cl_mem)(bottom[0]->mutable_gpu_diff());
      cl_mem prob_data = (cl_mem)(prob_.gpu_data());
      cl_mem top_data = (cl_mem)(top[0]->gpu_data());
      greentea_gpu_memcpy(prob_.count() * sizeof(Dtype),
          prob_data, 0, bottom_diff, 0, &ctx);
      cl_mem label = (cl_mem)(bottom[1]->gpu_data());
      const int_tp dim = prob_.count() / outer_num_;
      const int_tp nthreads = outer_num_ * inner_num_;
      cl_mem counts = (cl_mem)(prob_.mutable_gpu_diff());

      viennacl::ocl::kernel &oclk_softmax_loss_backward = program.get_kernel(
          CL_KERNEL_SELECT("softmax_loss_backward"));
      viennacl::ocl::enqueue(
          oclk_softmax_loss_backward(nthreads, WrapHandle(top_data, &ctx),
              WrapHandle(label, &ctx), WrapHandle(bottom_diff, &ctx),
              outer_num_, dim, inner_num_, has_ignore_label_ ? 1 : 0,
              ignore_label_, WrapHandle(counts, &ctx)),
          ctx.get_queue());

      Dtype valid_count = -1;
      if (normalization_ == LossParameter_NormalizationMode_VALID &&
          has_ignore_label_) {
        greentea_gpu_asum<Dtype>(this->device_->id(),
            nthreads, counts, 0, &valid_count);
      }
      const Dtype loss_weight = top[0]->cpu_diff()[0] /
      get_normalizer(normalization_, valid_count);
      greentea_gpu_scal<Dtype>(this->device_->id(),
          prob_.count(), loss_weight, bottom_diff, 0);
#endif  // USE_GREENTEA
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossLayer);

}  // namespace caffe
