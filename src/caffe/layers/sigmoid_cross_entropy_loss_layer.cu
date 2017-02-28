#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

#ifdef USE_CUDA
template <typename Dtype>
__global__ void SigmoidCrossEntropyLossForwardGPU(const int_tp nthreads,
          const Dtype* input_data, const Dtype* target, Dtype* loss,
          const bool has_ignore_label_, const int_tp ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    const int_tp target_value = static_cast<int_tp>(target[i]);
    if (has_ignore_label_ && target_value == ignore_label_) {
      loss[i] = 0;
      counts[i] = 0;
    } else {
      loss[i] = input_data[i] * (target[i] - (input_data[i] >= 0)) -
          log(1 + exp(input_data[i] - 2 * input_data[i] *
          (input_data[i] >= 0)));
      counts[i] = 1;
    }
  }
}

template <typename Dtype>
__global__ void SigmoidCrossEntropyLossIgnoreDiffGPU(const int_tp count,
    const int_tp ignore_label, const Dtype* target, Dtype* diff) {
  CUDA_KERNEL_LOOP(i, count) {
    const int_tp target_value = static_cast<int_tp>(target[i]);
    if (target_value == ignore_label) {
      diff[i] = 0;
    }
  }
}
#endif  // USE_CUDA


template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int_tp count = bottom[0]->count();
  const int_tp num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* target = bottom[1]->gpu_data();
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();

  Dtype loss;
  Dtype* count_data = bottom[1]->mutable_gpu_diff();
  Dtype valid_count;

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    // NOLINT_NEXT_LINE(whitespace/operators)
    SigmoidCrossEntropyLossForwardGPU<Dtype>
      CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS)(count, input_data, target, loss_data,
            has_ignore_label_, ignore_label_, count_data);
    caffe_gpu_asum(count, loss_data, &loss);
    // Only launch another CUDA kernel if we actually need the valid count.
    if (normalization_ == LossParameter_NormalizationMode_VALID &&
        has_ignore_label_) {
      caffe_gpu_asum(count, count_data, &valid_count);
    } else {
      valid_count = count;
    }
    caffe_gpu_asum(count, loss_data, &loss);
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();
    viennacl::ocl::kernel &oclk_sce_loss_forward =
        program.get_kernel(
            CL_KERNEL_SELECT("sce_loss_forward"));
    viennacl::ocl::enqueue(
        oclk_sce_loss_forward(count,
            WrapHandle((cl_mem)input_data, &ctx),
            WrapHandle((cl_mem)target, &ctx),
            WrapHandle((cl_mem)loss_data, &ctx),
            has_ignore_label_?1:0, ignore_label_,
            WrapHandle((cl_mem)count_data, &ctx)),
        ctx.get_queue());
    // Only launch another OpenCL kernel if we actually need the valid count.
    if (normalization_ == LossParameter_NormalizationMode_VALID &&
        has_ignore_label_) {
      greentea_gpu_asum<Dtype>(this->device_->id(),
                               count, (cl_mem)count_data, 0, &valid_count);
    } else {
      valid_count = count;
    }
    greentea_gpu_asum<Dtype>(this->device_->id(),
                             count, (cl_mem)loss_data, 0, &loss);
#endif  // USE_GREENTEA
  }
  normalizer_ = get_normalizer(normalization_, valid_count);
  top[0]->mutable_cpu_data()[0] = loss / normalizer_;
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL)<< this->type()
    << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const int_tp count = bottom[0]->count();
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
      // First, compute the diff
      caffe_copy(count, sigmoid_output_data, bottom_diff);
      caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);
      // Zero out gradient of ignored targets.
      if (has_ignore_label_) {
        // NOLINT_NEXT_LINE(whitespace/operators)
        SigmoidCrossEntropyLossIgnoreDiffGPU<Dtype>
          CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
          CAFFE_CUDA_NUM_THREADS)(count, ignore_label_, target, bottom_diff);
      }
      // Scale down gradient
      Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer_;
      caffe_gpu_scal(count, loss_weight, bottom_diff);
#endif  // USE_CUDA
    } else {
#ifdef USE_GREENTEA
      viennacl::ocl::context &ctx = viennacl::ocl::get_context(
          this->device_->id());
      viennacl::ocl::program &program = this->device_->program();

      // First, compute the diff
      greentea_copy<Dtype>(count, (cl_mem)sigmoid_output_data, 0,
                           (cl_mem)bottom_diff, 0, &ctx);
      greentea_gpu_axpy<Dtype>(this->device_->id(), count,
                               Dtype(-1), (cl_mem)target, 0,
                               (cl_mem)bottom_diff, 0);

      // Zero out gradient of ignored targets.
      if (has_ignore_label_) {
        viennacl::ocl::kernel &oclk_sce_loss_ignore_diff = program.get_kernel(
            CL_KERNEL_SELECT("sce_loss_ignore_diff"));

        viennacl::ocl::enqueue(
            oclk_sce_loss_ignore_diff(count,
                    ignore_label_,
                    WrapHandle((cl_mem)(target), &ctx),
                    WrapHandle((cl_mem)(bottom_diff), &ctx)),
            ctx.get_queue());
      }

      // Scale down gradient
      Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer_;
      greentea_gpu_scal(this->device_->id(), count, loss_weight,
                        (cl_mem)bottom_diff, 0);
#endif  // USE_GREENTEA
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SigmoidCrossEntropyLossLayer);

}  // namespace caffe
