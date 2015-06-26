#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea_im2col.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

#ifdef USE_CUDA
template<typename Dtype>
__global__ void SoftmaxLossForwardGPU(const int nthreads,
                                      const Dtype* prob_data,
                                      const Dtype* label, Dtype* loss,
                                      const int num, const int dim,
                                      const int spatial_dim,
                                      const bool has_ignore_label_,
                                      const int ignore_label_, Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
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

  if (this->device_context_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* label = bottom[1]->gpu_data();
    const int num = prob_.num();
    const int dim = prob_.count() / num;
    const int spatial_dim = prob_.height() * prob_.width();
    const int nthreads = num * spatial_dim;
    // Since this memory is not used for anything until it is overwritten
    // on the backward pass, we use it here to avoid having to allocate new GPU
    // memory to accumulate intermediate results in the kernel.
    Dtype* loss_data = bottom[0]->mutable_gpu_diff();
    // Similarly, this memory is never used elsewhere, and thus we can use it
    // to avoid having to allocate additional GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxLossForwardGPU<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS)(nthreads, prob_data, label, loss_data,
        num, dim, spatial_dim, has_ignore_label_, ignore_label_, counts);
    Dtype loss;
    caffe_gpu_asum(nthreads, loss_data, &loss);
    if (normalize_) {
      Dtype count;
      caffe_gpu_asum(nthreads, counts, &count);
      loss /= count;
    } else {
      loss /= num;
    }
    top[0]->mutable_cpu_data()[0] = loss;
    if (top.size() == 2) {
      top[1]->ShareData(prob_);
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_context_->id());
    viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(
        this->device_context_->id());

    cl_mem prob_data = (cl_mem) (prob_.gpu_data());
    cl_mem label = (cl_mem) (bottom[1]->gpu_data());
    const int num = prob_.num();
    const int dim = prob_.count() / num;
    const int spatial_dim = prob_.height() * prob_.width();
    const int nthreads = num * spatial_dim;
    cl_mem loss_data = (cl_mem) (bottom[0]->mutable_gpu_diff());
    cl_mem counts = (cl_mem) (prob_.mutable_gpu_diff());

    viennacl::ocl::kernel &oclk_softmax_loss_forward = program.get_kernel(
        CL_KERNEL_SELECT("softmax_loss_forward"));
    viennacl::ocl::enqueue(
        oclk_softmax_loss_forward(nthreads, WrapHandle(prob_data, &ctx),
                                  WrapHandle(label, &ctx),
                                  WrapHandle(loss_data, &ctx), num, dim,
                                  spatial_dim, has_ignore_label_ ? 1 : 0,
                                  ignore_label_, WrapHandle(counts, &ctx)),
        ctx.get_queue());

    Dtype loss;

    greentea_gpu_asum(this->device_context_->id(), nthreads, loss_data, 0,
                      &loss);
    if (normalize_) {
      Dtype count;
      greentea_gpu_asum(this->device_context_->id(), nthreads, counts, 0,
                        &count);
      loss /= count;
    } else {
      loss /= num;
    }
    top[0]->mutable_cpu_data()[0] = loss;
    if (top.size() == 2) {
      top[1]->ShareData(prob_);
    }
#endif  // USE_GREENTEA
  }
}

#ifdef USE_CUDA
template<typename Dtype>
__global__ void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
                                       const Dtype* label, Dtype* bottom_diff,
                                       const int num, const int dim,
                                       const int spatial_dim,
                                       const bool has_ignore_label_,
                                       const int ignore_label_, Dtype* counts) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
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
    LOG(FATAL)<< this->type()
    << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    if (this->device_context_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      const Dtype* prob_data = prob_.gpu_data();
      const Dtype* top_data = top[0]->gpu_data();
      caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
      const Dtype* label = bottom[1]->gpu_data();
      const int num = prob_.num();
      const int dim = prob_.count() / num;
      const int spatial_dim = prob_.height() * prob_.width();
      const int nthreads = num * spatial_dim;
      // Since this memory is never used for anything else,
      // we use to to avoid allocating new GPU memory.
      Dtype* counts = prob_.mutable_gpu_diff();
      // NOLINT_NEXT_LINE(whitespace/operators)
      SoftmaxLossBackwardGPU<Dtype>CUDA_KERNEL(CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS)(nthreads, top_data, label, bottom_diff,
          num, dim, spatial_dim, has_ignore_label_, ignore_label_, counts);
      const Dtype loss_weight = top[0]->cpu_diff()[0];
      if (normalize_) {
        Dtype count;
        caffe_gpu_asum(nthreads, counts, &count);
        caffe_gpu_scal(prob_.count(), loss_weight / count, bottom_diff);
      } else {
        caffe_gpu_scal(prob_.count(), loss_weight / num, bottom_diff);
      }
#endif  // USE_CUDA
    } else {
#ifdef USE_GREENTEA
      viennacl::ocl::context &ctx = viennacl::ocl::get_context(
          this->device_context_->id());
      viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(
          this->device_context_->id());

      cl_mem bottom_diff = (cl_mem)(bottom[0]->mutable_gpu_diff());
      cl_mem prob_data = (cl_mem)(prob_.gpu_data());
      cl_mem top_data = (cl_mem)(top[0]->gpu_data());
      greentea_gpu_memcpy(prob_.count() * sizeof(Dtype),
                          prob_data, 0, bottom_diff, 0, &ctx);
      cl_mem label = (cl_mem)(bottom[1]->gpu_data());
      const int num = prob_.num();
      const int dim = prob_.count() / num;
      const int spatial_dim = prob_.height() * prob_.width();
      const int nthreads = num * spatial_dim;
      cl_mem counts = (cl_mem)(prob_.mutable_gpu_diff());

      viennacl::ocl::kernel &oclk_softmax_loss_backward = program.get_kernel(
          CL_KERNEL_SELECT("softmax_loss_backward"));
      viennacl::ocl::enqueue(
          oclk_softmax_loss_backward(nthreads, WrapHandle(top_data, &ctx),
                    WrapHandle(label, &ctx), WrapHandle(bottom_diff, &ctx),
                    num, dim, spatial_dim, has_ignore_label_ ? 1 : 0,
                    ignore_label_, WrapHandle(counts, &ctx)),
          ctx.get_queue());

      const Dtype loss_weight = top[0]->cpu_diff()[0];
      if (normalize_) {
        Dtype count;
        greentea_gpu_asum(this->device_context_->id(),
                          nthreads, counts, 0, &count);
        greentea_gpu_scal(this->device_context_->id(),
                          prob_.count(), loss_weight / count, bottom_diff, 0);
      } else {
        greentea_gpu_scal(this->device_context_->id(),
                          prob_.count(), loss_weight / num, bottom_diff, 0);
      }
#endif  // USE_GREENTEA
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossLayer);

}  // namespace caffe
