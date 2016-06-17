#include <algorithm>
#include <vector>

#include "caffe/layers/contrastive_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

template<typename Dtype>
void ContrastiveLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const bool legacy_version = this->layer_param_.contrastive_loss_param()
      .legacy_version();

  const int_tp count = bottom[0]->count();

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    caffe_gpu_sub(count, bottom[0]->gpu_data(),  // a
                  bottom[1]->gpu_data(),  // b
                  diff_.mutable_gpu_data());  // a_i-b_i
    caffe_gpu_powx(count, diff_.mutable_gpu_data(),  // a_i-b_i
                   Dtype(2), diff_sq_.mutable_gpu_data());  // (a_i-b_i)^2
    caffe_gpu_gemv(CblasNoTrans, bottom[0]->num(), bottom[0]->channels(),
                   Dtype(1.0),
                   diff_sq_.gpu_data(),  // (a_i-b_i)^2
                   summer_vec_.gpu_data(), Dtype(0.0),
                   dist_sq_.mutable_gpu_data());  // \Sum (a_i-b_i)^2
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    greentea_gpu_sub<Dtype>(this->device_->id(), count,
                            (cl_mem) (bottom[0]->gpu_data()), 0,
                            (cl_mem) (bottom[1]->gpu_data()), 0,
                            (cl_mem) (diff_.mutable_gpu_data()), 0);
    greentea_gpu_powx<Dtype>(this->device_->id(), count,
                             (cl_mem) (diff_.mutable_gpu_data()),
                             0,  // a_i-b_i
                             Dtype(2), (cl_mem) (diff_sq_.mutable_gpu_data()),
                             0);  // (a_i-b_i)^2
    greentea_gpu_gemv<Dtype>(this->device_->id(), CblasNoTrans,
                             bottom[0]->num(), bottom[0]->channels(),
                             Dtype(1.0), (cl_mem) (diff_sq_.gpu_data()),
                             0,  // (a_i-b_i)^2
                             (cl_mem) (summer_vec_.gpu_data()), 0, Dtype(0.0),
                             (cl_mem) (dist_sq_.mutable_gpu_data()), 0);
#endif  // USE_GREENTEA
  }

  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  Dtype loss(0.0);
  for (int_tp i = 0; i < bottom[0]->num(); ++i) {
    if (static_cast<int_tp>(bottom[2]->cpu_data()[i])) {  // similar pairs
      loss += dist_sq_.cpu_data()[i];
    } else {  // dissimilar pairs
      if (legacy_version) {
        loss += std::max(margin - dist_sq_.cpu_data()[i], Dtype(0.0));
      } else {
        Dtype dist = std::max(margin - (Dtype) sqrt(dist_sq_.cpu_data()[i]),
                              Dtype(0.0));
        loss += dist * dist;
      }
    }
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

#ifdef USE_CUDA
template<typename Dtype>
__global__ void CLLBackward(const int_tp count, const int_tp channels,
                            const Dtype margin, const bool legacy_version,
                            const Dtype alpha, const Dtype* y,
                            const Dtype* diff, const Dtype* dist_sq,
                            Dtype *bottom_diff) {
  CUDA_KERNEL_LOOP(i, count) {
    int_tp n = i / channels;  // the num index, to access y and dist_sq
    if (static_cast<int_tp>(y[n])) {  // similar pairs
      bottom_diff[i] = alpha * diff[i];
    } else {  // dissimilar pairs
      Dtype mdist(0.0);
      Dtype beta(0.0);
      if (legacy_version) {
        mdist = (margin - dist_sq[n]);
        beta = -alpha;
      } else {
        Dtype dist = sqrt(dist_sq[n]);
        mdist = (margin - dist);
        beta = -alpha * mdist / (dist + Dtype(1e-4)) * diff[i];
      }
      if (mdist > 0.0) {
        bottom_diff[i] = beta;
      } else {
        bottom_diff[i] = 0;
      }
    }
  }
}
#endif  // USE_CUDA

template<typename Dtype>
void ContrastiveLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const bool legacy_version = this->layer_param_.contrastive_loss_param()
      .legacy_version();

  for (int_tp i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const int_tp count = bottom[0]->count();
      const int_tp channels = bottom[0]->channels();
      Dtype margin = this->layer_param_.contrastive_loss_param().margin();
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0]
          / static_cast<Dtype>(bottom[0]->num());

      if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
        // NOLINT_NEXT_LINE(whitespace/operators)
        CLLBackward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
                                       CAFFE_CUDA_NUM_THREADS)(
            count, channels, margin, legacy_version, alpha,
            bottom[2]->gpu_data(),  // pair similarity 0 or 1
            diff_.gpu_data(),  // the cached eltwise difference between a and b
            dist_sq_.gpu_data(),  // the cached square distance between a and b
            bottom[i]->mutable_gpu_diff());
        CUDA_POST_KERNEL_CHECK;
#endif  // USE_CUDA
      } else {
#ifdef USE_GREENTEA
        viennacl::ocl::context &ctx = viennacl::ocl::get_context(
            this->device_->id());
        viennacl::ocl::program &program = this->device_->program();
        viennacl::ocl::kernel &oclk_cll = program.get_kernel(
            legacy_version ? CL_KERNEL_SELECT("cll_backward_legacy") :
                CL_KERNEL_SELECT("cll_backward"));
        viennacl::ocl::enqueue(
            oclk_cll(
                count, channels, margin, alpha,
                WrapHandle((cl_mem) (bottom[2]->gpu_data()), &ctx),
                WrapHandle((cl_mem) (diff_.gpu_data()), &ctx),
                WrapHandle((cl_mem) (dist_sq_.gpu_data()), &ctx),
                WrapHandle((cl_mem) (bottom[i]->mutable_gpu_diff()), &ctx)),
            ctx.get_queue());

#endif  // USE_GREENTEA
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ContrastiveLossLayer);

}  // namespace caffe
