#include <vector>

#include "caffe/layers/reduction_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void ReductionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* mult_data = NULL;

  int_tp bottom_data_off = 0;
  int_tp top_data_off = 0;

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    if (sum_multiplier_.count() > 0) {
      mult_data = sum_multiplier_.gpu_data();
    }
    Dtype* top_data = top[0]->mutable_cpu_data();
    for (int_tp i = 0; i < num_; ++i) {
      switch (op_) {
        case ReductionParameter_ReductionOp_SUM:
        case ReductionParameter_ReductionOp_MEAN:
          caffe_gpu_dot(dim_, mult_data, bottom_data + bottom_data_off,
                        top_data + top_data_off);
          break;
        case ReductionParameter_ReductionOp_ASUM:
          caffe_gpu_asum(dim_, bottom_data + bottom_data_off,
                         top_data + top_data_off);
          break;
        case ReductionParameter_ReductionOp_SUMSQ:
          caffe_gpu_dot(dim_, bottom_data + bottom_data_off,
                        bottom_data + bottom_data_off, top_data + top_data_off);
          break;
        default:
          LOG(FATAL)<< "Unknown reduction op: "
          << ReductionParameter_ReductionOp_Name(op_);
        }
        bottom_data_off += dim_;
        ++top_data_off;
      }
    if (coeff_ != Dtype(1)) {
      // Reset the top_data pointer.
      top_data = top[0]->mutable_gpu_data();
      caffe_gpu_scal(num_, coeff_, top_data);
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    if (sum_multiplier_.count() > 0) {
      mult_data = sum_multiplier_.gpu_data();
    }
    Dtype* top_data = top[0]->mutable_cpu_data();
    for (int_tp i = 0; i < num_; ++i) {
      switch (op_) {
        case ReductionParameter_ReductionOp_SUM:
        case ReductionParameter_ReductionOp_MEAN:
          greentea_gpu_dot<Dtype>(this->device_->id(), dim_,
                                  (cl_mem) mult_data, 0, (cl_mem) bottom_data,
                                  bottom_data_off, top_data + top_data_off);
          break;
        case ReductionParameter_ReductionOp_ASUM:
          greentea_gpu_asum<Dtype>(this->device_->id(), dim_,
                                   (cl_mem) bottom_data, bottom_data_off,
                                   top_data + top_data_off);
          break;
        case ReductionParameter_ReductionOp_SUMSQ:
          greentea_gpu_dot<Dtype>(this->device_->id(), dim_,
                                  (cl_mem) bottom_data, bottom_data_off,
                                  (cl_mem) bottom_data, bottom_data_off,
                                  top_data + top_data_off);
          break;
        default:
          LOG(FATAL)<< "Unknown reduction op: "
          << ReductionParameter_ReductionOp_Name(op_);
        }
        bottom_data_off += dim_;
        ++top_data_off;
      }
    if (coeff_ != Dtype(1)) {
      // Reset the top_data pointer.
      top_data = top[0]->mutable_gpu_data();
      greentea_gpu_scal<Dtype>(this->device_->id(), num_, coeff_,
                               (cl_mem) top_data, 0);
    }
#endif  // USE_GREENTEA
  }
}

template<typename Dtype>
void ReductionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  // Get bottom_data, if needed.
  const Dtype* bottom_data = NULL;
  switch (op_) {
    // Operations that don't need bottom_data
    case ReductionParameter_ReductionOp_SUM:
    case ReductionParameter_ReductionOp_MEAN:
      break;
      // Operations that need bottom_data
    case ReductionParameter_ReductionOp_ASUM:
    case ReductionParameter_ReductionOp_SUMSQ:
      bottom_data = bottom[0]->gpu_data();
      break;
    default:
      LOG(FATAL)<< "Unknown reduction op: "
      << ReductionParameter_ReductionOp_Name(op_);
    }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  int_tp bottom_data_off = 0;
  int_tp bottom_diff_off = 0;
  int_tp top_diff_off = 0;

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    for (int_tp i = 0; i < num_; ++i) {
      const Dtype bottom_coeff = (*(top_diff + top_diff_off)) * coeff_;
      switch (op_) {
        case ReductionParameter_ReductionOp_SUM:
        case ReductionParameter_ReductionOp_MEAN:
          caffe_gpu_set(dim_, bottom_coeff, bottom_diff + bottom_diff_off);
          break;
        case ReductionParameter_ReductionOp_ASUM:
          caffe_gpu_sign(dim_, bottom_data + bottom_data_off,
                         bottom_diff + bottom_diff_off);
          caffe_gpu_scal(dim_, bottom_coeff, bottom_diff + bottom_diff_off);
          break;
        case ReductionParameter_ReductionOp_SUMSQ:
          caffe_gpu_scale(dim_, 2 * bottom_coeff, bottom_data + bottom_data_off,
                          bottom_diff + bottom_diff_off);
          break;
        default:
          LOG(FATAL)<< "Unknown reduction op: "
          << ReductionParameter_ReductionOp_Name(op_);
        }
      bottom_data_off += dim_;
      bottom_diff_off += dim_;
      ++top_diff_off;
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    for (int_tp i = 0; i < num_; ++i) {
      const Dtype bottom_coeff = (*(top_diff + top_diff_off)) * coeff_;
      switch (op_) {
        case ReductionParameter_ReductionOp_SUM:
        case ReductionParameter_ReductionOp_MEAN:
          greentea_gpu_set<Dtype>(this->device_->id(), dim_,
                                  bottom_coeff, (cl_mem) bottom_diff,
                                  bottom_diff_off);
          break;
        case ReductionParameter_ReductionOp_ASUM:
          greentea_gpu_sign<Dtype>(this->device_->id(), dim_,
                                   (cl_mem) bottom_data, bottom_data_off,
                                   (cl_mem) bottom_diff, bottom_diff_off);
          greentea_gpu_scal<Dtype>(this->device_->id(), dim_,
                                   bottom_coeff, (cl_mem) bottom_diff,
                                   bottom_diff_off);
          break;
        case ReductionParameter_ReductionOp_SUMSQ:
          greentea_gpu_scale<Dtype>(this->device_->id(), dim_,
                                    2 * bottom_coeff, (cl_mem) bottom_data,
                                    bottom_data_off, (cl_mem) bottom_diff,
                                    bottom_diff_off);
          break;
        default:
          LOG(FATAL)<< "Unknown reduction op: "
          << ReductionParameter_ReductionOp_Name(op_);
        }
      bottom_data_off += dim_;
      bottom_diff_off += dim_;
      ++top_diff_off;
    }
#endif  // USE_GREENTEA
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ReductionLayer);

}  // namespace caffe
