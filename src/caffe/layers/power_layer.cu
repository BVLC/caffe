#include <vector>

#include "caffe/layers/power_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

template<typename Dtype>
void PowerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int_tp count = bottom[0]->count();

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    // Special case where we can ignore the input: scale or power is 0.
    if (diff_scale_ == Dtype(0)) {
      Dtype value = (power_ == 0) ? Dtype(1) : pow(shift_, power_);
      caffe_gpu_set(count, value, top_data);
      return;
    }
    const Dtype* bottom_data = bottom[0]->gpu_data();
    caffe_copy(count, bottom_data, top_data);
    if (scale_ != Dtype(1)) {
      caffe_gpu_scal(count, scale_, top_data);
    }
    if (shift_ != Dtype(0)) {
      caffe_gpu_add_scalar(count, shift_, top_data);
    }
    if (power_ != Dtype(1)) {
      caffe_gpu_powx(count, top_data, power_, top_data);
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());

    if (diff_scale_ == Dtype(0)) {
      Dtype value = (power_ == 0) ? Dtype(1) : pow(shift_, power_);
      greentea_gpu_set<Dtype>(this->device_->id(), count, value,
                              (cl_mem) top_data, 0);
      return;
    }
    const Dtype* bottom_data = bottom[0]->gpu_data();
    greentea_copy<Dtype>(count, (cl_mem) bottom_data, 0, (cl_mem) top_data, 0,
                         &ctx);
    if (scale_ != Dtype(1)) {
      greentea_gpu_scal(this->device_->id(), count, scale_,
                        (cl_mem) top_data, 0);
    }
    if (shift_ != Dtype(0)) {
      greentea_gpu_add_scalar<Dtype>(this->device_->id(), count, shift_,
                                     (cl_mem) top_data, 0);
    }
    if (power_ != Dtype(1)) {
      greentea_gpu_powx<Dtype>(this->device_->id(), count,
                               (cl_mem) top_data, 0, power_, (cl_mem) top_data,
                               0);
    }
#endif  // USE_GREENTEA
  }
}

template<typename Dtype>
void PowerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                     const vector<bool>& propagate_down,
                                     const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int_tp count = bottom[0]->count();
    const Dtype* top_diff = top[0]->gpu_diff();

    if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
      if (diff_scale_ == Dtype(0) || power_ == Dtype(1)) {
        caffe_gpu_set(count, diff_scale_, bottom_diff);
      } else {
        const Dtype* bottom_data = bottom[0]->gpu_data();
        // Compute dy/dx = scale * power * (shift + scale * x)^(power - 1)
        //               = diff_scale * y / (shift + scale * x)
        if (power_ == Dtype(2)) {
          // Special case for y = (shift + scale * x)^2
          //     -> dy/dx = 2 * scale * (shift + scale * x)
          //              = diff_scale * shift + diff_scale * scale * x
          caffe_gpu_axpby(count, diff_scale_ * scale_, bottom_data, Dtype(0),
                          bottom_diff);
          if (shift_ != Dtype(0)) {
            caffe_gpu_add_scalar(count, diff_scale_ * shift_, bottom_diff);
          }
        } else if (shift_ == Dtype(0)) {
          // Special case for y = (scale * x)^power
          //     -> dy/dx = scale * power * (scale * x)^(power - 1)
          //              = scale * power * (scale * x)^power * (scale * x)^(-1)
          //              = power * y / x
          const Dtype* top_data = top[0]->gpu_data();
          caffe_gpu_div(count, top_data, bottom_data, bottom_diff);
          caffe_gpu_scal(count, power_, bottom_diff);
        } else {
          caffe_copy(count, bottom_data, bottom_diff);
          if (scale_ != Dtype(1)) {
            caffe_gpu_scal(count, scale_, bottom_diff);
          }
          if (shift_ != Dtype(0)) {
            caffe_gpu_add_scalar(count, shift_, bottom_diff);
          }
          const Dtype* top_data = top[0]->gpu_data();
          caffe_gpu_div<Dtype>(count, top_data, bottom_diff, bottom_diff);
          if (diff_scale_ != Dtype(1)) {
            caffe_gpu_scal(count, diff_scale_, bottom_diff);
          }
        }
      }
      caffe_gpu_mul(count, top_diff, bottom_diff, bottom_diff);
#endif  // USE_CUDA
    } else {
#ifdef USE_GREENTEA
      viennacl::ocl::context &ctx = viennacl::ocl::get_context(
          this->device_->id());

      if (diff_scale_ == Dtype(0) || power_ == Dtype(1)) {
        greentea_gpu_set<Dtype>(this->device_->id(), count, diff_scale_,
                                (cl_mem) bottom_diff, 0);
      } else {
        const Dtype* bottom_data = bottom[0]->gpu_data();
        // Compute dy/dx = scale * power * (shift + scale * x)^(power - 1)
        //               = diff_scale * y / (shift + scale * x)
        if (power_ == Dtype(2)) {
          // Special case for y = (shift + scale * x)^2
          //     -> dy/dx = 2 * scale * (shift + scale * x)
          //              = diff_scale * shift + diff_scale * scale * x
          greentea_gpu_axpby(this->device_->id(), count,
                             diff_scale_ * scale_, (cl_mem) bottom_data, 0,
                             Dtype(0), (cl_mem) bottom_diff, 0);
          if (shift_ != Dtype(0)) {
            greentea_gpu_add_scalar(this->device_->id(), count,
                                    diff_scale_ * shift_, (cl_mem) bottom_diff,
                                    0);
          }
        } else if (shift_ == Dtype(0)) {
          // Special case for y = (scale * x)^power
          //     -> dy/dx = scale * power * (scale * x)^(power - 1)
          //              = scale * power * (scale * x)^power * (scale * x)^(-1)
          //              = power * y / x
          const Dtype* top_data = top[0]->gpu_data();
          greentea_gpu_div<Dtype>(this->device_->id(), count,
                                  (cl_mem) top_data, 0, (cl_mem) bottom_data, 0,
                                  (cl_mem) bottom_diff, 0);
          greentea_gpu_scal<Dtype>(this->device_->id(), count, power_,
                                   (cl_mem) bottom_diff, 0);
        } else {
          greentea_copy<Dtype>(count, (cl_mem) bottom_data, 0,
                               (cl_mem) bottom_diff, 0, &ctx);
          if (scale_ != Dtype(1)) {
            greentea_gpu_scal(this->device_->id(), count, scale_,
                              (cl_mem) bottom_diff, 0);
          }
          if (shift_ != Dtype(0)) {
            greentea_gpu_add_scalar(this->device_->id(), count, shift_,
                                    (cl_mem) bottom_diff, 0);
          }
          const Dtype* top_data = top[0]->gpu_data();
          greentea_gpu_div<Dtype>(this->device_->id(), count,
                                  (cl_mem) top_data, 0, (cl_mem) bottom_diff, 0,
                                  (cl_mem) bottom_diff, 0);
          if (diff_scale_ != Dtype(1)) {
            greentea_gpu_scal(this->device_->id(), count, diff_scale_,
                              (cl_mem) bottom_diff, 0);
          }
        }
      }
      greentea_gpu_mul<Dtype>(this->device_->id(), count,
                              (cl_mem) top_diff, 0, (cl_mem) bottom_diff, 0,
                              (cl_mem) bottom_diff, 0);
#endif  // USE_GREENTEA
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PowerLayer);

}  // namespace caffe
