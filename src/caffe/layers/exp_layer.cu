#include <vector>

#include "caffe/layers/exp_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void ExpLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  const int_tp count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    if (inner_scale_ == Dtype(1)) {
      caffe_gpu_exp(count, bottom_data, top_data);
    } else {
      caffe_gpu_scale(count, inner_scale_, bottom_data, top_data);
      caffe_gpu_exp(count, top_data, top_data);
    }
    if (outer_scale_ != Dtype(1)) {
      caffe_gpu_scal(count, outer_scale_, top_data);
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    if (inner_scale_ == Dtype(1)) {
      greentea_gpu_exp<Dtype>(this->device_->id(), count,
                              (cl_mem) bottom_data, 0, (cl_mem) top_data, 0);
    } else {
      greentea_gpu_scale<Dtype>(this->device_->id(),
                                count, inner_scale_,
                                (cl_mem) bottom_data, 0, (cl_mem) top_data, 0);
      greentea_gpu_exp<Dtype>(this->device_->id(), count,
                              (cl_mem) top_data, 0, (cl_mem) top_data, 0);
    }
    if (outer_scale_ != Dtype(1)) {
      greentea_gpu_scal<Dtype>(this->device_->id(),
                               count, outer_scale_,
                               (cl_mem) top_data, 0);
    }
#endif  // USE_GREENTEA
  }
}

template<typename Dtype>
void ExpLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                   const vector<bool>& propagate_down,
                                   const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const int_tp count = bottom[0]->count();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    caffe_gpu_mul(count, top_data, top_diff, bottom_diff);
    if (inner_scale_ != Dtype(1)) {
      caffe_gpu_scal(count, inner_scale_, bottom_diff);
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    greentea_gpu_mul<Dtype>(this->device_->id(), count,
                            (cl_mem) top_data, 0, (cl_mem) top_diff, 0,
                            (cl_mem) bottom_diff, 0);
    if (inner_scale_ != Dtype(1)) {
      greentea_gpu_scal<Dtype>(this->device_->id(), count, inner_scale_,
                               (cl_mem) bottom_diff, 0);
    }
#endif  // USE_GREENTEA
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ExpLayer);

}  // namespace caffe
