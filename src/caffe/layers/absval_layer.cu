#include <vector>

#include "caffe/layers/absval_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

template<typename Dtype>
void AbsValLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  const int_tp count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();
  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    caffe_gpu_abs(count, bottom[0]->gpu_data(), top_data);
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    greentea_gpu_abs<Dtype>(this->device_->id(), count,
                            (cl_mem) (bottom[0]->gpu_data()), 0,
                            (cl_mem) (top_data), 0);
#endif  // USE_GREENTEA
  }
}

template<typename Dtype>
void AbsValLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<Dtype>*>& bottom) {
  const int_tp count = top[0]->count();
  const Dtype* top_diff = top[0]->gpu_diff();
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
      caffe_gpu_sign(count, bottom_data, bottom_diff);
      caffe_gpu_mul(count, bottom_diff, top_diff, bottom_diff);
#endif  // USE_CUDA
    } else {
#ifdef USE_GREENTEA
      greentea_gpu_sign<Dtype>(this->device_->id(), count,
                               (cl_mem) bottom_data, 0, (cl_mem) bottom_diff,
                               0);
      greentea_gpu_mul<Dtype>(this->device_->id(), count,
                              (cl_mem) bottom_diff, 0, (cl_mem) top_diff, 0,
                              (cl_mem) bottom_diff, 0);
#endif  // USE_GREENTEA
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AbsValLayer);

}  // namespace caffe
