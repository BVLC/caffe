#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

template<typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();

  if (this->device_context_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    caffe_gpu_sub(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
                  diff_.mutable_gpu_data());
    Dtype dot;
    caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
    Dtype loss = dot / bottom[0]->num() / Dtype(2);
    top[0]->mutable_cpu_data()[0] = loss;
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    greentea_gpu_sub<Dtype>(this->device_context_->id(), count,
                            (cl_mem) (bottom[0]->gpu_data()), 0,
                            (cl_mem) (bottom[1]->gpu_data()), 0,
                            (cl_mem) (diff_.mutable_gpu_data()), 0);
    Dtype dot;
    greentea_gpu_dot<Dtype>(this->device_context_->id(), count,
                            (cl_mem) (diff_.gpu_data()), 0,
                            (cl_mem) (diff_.gpu_data()), 0, &dot);
    Dtype loss = dot / bottom[0]->num() / Dtype(2);
    top[0]->mutable_cpu_data()[0] = loss;
#endif  // USE_GREENTEA
  }
}

template<typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      if (this->device_context_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
        caffe_gpu_axpby(bottom[i]->count(),              // count
            alpha,                              // alpha
            diff_.gpu_data(),                   // a
            Dtype(0),                           // beta
            bottom[i]->mutable_gpu_diff());  // b
#endif  // USE_CUDA
      } else {
#ifdef USE_GREENTEA
        greentea_gpu_axpby(this->device_context_->id(), bottom[i]->count(),
                           alpha, (cl_mem) (diff_.gpu_data()), 0, Dtype(0),
                           (cl_mem) (bottom[i]->mutable_gpu_diff()), 0);
#endif  // USE_GREENTEA
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossLayer);

}  // namespace caffe
