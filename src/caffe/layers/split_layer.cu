#include <vector>

#include "caffe/layers/split_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void SplitLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  for (int_tp i = 0; i < top.size(); ++i) {
    top[i]->ShareData(*bottom[0]);
  }
}

template<typename Dtype>
void SplitLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                     const vector<bool>& propagate_down,
                                     const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    if (top.size() == 1) {
      caffe_copy(count_, top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
      return;
    }
    caffe_gpu_add(count_, top[0]->gpu_diff(), top[1]->gpu_diff(),
                  bottom[0]->mutable_gpu_diff());
    // Add remaining top blob diffs.
    for (int_tp i = 2; i < top.size(); ++i) {
      const Dtype* top_diff = top[i]->gpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      caffe_gpu_axpy(count_, Dtype(1.), top_diff, bottom_diff);
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());

    if (top.size() == 1) {
      greentea_copy<Dtype>(count_, (cl_mem) (top[0]->gpu_diff()), 0,
                    (cl_mem) (bottom[0]->mutable_gpu_diff()), 0, &ctx);
      return;
    }
    greentea_gpu_add<Dtype>(this->device_->id(), count_,
                     (cl_mem) (top[0]->gpu_diff()), 0,
                     (cl_mem) (top[1]->gpu_diff()), 0,
                     (cl_mem) (bottom[0]->mutable_gpu_diff()), 0);
    // Add remaining top blob diffs.
    for (int_tp i = 2; i < top.size(); ++i) {
      const Dtype* top_diff = top[i]->gpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      greentea_gpu_axpy<Dtype>(this->device_->id(), count_, Dtype(1.),
                        (cl_mem) top_diff, 0, (cl_mem) bottom_diff, 0);
    }
#endif  // USE_GREENTEA
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SplitLayer);

}  // namespace caffe
