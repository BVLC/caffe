#include <vector>

#include "caffe/layers/silence_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#endif

namespace caffe {

template<typename Dtype>
void SilenceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  // Do nothing.
}

template<typename Dtype>
void SilenceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom) {
  for (int_tp i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      caffe_gpu_set(bottom[i]->count(), Dtype(0),
                    bottom[i]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SilenceLayer);

}  // namespace caffe
