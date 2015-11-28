#include <vector>

#include "caffe/common_layers.hpp"
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
#include "caffe/layer.hpp"
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SilenceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      caffe_set(bottom[i]->count(), Dtype(0),
<<<<<<< HEAD
<<<<<<< HEAD
                bottom[i]->mutable_cpu_diff());
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
                bottom[i]->mutable_cpu_diff());
=======
                bottom[i]->mutable_cpu_data());
>>>>>>> origin/BVLC/parallel
=======
                bottom[i]->mutable_cpu_diff());
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SilenceLayer);
#endif

INSTANTIATE_CLASS(SilenceLayer);
<<<<<<< HEAD
<<<<<<< HEAD
REGISTER_LAYER_CLASS(Silence);

=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
REGISTER_LAYER_CLASS(Silence);

=======
REGISTER_LAYER_CLASS(SILENCE, SilenceLayer);
>>>>>>> origin/BVLC/parallel
=======
REGISTER_LAYER_CLASS(Silence);

>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
}  // namespace caffe
