#include "caffe/layers/mtcnn_bbox_layer.hpp"

namespace caffe {

static float fix(float x) { return static_cast<int>(x + 1e-4); }

template <typename Dtype>
void MTCNNBBoxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
  Forward_cpu_const(bottom, top);
}

template <typename Dtype>
void MTCNNBBoxLayer<Dtype>::Forward_cpu_const(
    const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) const {
  const auto bbox_reg = bottom[0]->cpu_data();
  const auto &shape = bottom[1]->shape();
  const auto prob = bottom[1]->cpu_data() + shape[2] * shape[3];
  const auto scale = bottom[2]->cpu_data()[0];

  int cnt = 0;
  for (int h = 0; h < shape[2]; h++) {
    for (int w = 0; w < shape[3]; w++) {
      int idx = h * shape[3] + w;
      auto map = prob[idx];
      if (map >= threshold_) {
	cnt++;
      }
    }
  }
  if (cnt == 0) {
    return;
  }

  top[0]->Reshape(1, 1, cnt, 9);
  auto top_data = top[0]->mutable_cpu_data();

  for (int h = 0; h < shape[2]; h++) {
    for (int w = 0; w < shape[3]; w++) {
      int idx = h * shape[3] + w;
      auto map = prob[idx];
      if (map >= threshold_) {
        *top_data++ = (fix((stride_ * h + 1) / scale - 1));
        *top_data++ = (fix((stride_ * w + 1) / scale - 1));
        *top_data++ = (fix((stride_ * h + cellsize_) / scale - 1));
        *top_data++ = (fix((stride_ * w + cellsize_) / scale - 1));
        *top_data++ = map;
        *top_data++ = (bbox_reg[0 * shape[2] * shape[3] + idx]);
        *top_data++ = (bbox_reg[1 * shape[2] * shape[3] + idx]);
        *top_data++ = (bbox_reg[2 * shape[2] * shape[3] + idx]);
        *top_data++ = (bbox_reg[3 * shape[2] * shape[3] + idx]);
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MTCNNBBoxLayer);
STUB_GPU_FORWARD_CONST(MTCNNBBoxLayer,Forward_const);
#endif

INSTANTIATE_CLASS(MTCNNBBoxLayer);
REGISTER_LAYER_CLASS(MTCNNBBox);

} // namespace caffe
