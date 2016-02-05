#include <caffe/layer_factory.hpp>
#include <caffe/layers/mil_layer.hpp>

#include <algorithm>
#include <vector>

#include <cfloat>

using std::max;
using std::min;

namespace caffe {
  template <typename Dtype>
  MILLayer<Dtype>::MILLayer(const LayerParameter& param): Layer<Dtype>(param) {}

  template <typename Dtype>
void MILLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1) << "MIL Layer takes a single blob as input.";
  CHECK_EQ(top.size(), 1) << "MIL Layer takes a single blob as output.";
}
template <typename Dtype>
void MILLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
}

  template <typename Dtype>
  const char* MILLayer<Dtype>::type() const { return "MIL"; }

  template <typename Dtype>
  int MILLayer<Dtype>::ExactNumBottomBlobs() const { return 1; }

  template <typename Dtype>
  int MILLayer<Dtype>::ExactNumTopBlobs() const { return 1; }

  template <typename Dtype>
void MILLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int offset;
  channels_   = bottom[0]->channels();
  num_images_ = bottom[0]->num();
  height_     = bottom[0]->height();
  width_      = bottom[0]->width();

  for (int j = 0; j < channels_; j++) {
    for (int i = 0; i < num_images_; i++) {
      Dtype prob, max_prob;

      switch (this->layer_param_.mil_param().type()) {
        case MILParameter_MILType_MAX:
          prob = -FLT_MAX;
          offset = bottom[0]->offset(i, j);
          for (int k = 0; k < height_; k++) {
            for (int l = 0; l < width_; l++) {
              prob = max(prob, bottom_data[offset]);
              offset = offset + 1;
            }
          }
          top_data[i*channels_ + j] = prob;
          break;

        case MILParameter_MILType_NOR:
          prob = 1.; max_prob = -FLT_MAX;
          offset = bottom[0]->offset(i, j);
          for (int k = 0; k < height_; k++) {
            for (int l = 0; l < width_; l++) {
              assert(0 <= bottom_data[offset] && bottom_data[offset] <= 1);
              prob = prob*(1. - bottom_data[offset]);
              max_prob = max(max_prob, bottom_data[offset]);
              offset = offset + 1;
            }
          }
          top_data[i*channels_ + j] = max(Dtype(1.) - prob, max_prob);
          assert(0 <= top_data[i*channels_ + j] &&
                 top_data[i*channels_ + j] <= 1);
          break;
      }
    }
  }
}

template <typename Dtype>
void MILLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int offset;

  if (propagate_down[0]) {
    for (int j = 0; j < channels_; j++) {
      for (int i = 0; i < num_images_; i++) {
        offset = bottom[0]->offset(i, j);

        for (int k = 0; k < height_; k++) {
          for (int l = 0; l < width_; l++) {
            switch (this->layer_param_.mil_param().type()) {
              case MILParameter_MILType_MAX:
                bottom_diff[offset] =
                  top_diff[i*channels_ + j] * (top_data[i*channels_ + j]
                    == bottom_data[offset]);
                break;
              case MILParameter_MILType_NOR:
                bottom_diff[offset] = top_diff[i * channels_ + j] *
                  min(Dtype(1.), ((1-top_data[i * channels_ + j]) /
                      (1-bottom_data[offset])));
                break;
            }
            offset = offset + 1;
          }
        }
      }
    }
  }
}

INSTANTIATE_CLASS(MILLayer);
REGISTER_LAYER_CLASS(MIL);
}  // namespace caffe
