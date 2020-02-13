#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/cyclic_roll_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void CyclicRollLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
  CHECK_EQ(bottom[0]->height(), bottom[0]->width()) <<
    "feature maps must be square";
  CHECK_EQ(bottom[0]->num()%4, 0) <<
    "number of batches must can be divided by 4";
}

template <typename Dtype>
void CyclicRollLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> shape = bottom[0]->shape();
  shape[1] *= 4;
  top[0]->Reshape(shape);
}

template <typename Dtype>
void CyclicRollLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->shape(0);
  const int bottom_channels = bottom[0]->shape(1);
  const int top_channels = top[0]->shape(1);
  const int size = bottom[0]->shape(2);
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int bottom_batch_rotid, rotation, top_channel_id,
    bottom_index, top_index, top_batch_id;
  for (int i = 0; i < num; ++i) {
    bottom_batch_rotid = i%4;
    for (int bottom_channel_id = 0; bottom_channel_id < bottom_channels;
      ++bottom_channel_id) {
      for (int top_batch_rotid = 0; top_batch_rotid < 4; ++top_batch_rotid) {
        rotation = (bottom_batch_rotid-top_batch_rotid)%4;
        rotation = (rotation >= 0)?rotation:(4 + rotation);
        top_channel_id = 4*bottom_channel_id + rotation;
        top_batch_id = i - bottom_batch_rotid + top_batch_rotid;
        for (int h = 0; h < size; ++h) {
          for (int w = 0; w < size; ++w) {
            bottom_index =
              ((i*bottom_channels+bottom_channel_id)*size+h)*size+w;
            switch (rotation) {
              case 0: top_index =
                ((top_batch_id*top_channels+top_channel_id)*size+h)*size+w;
              break;
              case 1: top_index =
                ((top_batch_id*top_channels+top_channel_id)*size+size-1-w)*
                size+h;
              break;
              case 2: top_index =
                ((top_batch_id*top_channels+top_channel_id)*size+size-1-h)*
                size+size-1-w;
              break;
              case 3: top_index =
                ((top_batch_id*top_channels+top_channel_id)*size+w)*
                size+size-1-h;
              break;
              default: top_index = 0;
              CHECK(0) << "rotation not supported";
              break;
            }
            top_data[top_index] = bottom_data[bottom_index];
          }
        }
      }
    }
  }
}

template <typename Dtype>
void CyclicRollLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const int num = bottom[0]->shape(0);
  const int bottom_channels = bottom[0]->shape(1);
  const int top_channels = top[0]->shape(1);
  const int size = bottom[0]->shape(2);
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int bottom_batch_rotid, rotation, top_channel_id,
    bottom_index, top_index, top_batch_id;
  // clear bottom diff
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  for (int i = 0; i < num; ++i) {
    bottom_batch_rotid = i%4;
    for (int bottom_channel_id = 0; bottom_channel_id < bottom_channels;
      ++bottom_channel_id) {
      for (int top_batch_rotid = 0; top_batch_rotid < 4; ++top_batch_rotid) {
        rotation = (bottom_batch_rotid-top_batch_rotid)%4;
        rotation = (rotation >= 0)?rotation:(4 + rotation);
        top_channel_id = 4*bottom_channel_id+ rotation;
        top_batch_id = i - bottom_batch_rotid + top_batch_rotid;
        for (int h = 0; h < size; ++h) {
          for (int w = 0; w < size; ++w) {
            bottom_index =
              ((i*bottom_channels+bottom_channel_id)*size+h)*size+w;
            switch (rotation) {
              case 0: top_index =
                ((top_batch_id*top_channels+top_channel_id)*size+h)*size+w;
              break;
              case 1: top_index =
                ((top_batch_id*top_channels+top_channel_id)*size+size-1-w)*
                size+h;
              break;
              case 2: top_index =
                ((top_batch_id*top_channels+top_channel_id)*size+size-1-h)*
                size+size-1-w;
              break;
              case 3: top_index =
                ((top_batch_id*top_channels+top_channel_id)*size+w)*
                size+size-1-h;
              break;
              default: top_index = 0;
              CHECK(0) << "rotation not supported";
              break;
            }
            bottom_diff[bottom_index] += top_diff[top_index];
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CyclicRollLayer);
#endif

INSTANTIATE_CLASS(CyclicRollLayer);
REGISTER_LAYER_CLASS(CyclicRoll);

}  // namespace caffe
