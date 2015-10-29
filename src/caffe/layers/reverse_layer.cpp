#include <utility>
#include <vector>

#include "caffe/common_layers.hpp"

namespace caffe {

template<typename Dtype>
void ReverseLayer<Dtype>::reverse_offsets(
        const vector<pair<int, int> >& reverse_segment) {
  int* offset_data = reverse_offset_.mutable_cpu_data();
  int reverse_idx = 0;
  for (int s = 0; s < reverse_segment.size(); ++s) {
    const int length = reverse_segment[s].second;
    const int start = reverse_segment[s].first;
    const int end = start + length - 1;
    int offset_front = reverse_unit_size_ * start;
    int offset_back = reverse_unit_size_ * end;
    for (int i = start; i < start + length/2; ++i) {
      offset_data[reverse_idx++] = offset_front;
      offset_data[reverse_idx++] = offset_back;
      offset_front += reverse_unit_size_;
      offset_back -= reverse_unit_size_;
    }
  }
}

template <typename Dtype>
void ReverseLayer<Dtype>::reverse(const Dtype* const src, Dtype* const dst) {
  const int* offset_data = reverse_offset_.cpu_data();
  for (int i = 0; i < num_reverse_pairs_; ++i) {
    const int offset_front = offset_data[i*2];
    const int offset_back = offset_data[i*2+1];
    caffe_copy(reverse_unit_size_, src + offset_front,
        dst + offset_back);
    caffe_copy(reverse_unit_size_, src + offset_back,
        dst + offset_front);
  }
}

template <typename Dtype>
void ReverseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  temp_.ReshapeLike(*bottom[0]);
  reverse_unit_size_ = bottom[0]->count(1, bottom[0]->num_axes());
  const int N = bottom[0]->shape(0);
  reverse_offset_.Reshape(N, 1, 1, 1);

  if (bottom.size() == 1) {
    vector<pair<int, int> > reverse_segment;
    reverse_segment.push_back(make_pair(0, N));
    this->reverse_offsets(reverse_segment);
    num_reverse_pairs_ = N/2;
  } else  {
    for (int i = 1; i < bottom[1]->num_axes(); ++i) {
      CHECK_EQ(bottom[1]->shape(i), 1)
          << "bottom[1]->shape(i) must be 1 for i > 0";
    }
    vector<pair<int, int> > reverse_segment;
    num_reverse_pairs_ = 0;
    for (int i = 0; i < N/2; ++i) {
      const int start = static_cast<int>(bottom[1]->cpu_data()[i*2]);
      const int length = static_cast<int>(bottom[1]->cpu_data()[i*2+1]);
      // negative values are used to indicate there is no more reverse segment
      if (start < 0 || length < 0)
        break;
      CHECK_LE(start+length, N)
          << "start, length: " << start << ", " << length
          << ": reverse segment exceeds maximum input index";
      reverse_segment.push_back(make_pair(start, length));
      num_reverse_pairs_ += length/2;
    }
    CHECK_LE(num_reverse_pairs_*2, N)
        << "total number of reverse examples exceeds total number of inputs";
    this->reverse_offsets(reverse_segment);
  }
}

template <typename Dtype>
void ReverseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  temp_.CopyFrom(*bottom[0]);
  top[0]->CopyFrom(*bottom[0]);
  this->reverse(temp_.cpu_data(), top[0]->mutable_cpu_data());
}

template <typename Dtype>
void ReverseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (bottom.size() > 1) {
    CHECK(!propagate_down[1])
        << "Cannot backpropagate to reverse segment Blob";
  }
  temp_.CopyFrom(*top[0], true);
  bottom[0]->CopyFrom(*top[0], true);
  this->reverse(temp_.cpu_diff(), bottom[0]->mutable_cpu_diff());
}

#ifdef CPU_ONLY
STUB_GPU(ReverseLayer);
#endif

INSTANTIATE_CLASS(ReverseLayer);
REGISTER_LAYER_CLASS(Reverse);

}  // namespace caffe
