#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/util/rank_element.hpp"

namespace caffe {

using std::vector;
using std::fill;
using std::pair;
using std::make_pair;
using std::greater;
using std::partial_sort;
using std::lexicographical_compare;

template <typename Dtype>
void DominantChannelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.dominant_channel_param().top_k();
  CHECK_GE(top_k_, 1) << " top k must not be less than 1.";
}

template <typename Dtype>
void DominantChannelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_LT(top_k_, channels_)
      << "top_k must be less than the number of channels.";
  top[0]->Reshape(num_, top_k_, 1, 1);
}

template <typename T, class Compare>
struct LexicographicalVectorCompare {
 public:
  bool operator() (const vector<T>& v1, const vector<T>& v2) const {
    Compare comp;
    return lexicographical_compare(v1.begin(), v1.end(),
				   v2.begin(), v2.end(), comp);
  }
};

template <typename Dtype>
void DominantChannelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int channel_offset = height_ * width_;
  for (int n = 0; n < num_; ++n) {
    // histogram capturing rank statistics, in particular how
    // many times a particular channel is 1st, 2nd, ..., top_k-th
    vector<vector<int> > histo(channels_);
    for (int c = 0; c < channels_; ++c) {
      histo[c].resize(top_k_, 0);
    }
    // accumulate rank statistics over image positions
    vector<int> rank(channels_);
    for (int h = 0; h < height_; ++h) {
      for (int w = 0; w < width_; ++w) {
	const Dtype* bottom_data = bottom[0]->cpu_data(n, 0, h, w);
	vector<Dtype> values(channels_);
	for (int c = 0; c < channels_; ++c) {
	  values[c] = bottom_data[c * channel_offset];
	}
	greater<Dtype> comp;
	partial_rank_element(rank, values, top_k_, comp);
	for (int j = 0; j < top_k_; ++j) {
	  ++histo[rank[j]][j];
	}
      }
    }
    //
    LexicographicalVectorCompare<int, greater<int> > comp;
    rank_element(rank, histo, comp);
    Dtype* top_data = top[0]->mutable_cpu_data(n);
    for (int j = 0; j < top_k_; ++j) {
      top_data[j] = rank[j];
    }
  }
}

template <typename Dtype>
void DominantChannelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_CLASS(DominantChannelLayer);

}  // namespace caffe
