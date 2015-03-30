#include <vector>
#include <algorithm>
#include <limits>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::vector;
using std::sort;
using std::nth_element;
using std::binary_search;
using std::random_shuffle;
using std::numeric_limits;

template <typename Dtype>
void AdaptiveBiasChannelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  AdaptiveBiasChannelParameter param = this->layer_param_.adaptive_bias_channel_param();
  num_iter_ = param.num_iter();
  CHECK_GT(num_iter_, 0);
  suppress_others_ = param.suppress_others();
  margin_others_ = param.margin_others();
  CHECK_GE(margin_others_, 0);
  bg_portion_ = param.bg_portion();
  fg_portion_ = param.fg_portion();
  CHECK(bg_portion_ >= 0 && bg_portion_ <= 1) << "BG portion needs to be in [0, 1]";
  CHECK(fg_portion_ >= 0 && fg_portion_ <= 1) << "FG portion needs to be in [0, 1]";
}

template <typename Dtype>
void AdaptiveBiasChannelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_GT(channels_, 1);
  //
  CHECK_EQ(bottom[1]->num(), num_) << "Input channels incompatible in num";
  max_labels_ = bottom[1]->channels();
  CHECK_GE(max_labels_, 1) << "Label blob needs to be non-empty";
  CHECK_EQ(bottom[1]->height(), 1) << "Label height";
  CHECK_EQ(bottom[1]->width(), 1) << "Label width";
  //
  top[0]->Reshape(num_, channels_, height_, width_);
}

template <typename Dtype>
void AdaptiveBiasChannelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // copy bottom[0] -> top[0]
  caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(), top[0]->mutable_cpu_data());

  for (int n = 0; n < num_; ++n) {

    // Find the labels present in the n-th image
    vector<int> labels;
    for (int j = 0; j < max_labels_; ++j) {
      const int label = static_cast<int>(*bottom[1]->cpu_data(n, j));
      if (label == -1) {
	continue;
      } else if (label >= 0 && label < channels_) {
	labels.push_back(label);
      } else {
	LOG(FATAL) << "Unexpected label " << label;
      }
    }
    // Do nothing if the image has no label
    if (labels.size() == 0) {
      continue;
    }

    // Find if background is present
    sort(labels.begin(), labels.end());
    const bool bgnd_present = binary_search(labels.begin(), labels.end(), 0);

    Dtype *top_data = top[0]->mutable_cpu_data(n);

    // Pointwise maximum score (or minimum score for score suppression)
    Blob<Dtype> extremum(1, 1, height_, width_);
    Dtype *extremum_data = extremum.mutable_cpu_data();
    // Difference between top score and current channel's score
    Blob<Dtype> diff(1, 1, height_, width_);
    Dtype *diff_data = diff.mutable_cpu_data();

    const int spatial_dim = height_ * width_;

    if (suppress_others_) {
      // Find the minimum score of present labels
      caffe_set(spatial_dim, numeric_limits<Dtype>::max(), extremum_data);
      for (int c = 0; c < channels_; ++c) {
	if (binary_search(labels.begin(), labels.end(), c)) {
	  for (int i = 0; i < spatial_dim; ++i) {
	    if (extremum_data[i] > top_data[c * spatial_dim + i]) {
	      extremum_data[i] = top_data[c * spatial_dim + i];
	    }
	  }
	}
      }
      // Clip the score of absent labels
      for (int c = 0; c < channels_; ++c) {
	if (!binary_search(labels.begin(), labels.end(), c)) {
	  for (int i = 0; i < spatial_dim; ++i) {
	    if (top_data[c * spatial_dim + i] > extremum_data[i]) {
	      top_data[c * spatial_dim + i] = extremum_data[i] - margin_others_;
	    }
	  }
	}
      }      
    }

    // Pointwise maximum score of present labels
    caffe_copy(spatial_dim, &top_data[labels[0] * spatial_dim], extremum_data);
    for (int l = 1; l < labels.size(); ++l) {
      const int label = labels[l];
      for (int i = 0; i < spatial_dim; ++i) {
	if (extremum_data[i] < top_data[label * spatial_dim + i]) {
	  extremum_data[i] = top_data[label * spatial_dim + i];
	}
      }
    }
    // Find original mean of maximum score
    double mean_max0 = 0;
    for (int i = 0; i < spatial_dim; ++i) {
      mean_max0 += extremum_data[i];
    }
    mean_max0 /= spatial_dim;

    // Swap over the labels num_iter_ times
    for (int t = 0; t < num_iter_; ++t) {
      // Shuffle the label order (except for the bgnd which is always first)
      random_shuffle(labels.begin() + static_cast<int>(bgnd_present), labels.end());
      for (int l = 0; l < labels.size(); ++l) {
	const int label = labels[l];
	// Find the score difference: (maximum score) - (current class score)
	caffe_sub(spatial_dim, extremum_data, &top_data[label * spatial_dim],
		  diff_data);
	// Find the n-th percentile of score difference
	const int nth = (label == 0) ? bg_portion_ * spatial_dim :
	  fg_portion_ * spatial_dim;
	nth_element(diff_data, diff_data + nth, diff_data + spatial_dim);
	// Add the n-th percentile
	caffe_add_scalar(spatial_dim, diff_data[nth], &top_data[label * spatial_dim]);
	// Update point-wise maximum (note that diff_data[nth] >= 0)
	for (int i = 0; i < spatial_dim; ++i) {
	  if (extremum_data[i] < top_data[label * spatial_dim + i]) {
	    extremum_data[i] = top_data[label * spatial_dim + i];
	  }
	}
      }
    }

    // Find final mean of maximum score
    double mean_max1 = 0;
    for (int i = 0; i < spatial_dim; ++i) {
      mean_max1 += extremum_data[i];
    }
    mean_max1 /= spatial_dim;
    // Subtract a constant from all channels to make sure that the mean
    // (over spatial positions) of the maximum score for current image
    // remains the same with the beginning
    caffe_add_scalar(channels_ * spatial_dim, static_cast<Dtype>(mean_max0 - mean_max1), top_data);

  }
}

template <typename Dtype>
void AdaptiveBiasChannelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
//STUB_GPU(AdaptiveBiasChannelLayer);
#endif

INSTANTIATE_CLASS(AdaptiveBiasChannelLayer);
REGISTER_LAYER_CLASS(ADAPTIVE_BIAS_CHANNEL, AdaptiveBiasChannelLayer);

}  // namespace caffe
