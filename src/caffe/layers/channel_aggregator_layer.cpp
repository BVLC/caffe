#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <limits>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::string;
using std::vector;

template <typename Dtype>
void ChannelAggregatorLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ChannelAggregatorParameter param = this->layer_param_.channel_aggregator_param();
  CHECK(param.has_label_map_file()) << "Ensure that we have a label map file";
  // Read the file with filenames and labels
  const string& label_map_file = param.label_map_file();
  LOG(INFO) << "Opening file " << label_map_file;
  std::ifstream infile(label_map_file.c_str());
  string linestr;
  while (std::getline(infile, linestr)) {
    std::istringstream iss(linestr);
    vector<int> labels;
    int label;
    while (iss >> label) {
      labels.push_back(label);
    }
    CHECK_GT(labels.size(), 0);
    label_map_.push_back(labels);
  }
  channels_out_ = label_map_.size();
  CHECK_GT(channels_out_, 0) << "label_map cannot be empty";
}

template <typename Dtype>
void ChannelAggregatorLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_in_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  // Check that label map is valid
  for (int i = 0; i < channels_out_; ++i) {
    for (int j = 0; j < label_map_[i].size(); ++j) {
      CHECK(label_map_[i][j] >= 0 && label_map_[i][j] < channels_in_)
	<< "Label map points to out-of-range elements";
    }
  }
  //
  top[0]->Reshape(num_, channels_out_, height_, width_);
}

template <typename Dtype>
void ChannelAggregatorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int spatial_dim = height_ * width_;
  caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
  for (int n = 0; n < num_; ++n) {
    for (int h = 0; h < height_; ++h) {
      for (int w = 0; w < width_; ++w) {
	const Dtype *bottom_data = bottom[0]->cpu_data(n, 0, h, w);
	for (int c_out = 0; c_out < channels_out_; ++c_out) {
	  Dtype *top_data = top[0]->mutable_cpu_data(n, c_out, h, w);
	  // Find the max value across the input channels
	  Dtype max_val = - std::numeric_limits<Dtype>::max();
	  for (vector<int>::const_iterator iter = label_map_[c_out].begin(); iter != label_map_[c_out].end(); ++iter) {
	    const int c_in = *iter;
	    const Dtype cur_val = bottom_data[c_in * spatial_dim];
	    if (iter == label_map_[c_out].begin() || max_val < cur_val) {
	      max_val = cur_val;
	    }
	  }
	  // Compute the output as softmax of input
	  for (vector<int>::const_iterator iter = label_map_[c_out].begin(); iter != label_map_[c_out].end(); ++iter) {
	    const int c_in = *iter;
	    *top_data += exp(bottom_data[c_in * spatial_dim] - max_val);
	  }
	  *top_data = max_val + log(*top_data);
	}
      }
    }
  }
}

template <typename Dtype>
void ChannelAggregatorLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const int spatial_dim = height_ * width_;
  caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
  for (int n = 0; n < num_; ++n) {
    for (int h = 0; h < height_; ++h) {
      for (int w = 0; w < width_; ++w) {
	const Dtype *bottom_data = bottom[0]->cpu_data(n, 0, h, w);
	Dtype *bottom_diff = bottom[0]->mutable_cpu_diff(n, 0, h, w);
	for (int c_out = 0; c_out < channels_out_; ++c_out) {
	  const Dtype *top_diff = top[0]->mutable_cpu_diff(n, c_out, h, w);
	  // Find the max value across the input channels
	  Dtype max_val = - std::numeric_limits<Dtype>::max();
	  for (vector<int>::const_iterator iter = label_map_[c_out].begin(); iter != label_map_[c_out].end(); ++iter) {
	    const int c_in = *iter;
	    const Dtype cur_val = bottom_data[c_in * spatial_dim];
	    if (iter == label_map_[c_out].begin() || max_val < cur_val) {
	      max_val = cur_val;
	    }
	  }
	  // Compute the normalizing constant
	  Dtype Z = 0;
	  for (vector<int>::const_iterator iter = label_map_[c_out].begin(); iter != label_map_[c_out].end(); ++iter) {
	    const int c_in = *iter;
	    Z += exp(bottom_data[c_in * spatial_dim] - max_val);
	  }
	  // Now propagate the gradient back to the input
	  for (vector<int>::const_iterator iter = label_map_[c_out].begin(); iter != label_map_[c_out].end(); ++iter) {
	    const int c_in = *iter;
	    bottom_diff[c_in * spatial_dim] += top_diff[0] * exp(bottom_data[c_in * spatial_dim] - max_val) / Z;
	  }
	}
      }
    }
  }
}

INSTANTIATE_CLASS(ChannelAggregatorLayer);

}  // namespace caffe
