#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/hebbian_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void HebbianPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) {
		return;
	}
	const Dtype* top_data = top[0]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	// Different pooling methods. We explicitly do the switch outside the for
	// loop to save time, although this results in more codes.
	caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
	// We'll output the mask to top[1] if it's of size >1.
	const bool use_top_mask = top.size() > 1;
	const int* mask = NULL;  // suppress warnings about uninitialized variables
	const Dtype* top_mask = NULL;
	switch (this->layer_param_.pooling_param().pool()) {
	case PoolingParameter_PoolMethod_MAX:
		// The main loop
		if (use_top_mask) {
			top_mask = top[1]->cpu_data();
		}
		else {
			mask = max_idx_.cpu_data();
		}
		for (int n = 0; n < top[0]->num(); ++n) {
			for (int c = 0; c < channels_; ++c) {
				for (int ph = 0; ph < pooled_height_; ++ph) {
					for (int pw = 0; pw < pooled_width_; ++pw) {
						const int index = ph * pooled_width_ + pw;
						const int bottom_index =
							use_top_mask ? top_mask[index] : mask[index];
						bottom_diff[bottom_index] += top_data[index];
					}
				}
				bottom_diff += bottom[0]->offset(0, 1);
				top_data += top[0]->offset(0, 1);
				if (use_top_mask) {
					top_mask += top[0]->offset(0, 1);
				}
				else {
					mask += top[0]->offset(0, 1);
				}
			}
		}
		break;
	case PoolingParameter_PoolMethod_AVE:
		// The main loop
		for (int n = 0; n < top[0]->num(); ++n) {
			for (int c = 0; c < channels_; ++c) {
				for (int ph = 0; ph < pooled_height_; ++ph) {
					for (int pw = 0; pw < pooled_width_; ++pw) {
						int hstart = ph * stride_h_ - pad_h_;
						int wstart = pw * stride_w_ - pad_w_;
						int hend = min(hstart + kernel_h_, height_ + pad_h_);
						int wend = min(wstart + kernel_w_, width_ + pad_w_);
						int pool_size = (hend - hstart) * (wend - wstart);
						hstart = max(hstart, 0);
						wstart = max(wstart, 0);
						hend = min(hend, height_);
						wend = min(wend, width_);
						for (int h = hstart; h < hend; ++h) {
							for (int w = wstart; w < wend; ++w) {
								bottom_diff[h * width_ + w] +=
									top_data[ph * pooled_width_ + pw] / pool_size;
							}
						}
					}
				}
				// offset
				bottom_diff += bottom[0]->offset(0, 1);
				top_data += top[0]->offset(0, 1);
			}
		}
		break;
	case PoolingParameter_PoolMethod_STOCHASTIC:
		NOT_IMPLEMENTED;
		break;
	default:
		LOG(FATAL) << "Unknown pooling method.";
	}
}


#ifdef CPU_ONLY
STUB_GPU(HebbianPoolingLayer);
#endif

INSTANTIATE_CLASS(HebbianPoolingLayer);

}