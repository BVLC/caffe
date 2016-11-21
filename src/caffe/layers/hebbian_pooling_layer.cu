#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/hebbian_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void HebbianPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		PoolingLayer::Forward_gpu(bottom, top);
	}

	template <typename Dtype>
	void HebbianPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (!propagate_down[0]) {
			return;
		}
		const Dtype* top_data = top[0]->gpu_data();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const int count = bottom[0]->count();
		caffe_gpu_set(count, Dtype(0.), bottom_diff);
		// We'll output the mask to top[1] if it's of size >1.
		const bool use_top_mask = top.size() > 1;
		const int* mask = NULL;
		const Dtype* top_mask = NULL;
		switch (this->layer_param_.pooling_param().pool()) {
		case PoolingParameter_PoolMethod_MAX:
			if (use_top_mask) {
				top_mask = top[1]->gpu_data();
			}
			else {
				mask = max_idx_.gpu_data();
			}
			// NOLINT_NEXT_LINE(whitespace/operators)
			MaxPoolBackward<Dtype><< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
				count, top_data, mask, top_mask, top[0]->num(), channels_,
				height_, width_, pooled_height_, pooled_width_,
				kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
				bottom_diff);
			break;
		case PoolingParameter_PoolMethod_AVE:
			// NOLINT_NEXT_LINE(whitespace/operators)
			AvePoolBackward << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
				count, top_data, top[0]->num(), channels_,
				height_, width_, pooled_height_, pooled_width_, kernel_h_,
				kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff);
			break;
		case PoolingParameter_PoolMethod_STOCHASTIC:
			// NOLINT_NEXT_LINE(whitespace/operators)
			StoPoolBackward << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
				count, rand_idx_.gpu_data(), top_data,
				top[0]->num(), channels_, height_, width_, pooled_height_,
				pooled_width_, kernel_h_, kernel_w_, stride_h_, stride_w_,
				bottom_diff);
			break;
		default:
			LOG(FATAL) << "Unknown pooling method.";
		}
		CUDA_POST_KERNEL_CHECK;
	}

	INSTANTIATE_LAYER_GPU_FUNCS(HebbianPoolingLayer);

}