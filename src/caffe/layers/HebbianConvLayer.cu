#include <algorithm>
#include <vector>

#include "caffe/layers/HebbianConvLayer.hpp"

namespace caffe {

	template <typename Dtype>
	void HebbianConvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	}

	template <typename Dtype>
	void HebbianConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		// this must be implemented
	}

	INSTANTIATE_LAYER_GPU_FUNCS(HebbianConvLayer);
}