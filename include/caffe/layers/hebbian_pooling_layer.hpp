#pragma once

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	/**
	* @brief Pools the input image by taking the max, average, etc. within regions.
	*        Same as pooling_layer, except that Backward() routine uses top data as input, instead of top diff.
	*
	* TODO(dox): thorough documentation for Forward, Backward, and proto params.
	*/
	template <typename Dtype>
	class HebbianPoolingLayer : public PoolingLayer<Dtype> {
	public:
		explicit HebbianPoolingLayer(const LayerParameter& param)
			: PoolingLayer<Dtype>(param) {}
	protected:
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	};
}