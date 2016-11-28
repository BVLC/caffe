#include <algorithm>
#include <vector>

#include "caffe/layers/hebbian_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void HebbianConvLayer<Dtype>::Add_feedback(const int size, const Dtype* feedback,
		Dtype* bottom) {
		caffe_gpu_axpy<Dtype>(size, (Dtype)1., feedback, bottom);
	}

	template <typename Dtype>
	void HebbianConvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* weight = this->blobs_[0]->gpu_data();
		for (int i = 0; i < bottom.size(); ++i) {
			Dtype* bottom_data = bottom[i]->mutable_gpu_data();
			const Dtype* feedback = bottom[i]->gpu_diff();
			Dtype* top_data = top[i]->mutable_gpu_data();
			for (int n = 0; n < this->num_; ++n) {
				Add_feedback(this->bottom_dim_, feedback + n * this->bottom_dim_, bottom_data + n * this->bottom_dim_);
				this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
					top_data + n * this->top_dim_);
				if (this->bias_term_) {
					const Dtype* bias = this->blobs_[1]->gpu_data();
					this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
				}
			}
		}
		// Hebbian learning (Oja's rule)
		Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
		for (int i = 0; i < top.size(); ++i) {
			const Dtype* top_diff = top[i]->gpu_diff(); // top_diff is set by layer above, which must be a pooling layer (in which it is "bottom_diff")
			const Dtype* bottom_data = bottom[i]->gpu_data();
			for (int n = 0; n < this->num_; ++n) {
				this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
						top_diff + n * this->top_dim_, weight_diff);
			}
		}
	}

	template <typename Dtype>
	void HebbianConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* weight = this->blobs_[0]->gpu_data();
		for (int i = 0; i < top.size(); ++i) {
			const Dtype* top_diff = top[i]->gpu_diff(); // top_diff is set by layer above, which must be a pooling layer (in which it is "bottom_diff")
			// Bias gradient, if necessary.
			if (this->bias_term_ && this->param_propagate_down_[1]) {
				Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
				for (int n = 0; n < this->num_; ++n) {
					this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
				}
			}
			if (propagate_down[i]) {
				Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
				for (int n = 0; n < this->num_; ++n) {
					// feedback to bottom, temporarily store in bottom_diff.
					this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
							bottom_diff + n * this->bottom_dim_);
				}
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(HebbianConvLayer);
}