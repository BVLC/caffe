// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

#include <iostream>


namespace caffe {

	template <typename Dtype>
	Dtype ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {


			const Dtype* bottom_data = bottom[0]->gpu_data();
			Dtype* top_data = (*top)[0]->mutable_gpu_data();
			Dtype* col_data = col_buffer_.mutable_gpu_data();
			Dtype* bias_data = bias_buffer_.mutable_gpu_data();
			Dtype* trans_data = trans_buffer_.mutable_gpu_data();
			const Dtype* weight = this->blobs_[0]->gpu_data();
			int weight_offset = M_ * K_;
			int col_offset = K_ * N_;
			int top_offset = M_ * N_;

			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
				N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
				reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
				(Dtype)0., bias_data);
			//#define RUN_BATCH_GFP
			for (int n = 0; n < num_; n+=mem_group_size) {

				size_t this_mem_group_size = min(mem_group_size,num_-n);
				// First, im2col
				bu_im2col_gpu(bottom_data + bottom[0]->offset(n), channels_, height_,
					width_, kernel_size_, pad_, stride_, col_data, this_mem_group_size);

				// Second, innerproduct with groups
				for (int g = 0; g < group_; ++g) {
					caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
						M_, N_*this_mem_group_size, K_,
						(Dtype)1., weight + weight_offset * g,
						col_data + col_offset * g * this_mem_group_size,
						(Dtype)0., trans_data + top_offset * g * this_mem_group_size);
				}
				// third, add bias
				cu_mat2im_c_gpu(trans_data, num_output_, N_, top_data + (*top)[0]->offset(n), (bias_term_)?(Dtype)1.:(Dtype)0., bias_data, this_mem_group_size);
			}

			return Dtype(0.);
	}



	template <typename Dtype>
	void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
			const Dtype* top_diff = top[0]->gpu_diff();
			const Dtype* weight = this->blobs_[0]->gpu_data();
			Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
			const Dtype* bottom_data = (*bottom)[0]->gpu_data();
			Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
			Dtype* col_data = col_buffer_.mutable_gpu_data();
			Dtype* col_diff = col_buffer_.mutable_gpu_diff();
			Dtype* trans_data = trans_buffer_.mutable_gpu_data();
			// bias gradient if necessary
			Dtype* bias_diff = NULL;

			if (bias_term_) {
				bias_diff = this->blobs_[1]->mutable_gpu_diff();
				CUDA_CHECK(cudaMemset(bias_diff, 0,
					sizeof(Dtype) * this->blobs_[1]->count()));
				for (int n = 0; n < num_; ++n) {
					caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
						1., top_diff + top[0]->offset(n),
						reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
						1., bias_diff);
				}
			}

			int weight_offset = M_ * K_;
			int col_offset = K_ * N_;
			int top_offset = M_ * N_;
			CUDA_CHECK(cudaMemset(weight_diff, 0,
				sizeof(Dtype) * this->blobs_[0]->count()));

			for (int n = 0; n < num_; n+=mem_group_size) {
				// since we saved memory in the forward pass by not storing all col data,
				// we will need to recompute them.
				size_t this_mem_group_size = min(mem_group_size,num_-n);

				bu_im2col_gpu(bottom_data + (*bottom)[0]->offset(n), channels_, height_,
					width_, kernel_size_, pad_, stride_, col_data, this_mem_group_size);

				cu_im2mat_gpu(top_diff + top[0]->offset(n),1, num_output_, N_,
					trans_data,
					this_mem_group_size);
				// gradient w.r.t. weight. Note that we will accumulate diffs.
				for (int g = 0; g < group_; ++g) {
					caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_*this_mem_group_size,
						(Dtype)1., trans_data + top_offset * g * this_mem_group_size,
						col_data + col_offset * g * this_mem_group_size,  (Dtype)1.,
						weight_diff + weight_offset * g);
				}
				// gradient w.r.t. bottom data, if necessary
				if (propagate_down) {
					for (int g = 0; g < group_; ++g) {
						caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_ * this_mem_group_size, M_,
							(Dtype)1., weight + weight_offset * g,
							trans_data + top_offset * g * this_mem_group_size,
							(Dtype)0., col_diff + col_offset * g * this_mem_group_size);
					}
					// col2im back to the data
					bu_col2im_gpu_rot(col_diff, channels_, height_, width_, kernel_size_, pad_,
						stride_, bottom_diff + (*bottom)[0]->offset(n), this_mem_group_size);
				}
			}
	}


	INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
