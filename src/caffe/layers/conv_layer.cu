// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"




namespace caffe {

	template <typename Dtype>
	Dtype ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {


		if (group_ != 1)
		{
			const Dtype* bottom_data = bottom[0]->gpu_data();
			Dtype* top_data = (*top)[0]->mutable_gpu_data();
			Dtype* col_data = col_buffer_.mutable_gpu_data();
			const Dtype* weight = this->blobs_[0]->gpu_data();
			int weight_offset = M_ * K_;
			int col_offset = K_ * N_;
			int top_offset = M_ * N_;
			for (int n = 0; n < num_; ++n) {
				// First, im2col
				im2col_gpu(bottom_data + bottom[0]->offset(n), channels_, height_,
					width_, kernel_size_, pad_, stride_, col_data);
				// Second, innerproduct with groups
				for (int g = 0; g < group_; ++g) {
					caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
						(Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
						(Dtype)0., top_data + (*top)[0]->offset(n) + top_offset * g);
				}
				// third, add bias
				if (bias_term_) {
					caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
						N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
						reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
						(Dtype)1., top_data + (*top)[0]->offset(n));
				}

			}

			
		}
		else
		{
			const Dtype* bottom_data = bottom[0]->gpu_data();
			Dtype* top_data = (*top)[0]->mutable_gpu_data();
			Dtype* col_data = col_buffer_.mutable_gpu_data();
			Dtype* bias_data = bias_buffer_.mutable_gpu_data();
			Dtype* trans_data = trans_buffer_.mutable_gpu_data();
			const Dtype* weight = this->blobs_[0]->gpu_data();
			int weight_offset = M_ * K_;
			int col_offset = K_ * N_;
			int top_offset = M_ * N_;
			//printf("%s %d: group size %d\n",__FILE__, __LINE__, mem_group_size);
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
							N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
							reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
							(Dtype)0., bias_data);
			for (int n = 0; n < num_; n+=mem_group_size) {
				size_t this_mem_group_size = min(mem_group_size,num_-n);
				//printf("%s %d: group size %d\n",__FILE__, __LINE__, this_mem_group_size);
				// First, im2col
				bu_im2col_gpu(bottom_data + bottom[0]->offset(n), channels_, height_,
					width_, kernel_size_, pad_, stride_, col_data, this_mem_group_size);
				
				// Second, innerproduct with groups
				caffe_gpu_gemm<Dtype>(CblasTrans, CblasTrans,  N_*this_mem_group_size, M_, K_,
					(Dtype)1.,  col_data, weight, (Dtype)0., trans_data);//top_data + (*top)[0]->offset(n) + top_offset * g*this_mem_group_size);

				// third, add bias
				if (bias_term_) {
					for (int i_batch = 0; i_batch< this_mem_group_size; i_batch++){
						//cublasSetStream(Caffe::cublas_handle(),cuda_streams[i_batch]);
						caffe_gpu_geam(CblasNoTrans, CblasTrans, num_output_, N_, (Dtype)1.0, bias_data,  trans_data + num_output_ * N_ * i_batch,
							(Dtype)1.0,
							top_data + (*top)[0]->offset(n+i_batch)
							);
					}
				}

			}

			//CUDA_CHECK(cudaMemcpy(res2, top_data + (*top)[0]->offset(0), num_output_ * N_ * mem_group_size * sizeof(Dtype), cudaMemcpyDeviceToHost));
		}

		//for (int channels = 0; channels < num_output_; channels++)
		//	for (int window = 0; window < N_; window++)
		//		for (int imgidx = 0; imgidx < mem_group_size; imgidx++)
		//		{
		//			if (res1[channels * N_ * mem_group_size + imgidx * N_ + window] != res2[imgidx * N_ * num_output_ + window * num_output_ + channels])
		//				cout << "WRONG!" << endl;
		//		}

		//for (int i = 0; i < num_output_ * N_ * mem_group_size; i++)
		//	if (res1[i] != res2[i])
		//		std::cout << "WRONG! " << res1[i] << " " << res2[i] << std::endl;

		//delete[] res1;
		//delete[] res2;
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
			if (group_>=1){
				int weight_offset = M_ * K_;
				int col_offset = K_ * N_;
				int top_offset = M_ * N_;
				CUDA_CHECK(cudaMemset(weight_diff, 0,
					sizeof(Dtype) * this->blobs_[0]->count()));
				for (int n = 0; n < num_; n++) {
					// since we saved memory in the forward pass by not storing all col data,
					// we will need to recompute them.
					
					im2col_gpu(bottom_data + (*bottom)[0]->offset(n), channels_, height_,
						width_, kernel_size_, pad_, stride_, col_data);
					// gradient w.r.t. weight. Note that we will accumulate diffs.
					for (int g = 0; g < group_; ++g) {
						caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
							(Dtype)1., top_diff + top[0]->offset(n) + top_offset * g,
							col_data + col_offset * g, (Dtype)1.,
							weight_diff + weight_offset * g);
					}
					// gradient w.r.t. bottom data, if necessary
					if (propagate_down) {
						for (int g = 0; g < group_; ++g) {
							caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
								(Dtype)1., weight + weight_offset * g,
								top_diff + top[0]->offset(n) + top_offset * g,
								(Dtype)0., col_diff + col_offset * g);
						}
						// col2im back to the data
						col2im_gpu(col_diff, channels_, height_, width_, kernel_size_, pad_,
							stride_, bottom_diff + (*bottom)[0]->offset(n));
					}
				}
			}else{
				int weight_offset = M_ * K_;
				int col_offset = K_ * N_;
				int top_offset = M_ * N_;

				CUDA_CHECK(cudaMemset(weight_diff, 0,
					sizeof(Dtype) * this->blobs_[0]->count()));
				for (int n = 0; n<num_; n+=mem_group_size){
					// since we saved memory in the forward pass by not storing all col data,
					// we will need to recompute them.
					size_t this_mem_group_size = min(mem_group_size,num_-n);
					//im2col_gpu(bottom_data + (*bottom)[0]->offset(n), channels_, height_,
					//	width_, kernel_size_, pad_, stride_, col_data);
					bu_im2col_gpu_rot(bottom_data + (*bottom)[0]->offset(n), channels_, height_,
						width_, kernel_size_, pad_, stride_, col_data, this_mem_group_size);
					// gradient w.r.t. weight. Note that we will accumulate diffs.

					for (int i_batch = 0; i_batch<this_mem_group_size;i_batch++){
							
					// Maybe we can add batched multiplication here, but since the matrix is large, the accleration seem to be insignificant.
					caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
							(Dtype)1., top_diff + top[0]->offset(n+i_batch),
							col_data+col_offset*i_batch, (Dtype)1.,
							weight_diff);
					}

					// gradient w.r.t. bottom data, if necessary
					// can group to batched gemm here
					if (propagate_down) {
						for (int i_batch = 0; i_batch<this_mem_group_size;i_batch++){
							batch_left_ptr_list[i_batch] = weight;
							batch_right_ptr_list[i_batch] = top_diff + top[0]->offset(n+i_batch);
							batch_result_ptr_list[i_batch] = col_diff+col_offset*i_batch;
						}
						

						CUDA_CHECK(cudaMemcpy(d_batch_left_ptr_list, batch_left_ptr_list, this_mem_group_size*sizeof(Dtype*),cudaMemcpyHostToDevice));
						CUDA_CHECK(cudaMemcpy(d_batch_right_ptr_list, batch_right_ptr_list, this_mem_group_size*sizeof(Dtype*),cudaMemcpyHostToDevice));
						CUDA_CHECK(cudaMemcpy(d_batch_result_ptr_list, batch_result_ptr_list, this_mem_group_size*sizeof(Dtype*),cudaMemcpyHostToDevice));

						caffe_gpu_gemm_batched<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
								(Dtype)1., d_batch_left_ptr_list,
								d_batch_right_ptr_list,
								(Dtype)0., d_batch_result_ptr_list,
								this_mem_group_size);

						/*for (int i_batch = 0; i_batch<this_mem_group_size;i_batch++){
							caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
									(Dtype)1., weight,
									top_diff + top[0]->offset(n+i_batch),
									(Dtype)0., col_diff + col_offset*i_batch);
						}*/
						// col2im back to the data
						bu_col2im_gpu(col_diff, channels_, height_, width_, kernel_size_, pad_,
							stride_, bottom_diff + (*bottom)[0]->offset(n), this_mem_group_size);
					}
				}
			}
	}


	INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
