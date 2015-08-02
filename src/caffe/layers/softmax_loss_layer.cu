#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

	template <typename Dtype>
	__global__ void SoftmaxLossForwardGPU(const int nthreads,
		const Dtype* prob_data, const Dtype* label, const Dtype *weight, Dtype* loss,
		const int num, const int dim, const int spatial_dim,
		const bool has_ignore_label_, const int ignore_label_,
		Dtype* counts) {
			CUDA_KERNEL_LOOP(index, nthreads) {
				const int n = index / spatial_dim;
				const int s = index % spatial_dim;
				const int label_value = static_cast<int>(label[n * spatial_dim + s]);
				if (has_ignore_label_ && label_value == ignore_label_) {
					loss[index] = 0;
					counts[index] = 0;
				} else {

					const Dtype weight_value = (weight != NULL) ? static_cast<Dtype>(weight[n * spatial_dim + s]) : 1;

					loss[index] = - weight_value * log(max(prob_data[n * dim + label_value * spatial_dim + s],
						Dtype(FLT_MIN)));
					counts[index] = weight_value;
				}
			}
	}

	template <typename Dtype>
	void SoftmaxWithLossLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
			softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
			const Dtype* prob_data = prob_.gpu_data();
			const Dtype* label = bottom[1]->gpu_data();
			bool has_weight = bottom.size() >= 3;
			const Dtype* weight = NULL;
			if (has_weight)
				weight = bottom[2]->gpu_data();
			const int dim = prob_.count() / outer_num_;
			const int nthreads = outer_num_ * inner_num_;
			// Since this memory is not used for anything until it is overwritten
			// on the backward pass, we use it here to avoid having to allocate new GPU
			// memory to accumulate intermediate results in the kernel.
			Dtype* loss_data = bottom[0]->mutable_gpu_diff();
			// Similarly, this memory is never used elsewhere, and thus we can use it
			// to avoid having to allocate additional GPU memory.
			Dtype* counts = prob_.mutable_gpu_diff();
			// NOLINT_NEXT_LINE(whitespace/operators)
			SoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
				CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, weight, loss_data,
				outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
			Dtype loss = Dtype(0.0);
			caffe_gpu_asum(nthreads, loss_data, &loss);
			if (normalize_) {
				Dtype count;
				caffe_gpu_asum(nthreads, counts, &count);
				if (count != Dtype(0.0))
					loss /= count;
			} else {
				loss /= outer_num_;
			}
			top[0]->mutable_cpu_data()[0] = loss;
			if (top.size() == 2) {
				top[1]->ShareData(prob_);
			}
	}

	template <typename Dtype>
	__global__ void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
		const Dtype* label, const Dtype *weight, Dtype* bottom_diff, const int num, const int dim,
		const int spatial_dim, const bool has_ignore_label_,
		const int ignore_label_, Dtype* counts) {
			const int channels = dim / spatial_dim;

			CUDA_KERNEL_LOOP(index, nthreads) {
				const int n = index / spatial_dim;
				const int s = index % spatial_dim;
				const int label_value = static_cast<int>(label[n * spatial_dim + s]);

				if (has_ignore_label_ && label_value == ignore_label_) {
					for (int c = 0; c < channels; ++c) {
						bottom_diff[n * dim + c * spatial_dim + s] = 0;
					}
					counts[index] = 0;
				} else {
					bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
					if (weight != NULL){
						const Dtype weight_value = static_cast<Dtype>(weight[n * spatial_dim + s]);
						for (int k = 0; k < channels; ++k)
							bottom_diff[n * dim + k * spatial_dim + s] *= weight_value;

						counts[index] = weight_value;
					}
					else
						counts[index] = 1;
				}
			}
	}

	template <typename Dtype>
	void SoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
			if (propagate_down[1]) {
				LOG(FATAL) << this->type()
					<< " Layer cannot backpropagate to label inputs.";
			}
			if (propagate_down[0]) {
				Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
				const Dtype* prob_data = prob_.gpu_data();
				const Dtype* top_data = top[0]->gpu_data();
				caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
				const Dtype* label = bottom[1]->gpu_data();
				bool has_weight = bottom.size() >= 3;
				const Dtype* weight = NULL;
				if (has_weight)
					weight = bottom[2]->gpu_data();
				const int dim = prob_.count() / outer_num_;
				const int nthreads = outer_num_ * inner_num_;
				// Since this memory is never used for anything else,
				// we use to to avoid allocating new GPU memory.
				Dtype* counts = prob_.mutable_gpu_diff();
				// NOLINT_NEXT_LINE(whitespace/operators)
				SoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
					CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, weight, bottom_diff,
					outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
				const Dtype loss_weight = top[0]->cpu_diff()[0];
				if (normalize_) {
					Dtype count;
					caffe_gpu_asum(nthreads, counts, &count);
					caffe_gpu_scal(prob_.count(), count == Dtype(0.0) ? Dtype(0.0) : (loss_weight / count), bottom_diff);
				} else {
					caffe_gpu_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
				}
			}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossLayer);

}  // namespace caffe
