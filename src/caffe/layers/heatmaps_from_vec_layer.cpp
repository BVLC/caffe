#include <vector>

#include "caffe/layers/heatmaps_from_vec_layer.hpp"

namespace caffe {

template <typename Dtype>
void HeatmapsFromVecLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) { 
	range_ = 1.5f;
	heatmap_size_ = 32;
	kernel_size_ = 3;
	gradient_fact_ = ((float)(heatmap_size_ - 1)) / (2.f * kernel_size_);
	gaussian_.resize((kernel_size_+1)*(kernel_size_+1)); // bottom-right quarter Gaussian values
	
	// un-normalized Gaussian (bottom-right quarter) in pixel space
	for (int k_r = 0; k_r <= kernel_size_; k_r++)
	{
		for (int k_c = 0; k_c <= kernel_size_; k_c++)
		{
			int linID = k_r * (kernel_size_+1) + k_c;
			gaussian_[linID] = exp(-0.5 * (k_r*k_r + k_c*k_c));
		}
	}
}

template <typename Dtype>
void HeatmapsFromVecLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) { // bottom has N x numJoints x 1 x 3
	std::vector<int> shape = bottom[0]->shape();
	num_vecs_ = shape[1]; // number of vectors corresponds to channels
	proj_vecs_.resize(2*num_vecs_);
	shape[2] = heatmap_size_;
	shape[3] = heatmap_size_;
	top[0]->Reshape(shape);
}

template <typename Dtype>
void HeatmapsFromVecLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	caffe_set(top[0]->count(), (Dtype)0.0, top_data);
	int example_size = top[0]->count(1); // this is C x H x W
	int num = top[0]->shape(0); // batch_size
	
	int u_id, v_id; // top-left corner is 0,0
	// x -> left to right, y -> top to bottom, assuming ortographic camera for the moment
	int top_step = heatmap_size_ * heatmap_size_;
	
	for (int n = 0; n < num; n++)
	{
		for (int j = 0; j < num_vecs_; j++)
		{
			u_id = (int)round((bottom_data[n * (3 * num_vecs_) + j * 3] + range_) / (2 * range_) * (heatmap_size_ - 1));
			v_id = (int)round((bottom_data[n * (3 * num_vecs_) + j * 3 + 1] + range_) / (2 * range_) * (heatmap_size_ - 1));
			proj_vecs_[j*2] = u_id;
			proj_vecs_[j*2 + 1] = v_id;
			
			for (int w = -kernel_size_; w <= kernel_size_; w++)
			{
				for (int h = -kernel_size_; h <= kernel_size_; h++)
				{
					if (u_id + w >= 0 && u_id + w < heatmap_size_ && v_id + h >= 0 && v_id + h < heatmap_size_)
					{ // in this case, put Gaussian in top blob centered at u_id, v_id
						top_data[n * example_size + j * top_step + (v_id + h) * heatmap_size_ + (u_id + w)] = gaussian_[abs(h) * (kernel_size_+1) + abs(w)];
					}
				}
			}
		}
	}
}

template <typename Dtype>
void HeatmapsFromVecLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[0])
	{
		const Dtype* top_data = top[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		caffe_set(bottom[0]->count(), (Dtype)0.0, bottom_diff); // init to 0 to enable accumulation
		
		int example_size = top[0]->count(1); // this is C x H x W
		int top_step = heatmap_size_ * heatmap_size_;
		int num = top[0]->shape(0); // batch_size
		int top_id;
		
		for (int n = 0; n < num; n++)
		{
			for (int j = 0; j < num_vecs_; j++)
			{
				for (int w = -kernel_size_; w <= kernel_size_; w++) // all gradients outside the Gaussian are 0
				{
					for (int h = -kernel_size_; h <= kernel_size_; h++)
					{
						if (proj_vecs_[j*2] + w >= 0 && proj_vecs_[j*2] + w < heatmap_size_ && proj_vecs_[j*2+1] + h >= 0 && proj_vecs_[j*2+1] + h < heatmap_size_)
						{ // in this case, accumulate gradient
							top_id = n * example_size + j * top_step + (proj_vecs_[j*2+1] + h) * heatmap_size_ + (proj_vecs_[j*2] + w);
							// x gradient
							bottom_diff[n * (3 * num_vecs_) + j * 3] += top_diff[top_id] * top_data[top_id] * w * gradient_fact_;
							// y gradient
							bottom_diff[n * (3 * num_vecs_) + j * 3 + 1] += top_diff[top_id] * top_data[top_id] * h * gradient_fact_;
						}
					}
				}
			}
		}
		
	}
}

INSTANTIATE_CLASS(HeatmapsFromVecLayer); 
REGISTER_LAYER_CLASS(HeatmapsFromVec);

}  // namespace caffe
