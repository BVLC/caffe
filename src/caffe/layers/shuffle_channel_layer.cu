#include <algorithm>
#include <vector>

#include "caffe/layers/shuffle_channel_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ShuffleChannelKernel(const int nthreads, const int feature_map_size,
	Dtype *output, const Dtype *input, int group_row, int group_column, int len) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int n = index / group_row / group_column;
		const int i = (index / group_column) % group_row;
		const int j = index % group_column;

		const Dtype* p_i = input + n * feature_map_size + (i * group_column + j) * len;
		Dtype* p_o = output + n * feature_map_size + (j * group_row + i) * len;

		for (int k = 0; k < len; k++)
			p_o[k] = p_i[k];
	}
}

template <typename Dtype>
void ShuffleChannelLayer<Dtype>::Resize_gpu(Dtype *output, const Dtype *input, int group_row, int group_column, int len)
{
    for (int i = 0; i < group_row; ++i) // 2
    {
        for(int j = 0; j < group_column ; ++j) // 3
        {
            const Dtype* p_i = input + (i * group_column + j ) * len;
            Dtype* p_o = output + (j * group_row + i ) * len;

            caffe_copy(len, p_i, p_o);
        }
    }
}

template <typename Dtype>
void ShuffleChannelLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    const int num = bottom[0]->num();
    const int feature_map_size = bottom[0]->count(1);
    const int sp_sz = bottom[0]->count(2);
    const int chs = bottom[0]->channels();

    int group_row = group_;
    int group_column = int(chs / group_row);
    CHECK_EQ(chs, (group_column * group_row)) << "Wrong group size.";
	int count = num * group_column * group_row;
	ShuffleChannelKernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
		count, feature_map_size, top_data, bottom_data, group_row, group_column, sp_sz);
    //Dtype* temp_data = temp_blob_.mutable_gpu_data();
    //for(int n = 0; n < num; ++n)
    //{
    //    Resize_gpu(top_data + n*feature_map_size, bottom_data + n*feature_map_size, group_row, group_column, sp_sz);
    //}
    //caffe_copy(bottom[0]->count(), temp_blob_.gpu_data(), top_data);
}

template <typename Dtype>
void ShuffleChannelLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
      const Dtype* top_diff = top[0]->gpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

      const int num = bottom[0]->num();
      const int feature_map_size = bottom[0]->count(1);
      const int sp_sz = bottom[0]->count(2);
      const int chs = bottom[0]->channels();

      int group_row = int(chs / group_);
      int group_column = group_;
      int count = num * group_column * group_row;
	  ShuffleChannelKernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
		  count, feature_map_size, bottom_diff, top_diff, group_row, group_column, sp_sz);
      //Dtype* temp_diff = temp_blob_.mutable_gpu_diff();
    //  for(int n = 0; n < num; ++n)
    //  {
		  //Resize_gpu(bottom_diff + n * feature_map_size, top_diff + n*feature_map_size, group_row, group_column, sp_sz);
    //  }
      //caffe_copy(top[0]->count(), temp_blob_.gpu_diff(), bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ShuffleChannelLayer);

}  // namespace caffe
