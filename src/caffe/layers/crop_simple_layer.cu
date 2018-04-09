#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/crop_simple_layer.hpp"

namespace caffe {
  
template <typename Dtype> 
__global__ void DoCrop( const int nthreads, 
                        const int src_height, const int src_width, const Dtype* src_data, 
                        const int dest_height, const int dest_width, Dtype* dest_data)
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        int x  = index % dest_width; //w-pos
        int y  = (index / dest_width) % dest_height; //h-pos
        int cn = index / dest_width / dest_height; // channel*num

        float x_src = x + (src_width-dest_width)/2;
        float y_src = y + (src_height-dest_height)/2;

        int index_src = src_width*(src_height*cn + y_src) + x_src;

        // write sample to destination
        dest_data[index] = src_data[index_src];
    }
}

template <typename Dtype> 
__global__ void CropBackward( const int nthreads, 
                              const int src_height, const int src_width, const Dtype* src_data, 
                              const int dest_height, const int dest_width, Dtype* dest_data)
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        int x  = index % src_width; //w-pos
        int y  = (index / src_width) % src_height; //h-pos
        int cn = index / src_width / src_height; // channel*num
        
        float x_dest = x + (dest_width-src_width)/2;
        float y_dest = y + (dest_height-src_height)/2;

        int index_dest = dest_width*(dest_height*cn + y_dest) + x_dest;

        // write sample to destination
        dest_data[index_dest] = src_data[index];
    }
}

template <typename Dtype>
void CropSimpleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    Dtype* top_data = top[0]->mutable_gpu_data(); // dest
    int topwidth = top[0]->width();
    int topheight = top[0]->height();
    int topchannels = top[0]->channels();
    int topcount = top[0]->count();

    const Dtype* bottom_data = bottom[0]->gpu_data(); // source
    int bottomchannels = (bottom)[0]->channels();
    int bottomwidth = (bottom)[0]->width();
    int bottomheight = (bottom)[0]->height();
    int bottomcount = (bottom)[0]->count();

    int num = (bottom)[0]->num(); CHECK_EQ((bottom)[0]->num(), top[0]->num());
    
    DoCrop<Dtype><<<CAFFE_GET_BLOCKS(topcount), CAFFE_CUDA_NUM_THREADS>>>(
          topcount, bottomheight, bottomwidth, bottom_data, 
          topheight, topwidth, top_data);
    CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void CropSimpleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    int count = (top)[0]->count();
    caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
    CropBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, 
        (top)[0]->height(), (top)[0]->width(), top_diff, 
        bottom[0]->height(), bottom[0]->width(), bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CropSimpleLayer);

}  // namespace caffe
