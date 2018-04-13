#include "caffe/layers/reorg_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
    template <typename Dtype>
    __global__ void reorg_kernel(const Dtype *x, int w, int h, int c, int batch, int stride, int forward, Dtype *out)
    {
        int size = batch*c*h*w;
        int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
        if(i >= size) return;
        int in_index = i;
        int in_w = i%w;
        i = i/w;
        int in_h = i%h;
        i = i/h;
        int in_c = i%c;
        i = i/c;
        int b = i%batch;

        int out_c = c/(stride*stride);

        int c2 = in_c % out_c;
        int offset = in_c / out_c;
        int w2 = in_w*stride + offset % stride;
        int h2 = in_h*stride + offset / stride;
        int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));

        if(forward)
        {
            out[out_index] = x[in_index];
        }         
        else
        {
            out[in_index] = x[out_index];
        }
    }

    template<typename Dtype>
    void ReorgLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
        const Dtype *bottom_data = bottom[0]->gpu_data();
        int count = bottom[0]->count();
        Dtype *top_data = top[0]->mutable_gpu_data();
        reorg_kernel<Dtype>
         <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(bottom_data, width_, height_,
                  channels_, batch_num_, stride_, reverse_, top_data);
    }

    template<typename Dtype>
    void ReorgLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                                         const vector<Blob<Dtype> *> &bottom) {
        if(!propagate_down[0]){
            return;
        }
        int count = diff_.count();
        const Dtype *top_diff = diff_.mutable_gpu_diff();
        Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
        reorg_kernel<Dtype>
         <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(top_diff, width_, height_,
                  channels_, batch_num_, stride_, !reverse_, bottom_diff);
    }

INSTANTIATE_LAYER_GPU_FUNCS(ReorgLayer);

}  // namespace caffe
