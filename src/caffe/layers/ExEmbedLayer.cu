#include <vector>

#include "caffe/ExEmbedLayer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ExEmbedForward(const int nthreads, const Dtype* bottom_data,
                               const Dtype* weight, const int M, const int N,
                               const int bottomLen,
                               Dtype* top_data)
{
    CUDA_KERNEL_LOOP(cidx, nthreads)
    {
        const int n = cidx / (M * bottomLen);
        const int r = cidx % (M * bottomLen);
        const int m = r / bottomLen;
        //const int k = r % bottomLen;

        const int index = static_cast<int>(bottom_data[r]);

        if (index>=0)
        {
            const int weight_index = index * N + n;
            const int top_index = m * N + n;
            caffe_gpu_atomic_add(weight[weight_index],top_data+top_index);
        }        
    }
}

template <typename Dtype>
__global__ void ExEmbedBackward(const int nthreads, const Dtype* bottom_data,
                               const Dtype* top_diff, const int M, const int N,
                               const int bottomLen,
                               Dtype* weight_diff)
{
    CUDA_KERNEL_LOOP(cidx, nthreads)
    {
        const int n = cidx / (M * bottomLen);
        const int r = cidx % (M * bottomLen);
        const int m = r / bottomLen;
        //const int k = r % bottomLen;

        const int index = static_cast<int>(bottom_data[r]);

        if (index>=0)
        {
            const int weight_index = index * N + n;
            const int top_index = m * N + n;
            caffe_gpu_atomic_add(top_diff[top_index],
                                 weight_diff+weight_index);
        }        
    }
}

template <typename Dtype>
void ExEmbedLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top)
{
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* weight = this->blobs_[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    int bottomLen = bottom[0]->shape(1);

    caffe_gpu_set(top[0]->count(),Dtype(0),top_data);

    const int ccount = M_ * N_ * bottomLen;
    ExEmbedForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(ccount), CAFFE_CUDA_NUM_THREADS>>>(
        ccount, bottom_data, weight, M_, N_, bottomLen, top_data);

    if (bias_term_)
    {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, Dtype(1),
                              bias_multiplier_.gpu_data(), bias, Dtype(1),
                              top_data);
    }
}

template <typename Dtype>
void ExEmbedLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom)
{
    CHECK(!propagate_down[0]) << "Can't backpropagate to ExEmbedLayer input.";
    if (this->param_propagate_down_[0])
    {
        const Dtype* top_diff = top[0]->gpu_diff();
        const Dtype* bottom_data = bottom[0]->gpu_data();
        // Gradient with respect to weight
        Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
        caffe_gpu_set(this->blobs_[0]->count(),Dtype(0),weight_diff);
        
        int bottomLen = bottom[0]->shape(1);
        const int ccount = N_ * M_ * bottomLen;
        ExEmbedBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(ccount), CAFFE_CUDA_NUM_THREADS>>>(
            ccount, bottom_data, top_diff, M_, N_, bottomLen, weight_diff);
        
    }
    if (bias_term_ && this->param_propagate_down_[1])
    {
        const Dtype* top_diff = top[0]->gpu_diff();
        Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
        caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, Dtype(1), top_diff,
                              bias_multiplier_.gpu_data(), Dtype(1), bias_diff);
    }
}
    
INSTANTIATE_LAYER_GPU_FUNCS(ExEmbedLayer);

}
