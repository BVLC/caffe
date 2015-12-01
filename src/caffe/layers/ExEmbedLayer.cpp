#include <vector>
#include <iostream>

#include "caffe/ExEmbedLayer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
void ExEmbedLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    N_ = this->layer_param_.embed_param().num_output();
    CHECK_GT(N_, 0) << "ExEmbedLayer num_output must be positive.";
    K_ = this->layer_param_.embed_param().input_dim();
    CHECK_GT(K_, 0) << "ExEmbedLayer input_dim must be positive.";
    bias_term_ = this->layer_param_.embed_param().bias_term();
    // Check if we need to set up the weights
    if (this->blobs_.size() > 0)
    {
        LOG(INFO) << "Skipping parameter initialization";
    }
    else
    {
        if (bias_term_)
        {
            this->blobs_.resize(2);
        }
        else
        {
            this->blobs_.resize(1);
        }
        // Initialize the weights --
        // transposed from InnerProductLayer for spatial locality.
        vector<int> weight_shape(2);
        weight_shape[0] = K_;
        weight_shape[1] = N_;
        this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
        // fill the weights
        shared_ptr<Filler<Dtype> > weight_filler(
            GetFiller<Dtype>(this->layer_param_.embed_param().weight_filler()));
        weight_filler->Fill(this->blobs_[0].get());
        // If necessary, initialize and fill the bias term
        if (bias_term_)
        {
            vector<int> bias_shape(1, N_);
            this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
            shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
                this->layer_param_.embed_param().bias_filler()));
            bias_filler->Fill(this->blobs_[1].get());
        }
    }  // parameter initialization
    this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void ExEmbedLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top)
{
    // Figure out the dimensions

    M_ = bottom[0]->shape(0);
    vector<int> top_shape = bottom[0]->shape();
    top_shape[1]=N_;
    top[0]->Reshape(top_shape);
    // Set up the bias multiplier
    if (bias_term_)
    {
        vector<int> bias_shape(1, M_);
        bias_multiplier_.Reshape(bias_shape);
        caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
    }
}

template <typename Dtype>
void ExEmbedLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top)
{
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* weight = this->blobs_[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    int bottomLen = bottom[0]->shape(1);
    
    caffe_set(top[0]->count(),Dtype(0),top_data);
    
    //#pragma omp parallel for schedule(static)
    for (int n = 0; n < M_; ++n)
    {
        const Dtype* pData = bottom_data + n*bottomLen;
        for(int c = 0; c < bottomLen; c++)
        {   
            int index = static_cast<int>(pData[c]);
            if (index<0)
            {
                break;
            }
 
            DCHECK_LT(index, K_);
            DCHECK_EQ(static_cast<Dtype>(index), pData[c])
                << "non-integer input";
            caffe_axpy(N_, Dtype(1), weight + index * N_, top_data + n * N_);
        }
    }
    if (bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, Dtype(1),
                              bias_multiplier_.cpu_data(), bias, Dtype(1),
                              top_data);
    }
}

template <typename Dtype>
void ExEmbedLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom)
{
    CHECK(!propagate_down[0]) << "Can't backpropagate to ExEmbedLayer input.";
    if (this->param_propagate_down_[0])
    {
        const Dtype* top_diff = top[0]->cpu_diff();
        const Dtype* bottom_data = bottom[0]->cpu_data();
        // Gradient with respect to weight
        Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
        caffe_set(this->blobs_[0]->count(),Dtype(0),weight_diff);
        
        int bottomLen = bottom[0]->shape(1); 
        int index;
        for (int n = 0; n < M_; ++n)
        {
            const Dtype* pData = bottom_data + n * bottomLen;
            for (int c = 0; c < bottomLen; c++)
            {
                index = static_cast<int>(pData[c]);
                if (index<0)
                {
                    break;
                }
                DCHECK_LT(index, K_);
                DCHECK_EQ(static_cast<Dtype>(index), pData[c])
                    << "non-integer input";
                caffe_axpy(N_, Dtype(1), top_diff + n * N_,
                           weight_diff + index * N_);
            }
        }
    }
    if (bias_term_ && this->param_propagate_down_[1])
    {
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
        caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, Dtype(1), top_diff,
                              bias_multiplier_.cpu_data(), Dtype(1), bias_diff);
    }
}

#ifdef CPU_ONLY
STUB_GPU(ExEmbedLayer);
#endif

INSTANTIATE_CLASS(ExEmbedLayer);
REGISTER_LAYER_CLASS(ExEmbed);

}  // namespace caffe
