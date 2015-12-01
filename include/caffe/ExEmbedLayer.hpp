#ifndef CAFFE_EXEMBEDLAYER_HPP_
#define CAFFE_EXEMBEDLAYER_HPP_

#include "caffe/layer.hpp"

namespace caffe
{
template <typename Dtype>
class ExEmbedLayer : public Layer<Dtype>
{
public:
    explicit ExEmbedLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {};
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "ExMultiEmbed"; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom);

private:
    int M_, N_, K_;
    bool bias_term_;
    Blob<Dtype> bias_multiplier_;

};
}

#endif
