#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/randvec_layer.hpp"

#define MAX_RANDOM 10000

namespace caffe {

template <typename Dtype>
Dtype RandVecLayer<Dtype>::GetRandom(const Dtype lower, const Dtype upper) {
    CHECK(data_rng_);
    CHECK_LT(lower, upper) << "Upper bound must be greater than lower bound!";
    caffe::rng_t* data_rng =
        static_cast<caffe::rng_t*>(data_rng_->generator());
    return static_cast<Dtype>(((*data_rng)()) % static_cast<unsigned int>(
        (upper - lower) * MAX_RANDOM)) / static_cast<Dtype>(MAX_RANDOM)+lower;
}

template <typename Dtype>
void RandVecLayer<Dtype>::LayerSetUp(
const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const RandVecParameter& randvec_param = this->layer_param_.randvec_param();
    batch_size_ = randvec_param.batch_size();
    dim_ = randvec_param.dim();
    height_ = randvec_param.height();
    width_ = randvec_param.width();
    lower_ = randvec_param.lower();
    upper_ = randvec_param.upper();
    iter_idx_ = 1;
    vector<int> top_shape(2);
    top_shape[0] = batch_size_;
    top_shape[1] = dim_;
    if (height_ >0 && width_>0) {
        top_shape.resize(4);
        top_shape[0] = batch_size_;
        top_shape[1] = dim_;
        top_shape[2] = height_;
        top_shape[3] = width_;
    }
    top[0]->Reshape(top_shape);
}

template <typename Dtype>
void RandVecLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const unsigned int data_rng_seed = caffe_rng_rand();
    data_rng_.reset(new Caffe::RNG(data_rng_seed));
    int count = top[0]->count();
    for (int i = 0; i<count; ++i)
        top[0]->mutable_cpu_data()[i] = GetRandom(lower_, upper_);
}

INSTANTIATE_CLASS(RandVecLayer);
REGISTER_LAYER_CLASS(RandVec);

}  // namespace caffe