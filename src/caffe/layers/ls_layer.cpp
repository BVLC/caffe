 #include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void LSLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    vector<int> hidden_output_shape = bottom[0]->shape();
    vector<int> expected_output_shape = bottom[1]->shape();
    vector<int> weight_shape;
    weight_shape[0] = hidden_output_shape[0];
    weight_shape[1] = expected_output_shape[1];
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    
  } 
}

template <typename Dtype>
void LSLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  
}

template <typename Dtype>
void LSLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
 
}



#ifdef CPU_ONLY
STUB_GPU(LSLayer);
#endif

INSTANTIATE_CLASS(LSLayer);
REGISTER_LAYER_CLASS(LS);

}  // namespace caffe
