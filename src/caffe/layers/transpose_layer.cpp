#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void TransposeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(2);
    vector<int> weight_shape = this->blobs_[0].shape();
    vector<int> transposed_weight_shape(2);
    transposed_weight_shape[0] = weight_shape[1];
    transposed_weight_shape[1] = weight_shape[0];
    this->blobs_[1].reset(new Blob<Dtype>(transposed_weight_shape));
    
  }  
}

template <typename Dtype>
void TransposeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    // Dimension verification can be added later
}

template <typename Dtype>
void TransposeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* transposed_weight = this->blobs_[1]->mutable_cpu_data();

  // Transpose function code 
}



#ifdef CPU_ONLY
STUB_GPU(TransposeLayer);
#endif

INSTANTIATE_CLASS(TransposeLayer);
REGISTER_LAYER_CLASS(Transpose);

}  // namespace caffe
