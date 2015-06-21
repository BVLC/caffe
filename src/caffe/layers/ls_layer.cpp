 #include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common_layers.hpp"

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
    vector<int> weight_shape(2);
    weight_shape[0] = hidden_output_shape[1];
    weight_shape[1] = expected_output_shape[1];
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    
  } 
}

template <typename Dtype>
void LSLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // Dimension verification can be added later  
}

template <typename Dtype>
void LSLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* hidden_output = bottom[0]->cpu_data();
  const Dtype* expected_output = bottom[1]->cpu_data();
  Dtype* beta = this->blobs_[0]->mutable_cpu_data();
  vector<int> hidden_output_shape = bottom[0]->shape();
  vector<int> expected_output_shape = bottom[1]->shape();
  const int M=hidden_output_shape[0];
  const int N=hidden_output_shape[1];
  const int NRHS=expected_output_shape[1];
  // Inverse code here
  caffe_cpu_gelss<Dtype>(M,N,NRHS,hidden_output,beta,expected_output);
}



// #ifdef CPU_ONLY
// //STUB_GPU(LSLayer);
// #endif

INSTANTIATE_CLASS(LSLayer);
REGISTER_LAYER_CLASS(LS);

}  // namespace caffe
