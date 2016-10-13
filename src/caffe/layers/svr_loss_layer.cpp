#include <vector>

#include "caffe/layers/svr_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include<string>
using namespace std;
namespace caffe {

template <typename Dtype>
void SVRLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void SVRLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  //static int i = 0;
  const Dtype* diff = diff_.cpu_data();
  const Dtype* output = bottom[0]->cpu_data();
  const Dtype* score = bottom[1]->cpu_data();
  
  // Compute (f(x_n;w) - y_n)
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  // Compute sum of absolute values over the whole batch 
  // i.e. summation over N(|f(x_n;w) - y_n|)
  Dtype loss = caffe_cpu_asum(count, diff_.cpu_data());
  // Scale by (1 / batch_size) i.e. (1 / N)
  loss = loss / bottom[0]->num();
  top[0]->mutable_cpu_data()[0] = loss;
  // if(i % 20 == 0)
  // {
    // LOG(INFO) << "SVRLoss Layer: diff_ = " << *diff;
    // LOG(INFO) << "SVRLoss Layer: output = " << *output;
    // LOG(INFO) << "SVRLoss Layer: true score = " << *score;  
  // }
  // if(Layer<Dtype>::phase_ == TRAIN)
    // i++;
  
}

template <typename Dtype>
void SVRLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {  
  int count = bottom[0]->count();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();  
  //static int i = 0;
  Dtype loss_weight = top[0]->cpu_diff()[0] / (Dtype)bottom[0]->num();  
  // We don't propagate gradients to scores input
  if(propagate_down[0])
  {
    // Compute gradient 
    // Remember that gradient of |x| is |x| / x
    caffe_cpu_sign(count, diff_.cpu_data(), bottom_diff);    
    // Scale gradients by a reasonable scaling factor so that the gradient 
    // propagation does not cause large swings (unstable behavior) during 
    // training    
    caffe_scal(count, loss_weight, bottom_diff);
    // if(i % 20 == 0)
    // {      
      // LOG(INFO) << "SVRLoss Layer: gradient = " << *bottom_diff;
    // }
    // if(Layer<Dtype>::phase_ == TRAIN)
      // i++;  
  }
}

#ifdef CPU_ONLY
STUB_GPU(SVRLossLayer);
#endif

INSTANTIATE_CLASS(SVRLossLayer);
REGISTER_LAYER_CLASS(SVRLoss);

}  // namespace caffe
