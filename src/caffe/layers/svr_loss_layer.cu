#include <vector>

#include "caffe/layers/svr_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include<fstream>
using namespace std;

ofstream outFile;
ofstream scoreFile;

namespace caffe {

template <typename Dtype>
void SVRLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  static int i = 0;  
  
  if(i == 0)
  {
    outFile.open("output_.txt");
    scoreFile.open("scores_.txt");
  }
  int count = bottom[0]->count();
  // Compute (f(x_n;w) - y_n)
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  // Compute sum of absolute values over the whole batch 
  // i.e. summation over N(|f(x_n;w) - y_n|)
  Dtype loss;
  caffe_gpu_asum(count, diff_.gpu_data(), &loss);
  // Scale by (1 / batch_size) i.e. (1 / N)
  loss = loss / bottom[0]->num();
  top[0]->mutable_cpu_data()[0] = loss;
  LOG(INFO) << "Output = " << bottom[0]->cpu_data()[0];
  LOG(INFO) << "True Score = " << bottom[1]->cpu_data()[0];
  outFile << bottom[0]->cpu_data()[0] << endl;
  scoreFile << bottom[1]->cpu_data()[0] << endl;
  bottom[0]->gpu_data();
  bottom[1]->gpu_data();
  if(i == 12261)
  {
    outFile.close();
    scoreFile.close();
  }
  i++;
}

template <typename Dtype>
void SVRLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype loss_weight = top[0]->cpu_diff()[0] / (Dtype)bottom[0]->num();
  // We don't propagate gradients to scores input
  if(propagate_down[0])
  {
    // Compute gradient 
    // Remember that gradient of |x| is |x| / x
    caffe_gpu_sign(count, diff_.gpu_data(), bottom_diff);
    // Scale gradients by loss weight which is mostly (1 / batch size)    
    caffe_gpu_scal(count, loss_weight, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SVRLossLayer);

}  // namespace caffe
