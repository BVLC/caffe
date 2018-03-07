#include <vector>

#include "caffe/layers/dice_coef_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DiceCoefLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  const int batchsize = bottom[0]->num();
  const int dim = bottom[0]->count(1);
  
  vector<int> multiplier_shape(1, dim);
  vector<int> result_shape(1, batchsize);
  result_.Reshape(result_shape);
  result_tmp_.Reshape(result_shape);
  multiplier_.Reshape(multiplier_shape);
  tmp_.ReshapeLike(*bottom[0]);
  smooth = Dtype(1.);
  caffe_set(dim, Dtype(1), multiplier_.mutable_cpu_data());
  caffe_set(batchsize, smooth, result_tmp_.mutable_cpu_data());
  caffe_set(batchsize, smooth, result_.mutable_cpu_data());
}

template <typename Dtype>
void DiceCoefLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  caffe_mul(bottom[0]->count(), bottom[0]->cpu_data(), bottom[0]->cpu_data(), 
                tmp_.mutable_cpu_data());
  caffe_cpu_gemv(CblasNoTrans, bottom[0]->num(), bottom[0]->count(1), Dtype(1.), tmp_.cpu_data(), 
                    multiplier_.cpu_data(), Dtype(1.), result_tmp_.mutable_cpu_data());
  caffe_mul(bottom[1]->count(), bottom[1]->cpu_data(), bottom[1]->cpu_data(), 
                tmp_.mutable_cpu_data());
  caffe_cpu_gemv(CblasNoTrans, bottom[1]->num(), bottom[1]->count(1), Dtype(1.), tmp_.cpu_data(), 
                    multiplier_.cpu_data(), Dtype(1.), result_tmp_.mutable_cpu_data());
  caffe_mul(bottom[0]->count(), bottom[0]->cpu_data(), bottom[1]->cpu_data(), 
                tmp_.mutable_cpu_data());
  caffe_cpu_gemv(CblasNoTrans, bottom[1]->num(), bottom[1]->count(1), Dtype(2.), tmp_.cpu_data(), 
                    multiplier_.cpu_data(), Dtype(1.), result_.mutable_cpu_data());
  caffe_div(bottom[0]->num(), result_.cpu_data(), result_tmp_.cpu_data(), result_.mutable_cpu_data());
  
  Dtype loss = Dtype(1) - caffe_cpu_asum(bottom[0]->num(), result_.cpu_data()) / bottom[0]->num();
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void DiceCoefLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = Dtype(1.0);
      const int index = (i == 0) ? 1 : 0;
      caffe_copy(bottom[i]->count(), bottom[i]->cpu_data(), bottom[i]->mutable_cpu_diff());
      for (int j = 0; j < bottom[i]->num(); j++) {
        const Dtype alpha = sign * top[0]->cpu_diff()[0] / result_tmp_.cpu_data()[j];
        // LOG(INFO) << top[0]->cpu_diff()[0];
        caffe_cpu_axpby(
          bottom[i]->count(1),              // count
          alpha*Dtype(-2),                  // alpha
          bottom[index]->cpu_data()+j*bottom[i]->count(1),        // a
          alpha*result_.cpu_data()[j]*Dtype(2),                      // beta
          bottom[i]->mutable_cpu_diff()+j*bottom[i]->count(1)
        );  // b
      } 
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(DiceCoefLossLayer);
#endif

INSTANTIATE_CLASS(DiceCoefLossLayer);
REGISTER_LAYER_CLASS(DiceCoefLoss);

}  // namespace caffe
