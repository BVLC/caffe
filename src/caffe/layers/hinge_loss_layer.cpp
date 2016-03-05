#include <algorithm>
#include <vector>

#include "caffe/layers/hinge_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"




namespace caffe {

template <typename Dtype>
void HingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  // std::cout << "num" << num << std::endl;

  const Dtype* class_weights_data = this->class_weights_.cpu_data();
  Dtype* class_loss_data = this->class_weights_.mutable_cpu_diff();

  caffe_copy(count, bottom_data, bottom_diff);
  for (int i = 0; i < num; ++i) {
    bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
  }
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      bottom_diff[i * dim + j] = std::max(
        Dtype(0), 1 + bottom_diff[i * dim + j]);
    }
  }

  Dtype* loss = top[0]->mutable_cpu_data();
  switch (this->layer_param_.hinge_loss_param().norm()) {
  case HingeLossParameter_Norm_L1:
    if ( this->layer_param_.loss_param().class_weight_size() ) {
      for (int c = 0; c < this->num_classes_; ++c){
          class_loss_data[c] = caffe_cpu_strided_asum(num,bottom_diff+c,dim);
      }
      loss[0] = caffe_cpu_dot(this->num_classes_, class_weights_data, class_loss_data) / num;
    } else {
      loss[0] = caffe_cpu_asum(count, bottom_diff) / num;
    }
    break;
  case HingeLossParameter_Norm_L2:
    if (this->layer_param_.loss_param().class_weight_size()) {
        for (int c = 0; c < this->num_classes_; ++c) {
          class_loss_data[c] = caffe_cpu_strided_dot(num,bottom_diff+c,dim,bottom_diff+c,dim);
        }
        loss[0] = caffe_cpu_dot(this->num_classes_, class_weights_data, class_loss_data) / num;
    } else {
      loss[0] = caffe_cpu_dot(count, bottom_diff, bottom_diff) / num;
    }
    break;
  default:
    LOG(FATAL) << "Unknown Norm";
  }
}

template <typename Dtype>
void HingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {

    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* label = bottom[1]->cpu_data();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    const Dtype* class_weights_data = this->class_weights_.cpu_data();

    for (int i = 0; i < num; ++i) {
      bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
    }

    const Dtype loss_weight = top[0]->cpu_diff()[0];
    switch (this->layer_param_.hinge_loss_param().norm()) {
    case HingeLossParameter_Norm_L1:
      caffe_cpu_sign(count, bottom_diff, bottom_diff);
      if ( this->layer_param_.loss_param().class_weight_size() ) {
        for (int c = 0; c < this->num_classes_; ++c){
          caffe_strided_scal(num, class_weights_data[c]*loss_weight / num, bottom_diff+c,dim);
        }
      } else {
        caffe_scal(count, loss_weight / num, bottom_diff);
      }
      break;
    case HingeLossParameter_Norm_L2:
      if ( this->layer_param_.loss_param().class_weight_size() ) {
        for (int c = 0; c < this->num_classes_; ++c){
          caffe_strided_scal(num, class_weights_data[c]*loss_weight*2 / num, bottom_diff+c,dim);
        }
      } else {
        caffe_scal(count, loss_weight * 2 / num, bottom_diff);
      }
      break;
    default:
      LOG(FATAL) << "Unknown Norm";
    }
  }
}

INSTANTIATE_CLASS(HingeLossLayer);
REGISTER_LAYER_CLASS(HingeLoss);

}  // namespace caffe
