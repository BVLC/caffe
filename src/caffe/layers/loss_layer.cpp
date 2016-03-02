#include <vector>

#include "caffe/layers/loss_layer.hpp"
#include <algorithm>    // std::min

namespace caffe {

template <typename Dtype>
void LossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // LossLayers have a non-zero (1) loss by default.
  if (this->layer_param_.loss_weight_size() == 0) {
    this->layer_param_.add_loss_weight(Dtype(1));
  }

  LossParameter loss_param = this->layer_param_.loss_param();
  class_axis_ = loss_param.axis();
  num_classes_ = bottom[0]->shape(class_axis_);


  // std::cout << "loss_param " << loss_param.class_weight_size() << std::endl;
  // std::cout << "class_axis_ " << class_axis_ << std::endl;
  // std::cout << "num_classes_ " << num_classes_ << std::endl;

  vector<int> temp_shape;
  temp_shape.push_back(num_classes_);
  class_weights_.Reshape(temp_shape);

  // std::cout << "class_weights_ shape " << class_weights_.shape_string() << std::endl;
  Dtype* class_weights_data = class_weights_.mutable_cpu_data();
  caffe_set(num_classes_, Dtype(1), class_weights_data);

  for (int i=0; i < std::min(num_classes_,loss_param.class_weight_size()); ++i )
    class_weights_data[i] = loss_param.class_weight(i);

  // for (int i=0; i < num_classes_; ++i )
  //    std::cout << class_weights_data[i] << " ";
  //  std::cout << std::endl;
}

template <typename Dtype>
void LossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
      << "The data and label should have the same first dimension.";
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
}

INSTANTIATE_CLASS(LossLayer);

}  // namespace caffe
