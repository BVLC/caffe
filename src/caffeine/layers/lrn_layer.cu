#include "caffeine/layer.hpp"
#include "caffeine/vision_layers.hpp"


namespace caffeine {

template <typename Dtype>
void LRNLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) <<
      "Local Response Normalization Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << 
      "Local Response Normalization Layer takes a single blob as output.";
  (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
};

template <typename Dtype>
void LRNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  /*
  const int size = this->layer_param_->local_size();
  const int pre_pad = (size - 1) / 2;
  const Dtype alpha = this->layer_param_->alpha();
  const Dtype beta = this->layer_param_->beta();
  */
  NOT_IMPLEMENTED;
}

template <typename Dtype>
Dtype LRNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  NOT_IMPLEMENTED;
  return Dtype(0.);
}

INSTANTIATE_CLASS(LRNLayer);


}  // namespace caffeine
