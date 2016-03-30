#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/crop_simple_layer.hpp"

namespace caffe {

template <typename Dtype>
void CropSimpleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK(this->layer_param_.has_crop_param()) << "Need crop_param";
  CHECK(this->layer_param_.crop_param().has_crop_width()) << "Need crop_width";
  CHECK(this->layer_param_.crop_param().has_crop_height()) << "Need crop_height";
  crop_width_ = this->layer_param_.crop_param().crop_width();
  crop_height_ = this->layer_param_.crop_param().crop_height();
  CHECK(crop_width_ > 0 && crop_width_ <= bottom[0]->width() && crop_height_ > 0 && crop_height_ <= bottom[0]->height()) <<
      "crop_width and crop_height should be between 1 and blob size";
}

template <typename Dtype>
void CropSimpleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    // Allocate output
    top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), crop_height_, crop_width_);
}

template <typename Dtype>
void CropSimpleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                   const vector<Blob<Dtype>*>& top)
{
    LOG(FATAL) << "Forward CPU crop not implemented.";
}

template <typename Dtype>
void CropSimpleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{  
  LOG(FATAL) << "Backward CPU crop not implemented.";
}
  
#ifdef CPU_ONLY
STUB_GPU(CropSimpleLayer);
#endif

INSTANTIATE_CLASS(CropSimpleLayer);
REGISTER_LAYER_CLASS(CropSimple);

}  // namespace caffe
