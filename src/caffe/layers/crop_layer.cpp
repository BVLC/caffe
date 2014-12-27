#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/crop_layer.hpp"
#include "caffe/net.hpp"


namespace caffe {

template <typename Dtype>
void CropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const CropParameter& param = this->layer_param_.crop_param();
  CHECK_EQ(bottom.size(), 2) << "Wrong number of bottom blobs.";
  CHECK_EQ(bottom[0]->num_axes(), 4) << "Only works with 4D blobs.";
  CHECK_EQ(bottom[1]->num_axes(), 4) << "Only works with 4D blobs.";
  crop_h_ = param.offset_height();
  crop_w_ = param.offset_width();
}

template <typename Dtype>
void CropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Check that the image we are cropping minus the margin is bigger than the
  // destination image.
  CHECK_GT(bottom[0]->height()-crop_h_, bottom[1]->height())
      << "invalid offset";
  CHECK_GT(bottom[0]->width()-crop_w_, bottom[1]->width()) << "invalid offset";
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[1]->height(),
      bottom[1]->width());
}

template <typename Dtype>
void CropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < top[0]->num(); ++n) {
    for (int c = 0; c < top[0]->channels(); ++c) {
      for (int h = 0; h < top[0]->height(); ++h) {
        caffe_copy(top[0]->width(),
            bottom_data + bottom[0]->offset(n, c, crop_h_ + h, crop_w_),
            top_data + top[0]->offset(n, c, h));
      }
    }
  }
}

template <typename Dtype>
void CropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  if (propagate_down[0]) {
    caffe_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < top[0]->channels(); ++c) {
        for (int h = 0; h < top[0]->height(); ++h) {
          caffe_copy(top[0]->width(),
              top_diff + top[0]->offset(n, c, h),
              bottom_diff + bottom[0]->offset(n, c, crop_h_ + h, crop_w_));
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CropLayer);
#endif

INSTANTIATE_CLASS(CropLayer);
REGISTER_LAYER_CLASS(Crop);

}  // namespace caffe
