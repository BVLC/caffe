#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// take the first bottom and crop its height and width equal to the second bottom
// removed rows/cols are take from the bottom/right respectively
template <typename Dtype>
void SimpleCropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    CropParameter crop_param = this->layer_param_.crop_param();
    CHECK(crop_param.has_blob_name()) << "Must specify a blob to crop like";
	crop_like_blob_ = this->net_->blob_by_name(crop_param.blob_name());

	crop_h_ = crop_w_ = 0;
}

template <typename Dtype>
void SimpleCropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CropParameter crop_param = this->layer_param_.crop_param();

  // this is a little hacky because at network initialization time, the network
  // has not completed setting up its mapping from blob names to blobs, so we cannot
  // access it through the network at that time.  When Reshape is called after net
  // initialization (during Forward), if a proper blob by that name is available, it is
  // used.  If a wrong name is used, then the code will most likely error because no cropping
  // is performed.
  //crop_like_blob_ = this->net_->blob_by_name(crop_param.blob_name());

  int like_height = bottom[0]->height(), like_width = bottom[0]->width();
  //if (crop_like_blob_) {
	  like_height = crop_like_blob_->height();
	  like_width = crop_like_blob_->width();

	  CHECK_GE(bottom[0]->height(), like_height) << "SimpleCropLayer cannot crop height of " <<
		bottom[0]->height() << " to " << like_height;
	  CHECK_GE(bottom[0]->width(), like_width) << "SimpleCropLayer cannot crop width of " <<
		bottom[0]->width() << " to " << like_width;
  //}
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), like_height, like_width);
}

template <typename Dtype>
void SimpleCropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
void SimpleCropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
STUB_GPU(SimpleCropLayer);
#endif

INSTANTIATE_CLASS(SimpleCropLayer);
REGISTER_LAYER_CLASS(SimpleCrop);

}  // namespace caffe
