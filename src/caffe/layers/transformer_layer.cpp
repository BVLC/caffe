#include <vector>

#include <boost/math/special_functions.hpp>
#include "caffe/layers/transformer_layer.hpp"

namespace caffe {

template <typename Dtype>
void TransformerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Layer<Dtype>::LayerSetUp(bottom, top);
  top[0]->Reshape(bottom[0]->shape());
}

template <typename Dtype>
void TransformerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

template <typename Dtype>
void TransformerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  TransformerParameter trans_param = this->layer_param_.transformer_param();
  float rotate_angle = trans_param.rotate_angle()/(float)180;
  float cos_v = boost::math::cos_pi<float>(rotate_angle);
  float sin_v = boost::math::sin_pi<float>(rotate_angle);
  int new_x, new_y, new_n;
  for (int i = 0; i < bottom.size(); ++i) {
    int channels_ = bottom[i]->channels();
    int height_ = bottom[i]->height();
    int width_ = bottom[i]->width();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    float xcenter = (width_-1)/2.0;
    float ycenter = (height_-1)/2.0;
    for (int c = 0; c < channels_; ++c) {
      for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
          new_x = (int) (x-xcenter)*cos_v-(y-ycenter)*sin_v+xcenter;
          new_y = (int) (x-xcenter)*sin_v-(y-ycenter)*cos_v+ycenter;
          if (new_x < 0 || new_y < 0 || new_x >= width_ || new_y >= height_){
            continue;
          }
          new_n = new_y*width_+new_x;
          top_data[c*height_*width_+new_n] = bottom_data[c*height_*width_+y*width_+x];
        }
      }
    }
    bottom_data += bottom[0]->offset(0, 1);
    top_data += top[0]->offset(0, 1);
  }
}

template <typename Dtype>
void TransformerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  TransformerParameter trans_param = this->layer_param_.transformer_param();
  float rotate_angle = trans_param.rotate_angle()/(float)180;
  float cos_v = boost::math::cos_pi<float>(-rotate_angle);
  float sin_v = boost::math::sin_pi<float>(-rotate_angle);
  int new_x, new_y, new_n;
  for (int i = 0; i < top.size(); ++i) {
    int channels_ = top[i]->channels();
    int height_ = top[i]->height();
    int width_ = top[i]->width();
    const Dtype* top_data = top[i]->cpu_diff();
    Dtype* bottom_data = bottom[i]->mutable_cpu_diff();
    float xcenter = (width_-1)/2.0;
    float ycenter = (height_-1)/2.0;
    for (int c = 0; c < channels_; ++c) {
      for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
          new_x = (int) (x-xcenter)*cos_v-(y-ycenter)*sin_v+xcenter;
          new_y = (int) (x-xcenter)*sin_v-(y-ycenter)*cos_v+ycenter;
          new_n = new_y*width_+new_x;
          bottom_data[c*height_*width_+new_n] = top_data[c*height_*width_+y*width_+x];
        }
      }
    }
    bottom_data += bottom[0]->offset(0, 1);
    top_data += top[0]->offset(0, 1);
  }
}

#ifdef CPU_ONLY
STUB_GPU(TransformerLayer);
#endif

INSTANTIATE_CLASS(TransformerLayer);
REGISTER_LAYER_CLASS(Transformer);
}  // namespace caffe
