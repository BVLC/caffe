#include <vector>

#include <boost/math/special_functions/sin_pi.hpp>
#include <boost/math/special_functions/cos_pi.hpp>
#include "caffe/layers/transformer_layer.hpp"

namespace caffe {

template <typename Dtype>
void TransformerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  TransformerParameter trans_param = this->layer_param_.transformer_param();
  float rotate_angle = transformer_param.rotate_angle()/180;
  float cos_v = boost::math::cos_pi<int>(rotate_angle);
  float sin_v = boost::math::sin_pi<int>(rotate_angle);
  int new_x, new_y, new_n;
  for (int i = 0; i < bottom.size(); ++i) {
    int channels_ = bottom[i]->channels();
    int height_ = bottom[i]->height();
    int width_ = bottom[i]->width();
    for (int c = 0; c < channels_; ++c) {
      int xcenter = (width_-1)/2;
      int ycenter = (height_-1)/2;
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* top_data = top[i]->mutable_gpu_data();
      for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
          new_x = (int) (x-xcenter)*cos_v-(y-ycenter)*sin_v+xcenter;
          new_y = (int) (x-xcenter)*sin_v-(y-ycenter)*cos_v+ycenter;
          new_n = new_y*width_+new_x;
          top_data[new_n] = bottom_data[y*width_+x];
        }
      }
      bottom_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
    }
  }
}

template <typename Dtype>
void TransformerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  TransformerParameter trans_param = this->layer_param_.transformer_param();
  float rotate_angle = transformer_param.rotate_angle()/180;
  float cos_v = boost::math::cos_pi<int>(-rotate_angle);
  float sin_v = boost::math::sin_pi<int>(-rotate_angle);
  int new_x, new_y, new_n;
  for (int i = 0; i < top.size(); ++i) {
    int channels_ = top[i]->channels();
    int height_ = top[i]->height();
    int width_ = top[i]->width();
    for (int c = 0; c < channels_; ++c) {
      int xcenter = (width_-1)/2;
      int ycenter = (height_-1)/2;
      const Dtype* top_data = top[i]->gpu_diff();
      Dtype* bottom_data = bottom[i]->mutable_gpu_diff();
      for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
          new_x = (int) (x-xcenter)*cos_v-(y-ycenter)*sin_v+xcenter;
          new_y = (int) (x-xcenter)*sin_v-(y-ycenter)*cos_v+ycenter;
          new_n = new_y*width_+new_x;
          bottom_data[new_n] = top_data[y*width_+x];
        }
      }
      bottom_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TransformerLayer);

}  // namespace caffe
