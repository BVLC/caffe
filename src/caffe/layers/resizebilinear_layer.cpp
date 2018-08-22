#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/resizebilinear_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void ResizeBilinearLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
  // class ResizeBilinearParameter
  ResizeBilinearParameter rebilinear_param = this->layer_param_.resize_bilinear_param();
  factor_ = rebilinear_param.factor();
  CHECK_GT(factor_, 0) << "Only supports factor greater than 0";
} 

template <typename Dtype>
void ResizeBilinearLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  input_height_ = bottom[0]->height();
  input_width_ = bottom[0]->width();
  ResizeBilinearParameter rebilinear_param = this->layer_param_.resize_bilinear_param();
  if (rebilinear_param.has_factor()) {
    factor_ = rebilinear_param.factor();
	  output_height_ = static_cast<int>(input_height_ * factor_);
	  output_width_ = static_cast<int>(input_width_ * factor_);	
  } else if (rebilinear_param.has_height() && rebilinear_param.has_width()) {
	  output_height_ = rebilinear_param.height();
	  output_width_ = rebilinear_param.width();
  } else {
	LOG(FATAL);
  }
  CHECK_GT(input_height_, 0) << "input height shoule be positive";
  CHECK_GT(input_width_, 0) << "input width shoule be positive";
  CHECK_GT(output_height_, 0) << "output height shoule be positive";
  CHECK_GT(output_width_, 0) << "output width shoule be positive";
  top[0]->Reshape(num_, channels_, output_height_, output_width_);
}

template <typename Dtype>
void ResizeBilinearLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
  float height_scale = static_cast<float>(input_height_) / output_height_;
  float width_scale = static_cast<float>(input_width_) / output_width_;
  const Dtype *input_data = bottom[0]->cpu_data();
  Dtype *output_data = top[0]->mutable_cpu_data();
  for (int b = 0; b < num_; ++b) {
    for (int y = 0; y < output_height_; ++y) {
      float input_y = y * height_scale;
      int y0 = static_cast<int>(std::floor(input_y));
      int y1 = std::min(y0 + 1, input_height_ - 1);
      for (int x = 0; x < output_width_; ++x) {
        float input_x = x * width_scale;
        int x0 = static_cast<int>(std::floor(input_x));
        int x1 = std::min(x0 + 1, input_width_ - 1);
        for (int c = 0; c < channels_; ++c) {
          Dtype interpolation = static_cast<Dtype>(input_data[b * channels_ * input_height_ * input_width_ + c * input_height_ * input_width_ + y0 * input_width_ + x0] *
                                    (1 - (input_y - y0)) * (1 - (input_x - x0)) +
                                 input_data[b * channels_ * input_height_ * input_width_ + c * input_height_ * input_width_ + y1 * input_width_ + x0] *
                                    (input_y - y0) * (1 - (input_x - x0)) +
                                 input_data[b * channels_ * input_height_ * input_width_ + c * input_height_ * input_width_ + y0 * input_width_ + x1] *
                                    (1 - (input_y - y0)) * (input_x - x0) +
                                 input_data[b * channels_ * input_height_ * input_width_ + c * input_height_ * input_width_ + y1 * input_width_ + x1] *
                                    (input_y - y0) * (input_x - x0));
          output_data[b * channels_ * output_height_ * output_width_ + c * output_height_ * output_width_ + y * output_width_ + x] = interpolation;
        }
      }
    }
  }
}

template <typename Dtype>
void ResizeBilinearLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	NOT_IMPLEMENTED;
}


INSTANTIATE_CLASS(ResizeBilinearLayer);
REGISTER_LAYER_CLASS(ResizeBilinear);

} // namespace caffe
