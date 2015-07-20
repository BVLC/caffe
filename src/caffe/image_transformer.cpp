#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/random.hpp>

#include <string>
#include <vector>

#include "caffe/image_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
void ImageTransformer<Dtype>::InitRand() {
  const unsigned int rng_seed = caffe_rng_rand();
  rng_.reset(new Caffe::RNG(rng_seed));
}

template <typename Dtype>
int ImageTransformer<Dtype>::RandInt(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

template <typename Dtype>
float ImageTransformer<Dtype>::RandFloat(float min, float max) {
  CHECK(rng_);
  CHECK_GE(max, min);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  boost::uniform_real<float> random_distribution(min, caffe_nextafter<float>(max));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<float> >
      variate_generator(rng, random_distribution);
  return variate_generator();
}

template <typename Dtype>
void ImageTransformer<Dtype>::CVMatToArray(const cv::Mat& cv_img, Dtype* out) {
  int cv_channels = cv_img.channels();
  int cv_height = cv_img.rows;
  int cv_width = cv_img.cols;
  for (int h = 0; h < cv_height; ++h) {
    if (cv_img.elemSize1() == 1) {
	  // channel values are 1 byte wide (uchar)
      const uchar* ptr = cv_img.ptr<uchar>(h);
      int img_index = 0;
      for (int w = 0; w < cv_width; ++w) {
        for (int c = 0; c < cv_channels; ++c) {
          int out_index = (c * cv_height + h) * cv_width + w;
	      //LOG(INFO) << "c: " << c << " h: " << h << " w: " << w << " out_index: " << out_index << " value: " << ((float)ptr[img_index]);
	  	out[out_index] = static_cast<Dtype> (ptr[img_index++]);
        }
      }
	} else if (cv_img.elemSize1() == 4) {
	  // channel values are 4 bytes wide (float)
      const float* ptr = cv_img.ptr<float>(h);
      int img_index = 0;
      for (int w = 0; w < cv_width; ++w) {
        for (int c = 0; c < cv_channels; ++c) {
          int out_index = (c * cv_height + h) * cv_width + w;
	      //LOG(INFO) << "c: " << c << " h: " << h << " w: " << w << " out_index: " << out_index << " value: " << ((float)ptr[img_index]);
	  	out[out_index] = static_cast<Dtype> (ptr[img_index++]);
        }
      }
	}
  }
}

template <typename Dtype>
ResizeImageTransformer<Dtype>::ResizeImageTransformer(const ResizeTransformParameter& resize_param) : 
	ImageTransformer<Dtype>(), param_(resize_param) {
  ValidateParam();
}

template <typename Dtype>
void ResizeImageTransformer<Dtype>::ValidateParam() {
  int num_groups = 0;
  if (param_.width_size()) {
    CHECK(param_.height_size()) << "If width is specified, height must as well";
	CHECK_GT(param_.width(0), 0) << "width must be positive";
	CHECK_GT(param_.height(0), 0) << "height must be positive";

	if (param_.width_size() > 1) {
	  CHECK_GE(param_.width(1), param_.width(0)) << "width upper bound < lower bound";
	}
	if (param_.height_size() > 1) {
	  CHECK_GE(param_.height(1), param_.height(0)) << "height upper bound < lower bound";
	}
	num_groups++;
  }
  if (param_.size_size()) {
	CHECK_GT(param_.size(0), 0) << "Size must be positive";

	if (param_.size_size() > 1) {
	  CHECK_GE(param_.size(1), param_.size(0)) << "size upper bound < lower bound";
	}
	num_groups++;
  }
  if (param_.width_perc_size()) {
    CHECK(param_.height_perc_size()) << "If width_perc is specified, height_perc must as well";
	CHECK_GT(param_.width_perc(0), 0) << "width_perc must be positive";
	CHECK_GT(param_.height_perc(0), 0) << "height_perc must be positive";

	if (param_.width_perc_size() > 1) {
	  CHECK_GE(param_.width_perc(1), param_.width_perc(0)) << "width_perc upper bound < lower bound";
	}
	if (param_.height_perc_size() > 1) {
	  CHECK_GE(param_.height_perc(1), param_.height_perc(0)) << "height_perc upper bound < lower bound";
	}
	num_groups++;
  }
  if (param_.size_perc_size()) {
	CHECK_GT(param_.size_perc(0), 0) << "Size must be positive";

	if (param_.size_perc_size() > 1) {
	  CHECK_GE(param_.size_perc(1), param_.size_perc(0)) << "size_perc upper bound < lower bound";
	}
	num_groups++;
  }

  if (num_groups == 0) {
    CHECK(0) << "No group of resize parameters were specified";
  }
  if (num_groups > 1) {
    CHECK(0) << "Multiple groups of resize parameters were specified";
  }

}

template <typename Dtype>
void ResizeImageTransformer<Dtype>::SampleTransformParams(const vector<int>& in_shape) {
  CHECK_GE(in_shape.size(), 2);
  CHECK_LE(in_shape.size(), 4);
  int in_width = in_shape[in_shape.size() - 2];
  int in_height = in_shape[in_shape.size() - 1];

  if (param_.width_size()) {
    SampleFixedIndependent();
  } else if (param_.size_size()) {
    SampleFixedTied();
  } else if (param_.width_perc_size()) {
    SamplePercIndependent(in_width, in_height);
  } else if (param_.size_perc_size()) {
    SamplePercTied(in_width, in_height);
  } else {
    CHECK(0) << "Invalid resize param";
  }
}

template <typename Dtype>
void ResizeImageTransformer<Dtype>::SamplePercIndependent(int in_width, int in_height) {
  if (param_.width_perc_size() == 1) {
    cur_width_ = (int) (param_.width_perc(0) * in_width);
  } else {
    cur_width_ = (int) (this->RandFloat(param_.width_perc(0), param_.width_perc(1)) * in_width);
  }
  if (param_.height_perc_size() == 1) {
    cur_height_ = (int) (param_.height_perc(0) * in_height);
  } else {
    cur_height_ = (int) (this->RandFloat(param_.height_perc(0), param_.height_perc(1)) * in_height);
  }
}

template <typename Dtype>
void ResizeImageTransformer<Dtype>::SamplePercTied(int in_width, int in_height) {
  if (param_.size_perc_size() == 1) {
    cur_width_ = (int) (param_.size_perc(0) * in_width);
    cur_height_ = (int) (param_.size_perc(0) * in_height);
  } else {
    float perc = this->RandFloat(param_.size_perc(0), param_.size_perc(1));
    cur_width_ = (int) (perc *  in_width);
    cur_height_ = (int) (perc * in_height);
  }
}

template <typename Dtype>
void ResizeImageTransformer<Dtype>::SampleFixedIndependent() {
  if (param_.width_size() == 1) {
    cur_width_ = param_.width(0);
  } else {
    cur_width_ = this->RandInt(param_.width(1) - param_.width(0) + 1) + param_.width(0);
  }
  if (param_.height_size() == 1) {
    cur_height_ = param_.height(0);
  } else {
    cur_height_ = this->RandInt(param_.height(1) - param_.height(0) + 1) + param_.height(0);
  }
}

template <typename Dtype>
void ResizeImageTransformer<Dtype>::SampleFixedTied() {
  if (param_.size_size() == 1) {
    cur_width_ = cur_height_ = param_.size(0);
  } else {
    cur_width_ = cur_height_ = this->RandInt(param_.size(1) - param_.size(0) + 1) + param_.size(0);
  }
}

template <typename Dtype>
vector<int> ResizeImageTransformer<Dtype>::InferOutputShape(const vector<int>& in_shape) {
  CHECK_GE(in_shape.size(), 2);
  CHECK_LE(in_shape.size(), 4);

  vector<int> shape;
  for (int i = 0; i < in_shape.size() - 2; i++) {
    shape.push_back(in_shape[i]);
  }
  shape.push_back(cur_width_);
  shape.push_back(cur_height_);
  return shape;
}

template <typename Dtype>
void ResizeImageTransformer<Dtype>::Transform(const cv::Mat& in, cv::Mat& out) {
  int interpolation;
  switch (param_.interpolation()) {
    case ResizeTransformParameter::INTER_NEAREST:
	  interpolation = cv::INTER_NEAREST;
	  break;
    case ResizeTransformParameter::INTER_LINEAR:
	  interpolation = cv::INTER_LINEAR;
	  break;
    case ResizeTransformParameter::INTER_AREA:
	  interpolation = cv::INTER_AREA;
	  break;
    case ResizeTransformParameter::INTER_CUBIC:
	  interpolation = cv::INTER_CUBIC;
	  break;
    case ResizeTransformParameter::INTER_LANCZOS4:
	  interpolation = cv::INTER_LANCZOS4;
	  break;
	default:
	  interpolation = cv::INTER_NEAREST;
	  break;
  }
  cv::Size size(cur_width_, cur_height_);
  cv::resize(in, out, size, 0, 0, interpolation);
}

template <typename Dtype>
void SequenceImageTransformer<Dtype>::Transform(const cv::Mat& in, cv::Mat& out) {
  cv::Mat one = in;
  cv::Mat two;
  int i;
  for (i = 0; i < transformers_->size(); i++) {
    ImageTransformer<Dtype>* transformer = (*transformers_)[i];
	if (i % 2 == 0) {
	  transformer->Transform(one, two);
	} else {
	  transformer->Transform(two, one);
	}
  }
  // assign based on which variable last acted as the output variable.
  if ( (i - 1) % 2 == 0) {
    out = two;
  } else {
    out = one;
  }
}

template <typename Dtype>
void SequenceImageTransformer<Dtype>::SampleTransformParams(const vector<int>& in_shape) {
  vector<int> shape = in_shape;
  for (int i = 0; i < transformers_->size(); i++) {
    ImageTransformer<Dtype>* transformer = (*transformers_)[i];
	transformer->SampleTransformParams(shape);
	shape = transformer->InferOutputShape(shape);
  }
}

template <typename Dtype>
vector<int> SequenceImageTransformer<Dtype>::InferOutputShape(const vector<int>& in_shape) {
  vector<int> shape = in_shape;
  for (int i = 0; i < transformers_->size(); i++) {
    ImageTransformer<Dtype>* transformer = (*transformers_)[i];
	shape = transformer->InferOutputShape(shape);
  }
  return shape;
}

template <typename Dtype>
ProbImageTransformer<Dtype>::ProbImageTransformer(vector<ImageTransformer<Dtype>*>* transformers, vector<float> weights) :
  transformers_(transformers), probs_(weights) {
  CHECK(transformers_);
  CHECK_EQ(transformers_->size(), weights.size()) << "Number of transformers and weights must be equal: " <<
    transformers_->size() << " vs. " << weights.size();
  CHECK_GT(transformers_->size(), 0) << "Number of transformers must be positive";
  float sum = 0;
  for (int i = 0; i < probs_.size(); i++) {
    sum += probs_[i];
  }
  for (int i = 0; i < probs_.size(); i++) {
    probs_[i] /= sum;
  }
}

template <typename Dtype>
void ProbImageTransformer<Dtype>::Transform(const cv::Mat& in, cv::Mat& out) {
  CHECK_GE(cur_idx_, 0) << "cur_idx_ is not initialized";
  CHECK_LT(cur_idx_, probs_.size()) << "cur_idx_ is too big: " << cur_idx_ << " vs. " << probs_.size();

  (*transformers_)[cur_idx_]->Transform(in, out);
}

template <typename Dtype>
vector<int> ProbImageTransformer<Dtype>::InferOutputShape(const vector<int>& in_shape) {
  CHECK_GE(cur_idx_, 0) << "cur_idx_ is not initialized";
  CHECK_LT(cur_idx_, probs_.size()) << "cur_idx_ is too big: " << cur_idx_ << " vs. " << probs_.size();

  return (*transformers_)[cur_idx_]->InferOutputShape(in_shape);
}

template <typename Dtype>
void ProbImageTransformer<Dtype>::SampleTransformParams(const vector<int>& in_shape) {
  SampleIdx();

  CHECK_GE(cur_idx_, 0) << "cur_idx_ is not initialized";
  CHECK_LT(cur_idx_, probs_.size()) << "cur_idx_ is too big: " << cur_idx_ << " vs. " << probs_.size();

  (*transformers_)[cur_idx_]->SampleTransformParams(in_shape);
}

template <typename Dtype>
void ProbImageTransformer<Dtype>::SampleIdx() {
  float rand = this->RandFloat(0,1);
  float cum_prob = 0;
  int i;
  for (i = 0; i < probs_.size(); i++) {
    cum_prob += probs_[i];
	if (cum_prob >= rand) {
	  break;
    }
  }
  if (i == probs_.size()) {
    i--;
  }
  cur_idx_ = i;
}

// assume out is the proper size...
// assume out is CV_32F
template <typename Dtype>
void LinearImageTransformer<Dtype>::Transform(const cv::Mat& in, cv::Mat& out) {
  const int in_channels = in.channels();
  const int in_height = in.rows;
  const int in_width = in.cols;

  // out uses the same number of channels as in, but uses floats
  out.create(in.size(), CV_32F | (0x18 & in.type()));

  // set up lookup arrays
  float scales[4], shifts[4];  
  if (param_.shift_size() == 1) {
    shifts[0] = shifts[1] = shifts[2] = shifts[3] = param_.shift(0);
  } else {
    for (int i = 0; i < param_.shift_size(); i++) {
	  shifts[i] = param_.shift(i);
	}
  }
  if (param_.scale_size() == 1) {
    scales[0] = scales[1] = scales[2] = scales[3] = param_.scale(0);
  } else {
    for (int i = 0; i < param_.scale_size(); i++) {
	  scales[i] = param_.scale(i);
	}
  }
  
  for (int h = 0; h < in_height; ++h) {
    // channel values are 1 byte wide (uchar)
	if (in.elemSize1() == 1) {
      const uchar* in_ptr = in.ptr<uchar>(h);
      float* out_ptr = out.ptr<float>(h);
      int index = 0;
      for (int w = 0; w < in_width; ++w) {
        for (int c = 0; c < in_channels; ++c) {
  	      out_ptr[index] = scales[c] * in_ptr[index] + shifts[c];
          //LOG(INFO) << "c: " << c << " h: " << h << " w: " << w << " index: " << index << " in_val: " << ((float)in_ptr[index]) << " out_val: " << out_ptr[index];
  	      index++;
        }
      }
	}  else if (in.elemSize1() == 4) {
      const float* in_ptr = in.ptr<float>(h);
      float* out_ptr = out.ptr<float>(h);
      int index = 0;
      for (int w = 0; w < in_width; ++w) {
        for (int c = 0; c < in_channels; ++c) {
  	      out_ptr[index] = scales[c] * in_ptr[index] + shifts[c];
          //LOG(INFO) << "c: " << c << " h: " << h << " w: " << w << " index: " << index << " in_val: " << ((float)in_ptr[index]) << " out_val: " << out_ptr[index];
  	      index++;
        }
      }
	}
  }
}

template <typename Dtype>
vector<int> LinearImageTransformer<Dtype>::InferOutputShape(const vector<int>& in_shape) {
  CHECK_GE(in_shape.size(), 3) << "Must know the number of channels";
  int in_channels = in_shape[in_shape.size() - 3];
  if (param_.shift_size() != 1 && param_.shift_size() != in_channels) {
    CHECK(0) << "Number of shifts is " << param_.shift_size() << " but number of channels is " <<
	  in_channels;
  }
  if (param_.scale_size() != 1 && param_.scale_size() != in_channels) {
    CHECK(0) << "Number of scales is " << param_.scale_size() << " but number of channels is " <<
	  in_channels;
  }
  return in_shape;
}

template <typename Dtype>
vector<int> CropImageTransformer<Dtype>::InferOutputShape(const vector<int>& in_shape) {
  CHECK_GT(cur_height_, 0) << "Unitialized current settings: call SampleTransformParams() first";
  CHECK_GT(cur_width_, 0) << "Unitialized current settings: call SampleTransformParams() first";
  CHECK_GT(in_shape.size(), 2);
  CHECK_LE(in_shape.size(), 4);

  vector<int> shape;
  for (int i = 0; i < in_shape.size() - 2; i++) {
    shape.push_back(in_shape[i]);
  }
  shape.push_back(cur_width_);
  shape.push_back(cur_height_);
  return shape;
}

template <typename Dtype>
void CropImageTransformer<Dtype>::SamplePercIndependent(int in_width, int in_height) {
  if (param_.width_perc_size() == 1) {
    cur_width_ = (int) (param_.width_perc(0) * in_width);
  } else {
    cur_width_ = (int) (this->RandFloat(param_.width_perc(0), param_.width_perc(1)) * in_width);
  }
  if (param_.height_perc_size() == 1) {
    cur_height_ = (int) (param_.height_perc(0) * in_height);
  } else {
    cur_height_ = (int) (this->RandFloat(param_.height_perc(0), param_.height_perc(1)) * in_height);
  }
}

template <typename Dtype>
void CropImageTransformer<Dtype>::SamplePercTied(int in_width, int in_height) {
  if (param_.size_perc_size() == 1) {
    cur_width_ = (int) (param_.size_perc(0) * in_width);
    cur_height_ = (int) (param_.size_perc(0) * in_height);
  } else {
    float perc = this->RandFloat(param_.size_perc(0), param_.size_perc(1));
    cur_width_ = (int) (perc *  in_width);
    cur_height_ = (int) (perc * in_height);
  }
}

template <typename Dtype>
void CropImageTransformer<Dtype>::SampleFixedIndependent() {
  if (param_.width_size() == 1) {
    cur_width_ = param_.width(0);
  } else {
    cur_width_ = this->RandInt(param_.width(1) - param_.width(0) + 1) + param_.width(0);
  }
  if (param_.height_size() == 1) {
    cur_height_ = param_.height(0);
  } else {
    cur_height_ = this->RandInt(param_.height(1) - param_.height(0) + 1) + param_.height(0);
  }
}

template <typename Dtype>
void CropImageTransformer<Dtype>::SampleFixedTied() {
  if (param_.size_size() == 1) {
    cur_width_ = cur_height_ = param_.size(0);
  } else {
    cur_width_ = cur_height_ = this->RandInt(param_.size(1) - param_.size(0) + 1) + param_.size(0);
  }
}

template <typename Dtype>
void CropImageTransformer<Dtype>::ValidateParam() {
  int num_groups = 0;
  if (param_.width_size()) {
    CHECK(param_.height_size()) << "If width is specified, height must as well";
	CHECK_GT(param_.width(0), 0) << "width must be positive";
	CHECK_GT(param_.height(0), 0) << "height must be positive";

	if (param_.width_size() > 1) {
	  CHECK_GE(param_.width(1), param_.width(0)) << "width upper bound < lower bound";
	}
	if (param_.height_size() > 1) {
	  CHECK_GE(param_.height(1), param_.height(0)) << "height upper bound < lower bound";
	}
	num_groups++;
  }
  if (param_.size_size()) {
	CHECK_GT(param_.size(0), 0) << "Size must be positive";

	if (param_.size_size() > 1) {
	  CHECK_GE(param_.size(1), param_.size(0)) << "size upper bound < lower bound";
	}
	num_groups++;
  }
  if (param_.width_perc_size()) {
    CHECK(param_.height_perc_size()) << "If width_perc is specified, height_perc must as well";
	CHECK_GT(param_.width_perc(0), 0) << "width_perc must be positive";
	CHECK_GT(param_.height_perc(0), 0) << "height_perc must be positive";

	if (param_.width_perc_size() > 1) {
	  CHECK_GE(param_.width_perc(1), param_.width_perc(0)) << "width_perc upper bound < lower bound";
	}
	if (param_.height_perc_size() > 1) {
	  CHECK_GE(param_.height_perc(1), param_.height_perc(0)) << "height_perc upper bound < lower bound";
	}
	num_groups++;
  }
  if (param_.size_perc_size()) {
	CHECK_GT(param_.size_perc(0), 0) << "Size must be positive";

	if (param_.size_perc_size() > 1) {
	  CHECK_GE(param_.size_perc(1), param_.size_perc(0)) << "size_perc upper bound < lower bound";
	}
	num_groups++;
  }

  if (num_groups == 0) {
    CHECK(0) << "No group of resize parameters were specified";
  }
  if (num_groups > 1) {
    CHECK(0) << "Multiple groups of resize parameters were specified";
  }

}

template <typename Dtype>
void CropImageTransformer<Dtype>::SampleTransformParams(const vector<int>& in_shape) {
  CHECK_GE(in_shape.size(), 2);
  CHECK_LE(in_shape.size(), 4);
  int in_width = in_shape[in_shape.size() - 1];
  int in_height = in_shape[in_shape.size() - 2];

  if (param_.width_size()) {
    SampleFixedIndependent();
  } else if (param_.size_size()) {
    SampleFixedTied();
  } else if (param_.width_perc_size()) {
    SamplePercIndependent(in_width, in_height);
  } else if (param_.size_perc_size()) {
    SamplePercTied(in_width, in_height);
  } else {
    CHECK(0) << "Invalid crop param";
  }
}

template <typename Dtype>
void CropImageTransformer<Dtype>::Transform(const cv::Mat& in, cv::Mat& out) {
  int crop_h_pos, crop_w_pos;
  int in_height = in.rows;
  int in_width = in.cols;
  CHECK_GE(in_height, cur_height_) << "Cannot crop to larger height";
  CHECK_GE(in_width, cur_width_) << "Cannot crop to larger width";
  switch(param_.location()) {
    case CropTransformParameter::RANDOM:
	  crop_h_pos = this->RandInt(in_height - cur_height_ + 1);
	  crop_w_pos = this->RandInt(in_width - cur_width_ + 1);
	  break;
	case CropTransformParameter::CENTER:
	  crop_h_pos = (in_height - cur_height_) / 2;
	  crop_w_pos = (in_width - cur_width_) / 2;
	  break;
	case CropTransformParameter::RAND_CORNER:
	  {
	    bool left = (bool) this->RandInt(2);
	    bool up = (bool) this->RandInt(2);
	    if (left) {
	      crop_w_pos = 0;
	    } else {
	      crop_w_pos = in_width - cur_width_;
	    }
	    if (up) {
	      crop_h_pos = 0;
	    } else {
	      crop_h_pos = in_height - cur_height_;
	    }
	  }
	  break;
	default:
	  CHECK(0) << "Invalid CropLocation: " << param_.location();
	  break;
  }
  cv::Rect roi(crop_w_pos, crop_h_pos, cur_width_, cur_height_);
  out = in(roi);
}

INSTANTIATE_CLASS(ImageTransformer);
INSTANTIATE_CLASS(ResizeImageTransformer);
INSTANTIATE_CLASS(SequenceImageTransformer);
INSTANTIATE_CLASS(ProbImageTransformer);
INSTANTIATE_CLASS(LinearImageTransformer);
INSTANTIATE_CLASS(CropImageTransformer);

}  // namespace caffe
