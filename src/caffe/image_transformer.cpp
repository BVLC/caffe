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

void ImageTransformer::InitRand() {
  const unsigned int rng_seed = caffe_rng_rand();
  rng_.reset(new Caffe::RNG(rng_seed));
}

int ImageTransformer::RandInt(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

float ImageTransformer::RandFloat(float min, float max) {
  CHECK(rng_);
  CHECK_GE(max, min);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  boost::uniform_real<float> random_distribution(min, caffe_nextafter<float>(max));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<float> >
      variate_generator(rng, random_distribution);
  return variate_generator();
}

ResizeImageTransformer::ResizeImageTransformer(const ResizeTransformParameter& resize_param) : 
	ImageTransformer(), param_(resize_param) {
  ValidateParam();
}

void ResizeImageTransformer::ValidateParam() {
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

void ResizeImageTransformer::SampleTransformParams(const vector<int>& in_shape) {
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

void ResizeImageTransformer::SamplePercIndependent(int in_width, int in_height) {
  if (param_.width_perc_size() == 1) {
    cur_width_ = (int) (param_.width_perc(0) * in_width);
  } else {
    cur_width_ = (int) (RandFloat(param_.width_perc(0), param_.width_perc(1)) * in_width);
  }
  if (param_.height_perc_size() == 1) {
    cur_height_ = (int) (param_.height_perc(0) * in_height);
  } else {
    cur_height_ = (int) (RandFloat(param_.height_perc(0), param_.height_perc(1)) * in_height);
  }
}

void ResizeImageTransformer::SamplePercTied(int in_width, int in_height) {
  if (param_.size_perc_size() == 1) {
    cur_width_ = (int) (param_.size_perc(0) * in_width);
    cur_height_ = (int) (param_.size_perc(0) * in_height);
  } else {
    float perc = RandFloat(param_.size_perc(0), param_.size_perc(1));
    cur_width_ = (int) (perc *  in_width);
    cur_height_ = (int) (perc * in_height);
  }
}

void ResizeImageTransformer::SampleFixedIndependent() {
  if (param_.width_size() == 1) {
    cur_width_ = param_.width(0);
  } else {
    cur_width_ = RandInt(param_.width(1) - param_.width(0) + 1) + param_.width(0);
  }
  if (param_.height_size() == 1) {
    cur_height_ = param_.height(0);
  } else {
    cur_height_ = RandInt(param_.height(1) - param_.height(0) + 1) + param_.height(0);
  }
}

void ResizeImageTransformer::SampleFixedTied() {
  if (param_.size_size() == 1) {
    cur_width_ = cur_height_ = param_.size(0);
  } else {
    cur_width_ = cur_height_ = RandInt(param_.size(1) - param_.size(0) + 1) + param_.size(0);
  }
}

vector<int> ResizeImageTransformer::InferOutputShape(const vector<int>& in_shape) {
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

void ResizeImageTransformer::Transform(const cv::Mat& in, cv::Mat& out) {
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

void SequenceImageTransformer::Transform(const cv::Mat& in, cv::Mat& out) {
  cv::Mat one = in;
  cv::Mat two;
  int i;
  for (i = 0; i < transformers_->size(); i++) {
    ImageTransformer* transformer = (*transformers_)[i];
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

void SequenceImageTransformer::SampleTransformParams(const vector<int>& in_shape) {
  vector<int> shape = in_shape;
  for (int i = 0; i < transformers_->size(); i++) {
    ImageTransformer* transformer = (*transformers_)[i];
	transformer->SampleTransformParams(shape);
	shape = transformer->InferOutputShape(shape);
  }
}

vector<int> SequenceImageTransformer::InferOutputShape(const vector<int>& in_shape) {
  vector<int> shape = in_shape;
  for (int i = 0; i < transformers_->size(); i++) {
    ImageTransformer* transformer = (*transformers_)[i];
	shape = transformer->InferOutputShape(shape);
  }
  return shape;
}

ProbImageTransformer::ProbImageTransformer(vector<ImageTransformer*>* transformers, vector<float> weights) :
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

void ProbImageTransformer::Transform(const cv::Mat& in, cv::Mat& out) {
  CHECK_GE(cur_idx_, 0) << "cur_idx_ is not initialized";
  CHECK_LT(cur_idx_, probs_.size()) << "cur_idx_ is too big: " << cur_idx_ << " vs. " << probs_.size();

  (*transformers_)[cur_idx_]->Transform(in, out);
}

vector<int> ProbImageTransformer::InferOutputShape(const vector<int>& in_shape) {
  CHECK_GE(cur_idx_, 0) << "cur_idx_ is not initialized";
  CHECK_LT(cur_idx_, probs_.size()) << "cur_idx_ is too big: " << cur_idx_ << " vs. " << probs_.size();

  return (*transformers_)[cur_idx_]->InferOutputShape(in_shape);
}

void ProbImageTransformer::SampleTransformParams(const vector<int>& in_shape) {
  SampleIdx();

  CHECK_GE(cur_idx_, 0) << "cur_idx_ is not initialized";
  CHECK_LT(cur_idx_, probs_.size()) << "cur_idx_ is too big: " << cur_idx_ << " vs. " << probs_.size();

  (*transformers_)[cur_idx_]->SampleTransformParams(in_shape);
}

void ProbImageTransformer::SampleIdx() {
  float rand = RandFloat(0,1);
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
void LinearImageTransformer::Transform(const cv::Mat& in, cv::Mat& out) {
  const int in_channels = in.channels();
  const int in_rows = in.rows;
  const int in_columns = in.cols;

  int sizes[] = {in_channels, in_rows, in_columns};
  out.create(in.dims, sizes, CV_32F);

  for (int channel = 0; channel < in_channels; channel++) {
    float scale, shift;
	if (param_.shift_size() == 1) {
	  shift = param_.shift(0);
	} else {
	  shift = param_.shift(channel);
	}
	if (param_.scale_size() == 1) {
	  scale = param_.scale(0);
	} else {
	  scale = param_.scale(channel);
	}
    for (int row = 0; row < in_rows; row++) {
	  for (int col = 0; col < in_columns; col++) {
	     int in_offset = in.step[0] * channel + in.step[1] * row + in.step[2] * col;
	     int out_offset = out.step[0] * channel + out.step[1] * row + out.step[2] * col;
         if (param_.shift_first()) {
	       *(out.data + out_offset) = scale * (*(in.data + in_offset) + shift);
		 } else {
	       *(out.data + out_offset) = scale * (*(in.data + in_offset)) + shift;
		 }
	  }
	}
  }
}

vector<int> LinearImageTransformer::InferOutputShape(const vector<int>& in_shape) {
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

}  // namespace caffe
