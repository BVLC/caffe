#ifndef CAFFE_IMAGE_TRANSFORMER_HPP
#define CAFFE_IMAGE_TRANSFORMER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

// TODO: verify if the width/height dimension order is correct
class ImageTransformer {
 public:
  explicit ImageTransformer() { InitRand(); }
  virtual ~ImageTransformer() {}

  void InitRand();
  virtual void Transform(const cv::Mat& in, cv::Mat& out);
  virtual vector<int> InferOutputShape(const vector<int>& in_shape);
  virtual void SampleTransformParams(const vector<int>& in_shape);

 protected:
  int RandInt(int n);
  float RandFloat(float min, float max);
  shared_ptr<Caffe::RNG> rng_;
};

class ResizeImageTransformer : public ImageTransformer {
 public:
  explicit ResizeImageTransformer(const ResizeTransformParameter& resize_param);
  virtual ~ResizeImageTransformer() {}

  virtual void Transform(const cv::Mat& in, cv::Mat& out);
  virtual vector<int> InferOutputShape(const vector<int>& in_shape);
  virtual void SampleTransformParams(const vector<int>& in_shape);

 protected:
  void ValidateParam();
  void SampleFixedIndependent();
  void SampleFixedTied();
  void SamplePercIndependent(int in_width, int in_height);
  void SamplePercTied(int in_width, int in_height);
  ResizeTransformParameter param_;
  int cur_width_, cur_height_;
};

class SequenceImageTransformer : public ImageTransformer {
 public:
  explicit SequenceImageTransformer(vector<ImageTransformer*>* transformers) :
    transformers_(transformers) {}
  virtual ~SequenceImageTransformer() { if (transformers_) delete transformers_; } 

  virtual void Transform(const cv::Mat& in, cv::Mat& out);
  virtual vector<int> InferOutputShape(const vector<int>& in_shape);
  virtual void SampleTransformParams(const vector<int>& in_shape);

 protected:
  vector<ImageTransformer*>* transformers_;
};

class ProbImageTransformer : public ImageTransformer {
 public:
  explicit ProbImageTransformer(vector<ImageTransformer*>* transformers, vector<float> weights);
  virtual ~ProbImageTransformer() { if (transformers_) delete transformers_; }

  virtual void Transform(const cv::Mat& in, cv::Mat& out);
  virtual vector<int> InferOutputShape(const vector<int>& in_shape);
  virtual void SampleTransformParams(const vector<int>& in_shape);

 protected:
  void SampleIdx();
  vector<ImageTransformer*>* transformers_;
  vector<float> probs_;
  int cur_idx_;
};

// TODO: implement file parameters
class LinearImageTransformer : public ImageTransformer {
 public:
  explicit LinearImageTransformer(LinearTransformParameter param) :
    param_(param) {};
  virtual ~LinearImageTransformer() {};

  virtual void Transform(const cv::Mat& in, cv::Mat& out);
  virtual vector<int> InferOutputShape(const vector<int>& in_shape);
  virtual void SampleTransformParams(const vector<int>& in_shape) {}

 protected:
  LinearTransformParameter param_;
};

}  // namespace caffe

#endif  // CAFFE_IMAGE_TRANSFORMER_HPP_

