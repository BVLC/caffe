#ifndef DETECTNET_TRANSFORMATION_HPP
#define DETECTNET_TRANSFORMATION_HPP

#include <boost/array.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#ifndef CPU_ONLY
#include "caffe/util/gpu_memory.hpp"
#endif

namespace caffe {

template<typename Dtype>
class CoverageGenerator;

template<typename Dtype>
struct BboxLabel_;

struct AugmentSelection {
  // Color augmentations
  float hue_rotation;
  float saturation;
  // Spatial augmentations
  bool flip;
  cv::Size scale;
  float rotation;
  cv::Point crop_offset;

  bool doHueRotation() const { return std::abs(hue_rotation) > FLT_EPSILON; }
  bool doDesaturation() const { return saturation < (1.0 - 1.0/UINT8_MAX); }
  bool doScale(const cv::Size& s) const {
    return s.height != scale.height || s.width != scale.width;
  }
  bool doRotation() const { return std::abs(rotation) > FLT_EPSILON; }
};


/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class DetectNetTransformationLayer : public Layer<Dtype> {
 public:
  typedef cv::Size2i Size2i;
  typedef cv::Size_<Dtype> Size2v;
  typedef cv::Point2i Point2i;
  typedef cv::Point_<Dtype> Point2v;
  typedef cv::Rect Rect;
  typedef cv::Rect_<Dtype> Rectv;
  typedef cv::Vec<Dtype, 3> Vec3v;
  typedef cv::Mat_<cv::Vec<Dtype, 1> > Mat1v;
  typedef cv::Mat_<Vec3v> Mat3v;
  typedef BboxLabel_<Dtype> BboxLabel;

  explicit DetectNetTransformationLayer(const LayerParameter& param);

  virtual ~DetectNetTransformationLayer() {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DetectNetTransformation"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {}

  void transform(
      const Mat3v& inputImage,
      const vector<BboxLabel>& inputBboxes,
      Mat3v* outputImage,
      Dtype* outputLabel);

  AugmentSelection get_augmentations(cv::Size);

  // Image transformations
  Mat3v transform_image_cpu(const Mat3v&, const AugmentSelection&);
  Mat3v transform_hsv_cpu(const Mat3v&, const AugmentSelection&);
  Mat3v rotate_image_cpu(const Mat3v&, const float);
  Mat3v crop_image_cpu(const Mat3v&, const cv::Size&, const cv::Point&);

  // Label transformations
  void transform_label_cpu(vector<BboxLabel>, Dtype*,
      const AugmentSelection&, const cv::Size&);
  vector<BboxLabel> flip_label_cpu(const vector<BboxLabel>&,
      const cv::Size&);
  vector<BboxLabel> scale_label_cpu(const vector<BboxLabel>&,
      const cv::Size&, const cv::Size&);
  vector<BboxLabel> rotate_label_cpu(const vector<BboxLabel>&,
      const cv::Size&, const float);
  vector<BboxLabel> crop_label_cpu(const vector<BboxLabel>&, const cv::Point&);

  /**
   * @return a Dtype from [0..1].
   */
  Dtype randDouble();

  Mat1v getTransformationMatrix(Rect region, Dtype rotation) const;
  cv::Size getRotatedSize(cv::Size, float rotation) const;
  void matToBlob(const Mat3v& source, Dtype* destination) const;
  void matsToBlob(const vector<Mat3v>& source, Blob<Dtype>* destination) const;
  vector<Mat3v> blobToMats(const Blob<Dtype>& image) const;
  vector<vector<BboxLabel> > blobToLabels(const Blob<Dtype>& labels) const;
  Mat3v dataToMat(
      const Dtype* _data,
      Size2i dimensions) const;
  void retrieveMeanImage(Size2i dimensions = Size2i());
  void retrieveMeanChannels();

  void meanSubtract(Mat3v* source) const;
  void pixelMeanSubtraction(Mat3v* source) const;
  void channelMeanSubtraction(Mat3v* source) const;

  DetectNetAugmentationParameter a_param_;
  DetectNetGroundTruthParameter g_param_;
  TransformationParameter t_param_;

  shared_ptr<CoverageGenerator<Dtype> > coverage_;

  Phase phase_;

  Mat3v data_mean_;
#ifndef CPU_ONLY
  Blob<Dtype> mean_blob_;
  GPUMemory::MultiWorkspace gpu_workspace_augmentations_;
  GPUMemory::MultiWorkspace gpu_workspace_tmpdata_;
#endif
  boost::array<Dtype, 3> mean_values_;
  shared_ptr<Caffe::RNG> rng_;
};

}  // namespace caffe

#endif /* DETECTNET_TRANSFORMATION_HPP */
