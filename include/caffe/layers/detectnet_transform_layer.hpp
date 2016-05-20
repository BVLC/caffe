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

namespace caffe {

template<typename Dtype>
class CoverageGenerator;

template<typename Dtype>
struct BboxLabel_;

struct AugmentSelection;

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


  virtual void Backward_cpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {}

  void transform(
      const Mat3v& inputImage,
      const vector<BboxLabel>& inputBboxes,
      Mat3v* outputImage,
      Dtype* outputLabel);


  /**
   * @return a Dtype from [0..1].
   */
  Dtype randDouble();

  bool augmentation_flip(
      const Mat3v& img,
      Mat3v* img_aug,
      const vector<BboxLabel>& bboxlist,
      vector<BboxLabel>*);
  float augmentation_rotate(
      const Mat3v& img_src,
      Mat3v* img_aug,
      const vector<BboxLabel>& bboxlist,
      vector<BboxLabel>*);
  float augmentation_scale(
      const Mat3v& img,
      Mat3v* img_temp,
      const vector<BboxLabel>& bboxlist,
      vector<BboxLabel>*);
  void transform_scale(
      const Mat3v& img,
      Mat3v* img_temp,
      const vector<BboxLabel>& bboxList,
      vector<BboxLabel>* bboxList_aug,
      const Size2i& size);
  Point2i augmentation_crop(
      const Mat3v& img_temp,
      Mat3v* img_aug,
      const vector<BboxLabel>& bboxlist,
      vector<BboxLabel>*);

  void transform_crop(
      const Mat3v& img_temp,
      Mat3v* img_aug,
      const vector<BboxLabel>& bboxlist,
      vector<BboxLabel>* bboxlist_aug,
      Rect inner,
      Size2i outer_area,
      Point2i outer_offset) const;

  float augmentation_hueRotation(
      const Mat3v& img,
      Mat3v* result);

  float augmentation_desaturation(
      const Mat3v& img,
      Mat3v* result);

  Mat1v getTransformationMatrix(Rect region, Dtype rotation) const;
  Rect getBoundingRect(Rect region, Dtype rotation) const;
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
  boost::array<Dtype, 3> mean_values_;
  shared_ptr<Caffe::RNG> rng_;
};

}  // namespace caffe

#endif /* DETECTNET_TRANSFORMATION_HPP */
