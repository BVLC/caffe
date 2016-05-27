#ifndef DETECTNET_COVERAGE_HPP
#define DETECTNET_COVERAGE_HPP

#include <boost/ref.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <map>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"

namespace caffe {

template<typename Dtype>
class CoverageGenerator;

template<typename Dtype>
struct BboxLabel_ {
  typedef cv::Rect_<Dtype> Rectv;
  typedef cv::Point3_<Dtype> Point3v;

  // BboxLabel[0]  = boundingbox topleft X (objective)
  // BboxLabel[1]  = boundingbox topleft Y
  // BboxLabel[2]  = boundingbox width
  // BboxLabel[3]  = boundingbox height
  Rectv bbox;
  // BboxLabel[4]  = alpha angle (objective)
  Dtype alpha;
  // BboxLabel[5]  = class number (objective)
  Dtype classNumber;
  // BboxLabel[6]  = bbox.scenario()
  Dtype scenario;
  // BboxLabel[7]  = y axis rotation
  Dtype roty;
  // BboxLabel[8]  = truncated
  Dtype truncated;
  // BboxLabel[9]  = occluded
  Dtype occlusion;
  // BboxLabel[10]  = object length
  // BboxLabel[11]  = object width
  // BboxLabel[12]  = object height
  Point3v dimensions;
  // BboxLabel[13] = location_x
  // BboxLabel[14] = location_y
  // BboxLabel[15] = location_z
  Point3v location;
};

template <typename Dtype>
class TransformedLabel_ {
 public:
  typedef boost::reference_wrapper<Dtype> reference_wrapper_dtype;

  // foreground:
  Dtype& foreground;
  // top left corner:
  Dtype& topLeft_x;
  Dtype& topLeft_y;
  // bottom right corner:
  Dtype& bottomRight_x;
  Dtype& bottomRight_y;
  // inverse size:
  Dtype& dimension_w;
  Dtype& dimension_h;
  // obj_norm:
  Dtype& obj_norm;
  // coverage classes:
  vector<reference_wrapper_dtype> coverages;

  TransformedLabel_(
      const CoverageGenerator<Dtype>& coverageGenerator,
      Dtype* transformedLabels,
      size_t g_x,
      size_t g_y);
  virtual ~TransformedLabel_() {}
};

template <typename Dtype>
class CoverageRegion_ {
 public:
  typedef cv::Rect_<Dtype> Rectv;

  virtual ~CoverageRegion_() {}

  virtual Dtype area() const = 0;
  virtual Dtype intersectionArea(const Rectv& gridBox) const = 0;
};


template <typename Dtype>
class CoverageGenerator {
 public:
  typedef boost::reference_wrapper<Dtype> reference_wrapper_dtype;
  typedef cv::Mat3b Mat3b;
  typedef cv::Point2i Point2i;
  typedef cv::Point_<Dtype> Point2v;
  typedef cv::Rect Rect;
  typedef cv::Rect_<Dtype> Rectv;
  typedef cv::Scalar Scalar;
  typedef cv::Size_<Dtype> Size2v;
  typedef cv::Vec3i Vec3i;
  typedef BboxLabel_<Dtype> BboxLabel;
  typedef TransformedLabel_<Dtype> TransformedLabel;
  typedef CoverageRegion_<Dtype> CoverageRegion;
  typedef typename vector<reference_wrapper_dtype>::const_iterator
      coverage_iterator;
  typedef map<size_t, size_t> LabelMap;

  static const size_t TRANSFORMED_LABEL_SIZE = 8;

  explicit CoverageGenerator(const DetectNetGroundTruthParameter& param);
  virtual ~CoverageGenerator() {}

  /**
   * @brief computes gridbox labels from a list of bounding boxes.
   */
  virtual void generate(
      Dtype* transformedLabels,
      const vector <BboxLabel>& bboxlist) const;

  inline Vec3i dimensions() const {
    return Vec3i(
        label_size_,
        gridROI_.height,
        gridROI_.width);
  }

  inline bool validClass(size_t iClass) const {
    return labels_.count(iClass) > 0;
  }

  /**
   * @brief Constructs a CoverageGenerator provided settings passed via a
   * TransformationParameter object.
   * @return a pointer to the created object. Caller assumes ownership.
   */
  static CoverageGenerator<Dtype>* create(
      const DetectNetGroundTruthParameter& param);

 protected:
  /**
   * @brief Produce a coverage region from a bounding box defining that region.
   * Coverage region must exist only within the defined bounding box.
   * @return a pointer to the defined coverage region. Caller assumes ownership.
   */
  virtual CoverageRegion* coverageRegion(
      const Rectv& boundingBox) const = 0;


  /**
   * @brief Converts an image-space bounding box to the proper coverage region
   * bounding box defined by this object's TransformationParameters.
   * @param transformedLabels bounds of detection.
   */
  virtual Rectv coverageBoundingBox(const Rectv& boundingBox) const;

  /**
   * @return the inverse of the area of a given coverage region, in gridspace.
   */
  virtual Dtype objectNormValue(
      const CoverageRegion& coverageRegion) const;

  /**
   * @brief Zeroes all data represented by this gridbox array.
   */
  virtual void clearLabel(Dtype* transformedLabels) const;

  /**
   * @brief transforms an imagespace rectangle into a gridspace rectangle,
   * ensuring that the gridspace rectangle is always larger than its
   * corresponding imagespace rectangle.
   */
  virtual Rect imageRectToGridRect(const Rectv& area) const;

  /**
   * @brief Retrieves a gridbox label specification at the given x and y
   * coordinates.
   *
   * @param x x-coord
   * @param y y-coord
   */
  TransformedLabel transformedLabelAt(
      Dtype* transformedLabels,
      size_t x,
      size_t y) const;

  Scalar bboxToColor(const Point2i& tl, const Point2i& br) const;

  LabelMap assignLabels(
      const DetectNetGroundTruthParameter& g_param_) const;

  size_t findNumLabels(const LabelMap& labels) const;

  vector<BboxLabel> pruneBboxes(const vector<BboxLabel>& labels) const;

  const DetectNetGroundTruthParameter param_;
  const Rectv imageROI_;
  const Rect gridROI_;
  const Dtype gridBoxArea_;
  const Dtype minObjNorm_;
  const LabelMap labels_;
  const size_t label_size_;

  friend class TransformedLabel_<Dtype>;
};


template <typename Dtype>
class RectangularCoverageRegion: public CoverageRegion_<Dtype> {
 public:
  typedef cv::Rect_<Dtype> Rectv;

  explicit RectangularCoverageRegion(const Rectv& _region) : region(_region) {}

  virtual Dtype area() const { return region.area(); }
  virtual Dtype intersectionArea(const Rectv& gridBox) const {
    return (region & gridBox).area();
  }
 protected:
  Rectv region;
};


template <typename Dtype>
class RectangularCoverageGenerator: public CoverageGenerator<Dtype> {
 public:
  typedef cv::Rect_<Dtype> Rectv;

  explicit RectangularCoverageGenerator(
      const DetectNetGroundTruthParameter& param)
  : CoverageGenerator<Dtype>(param)
  { }

 protected:
  virtual CoverageRegion_<Dtype>* coverageRegion(
      const Rectv& boundingBox) const {
    return new RectangularCoverageRegion<Dtype>(boundingBox);
  }
};

}  // namespace caffe

#endif /* DETECTNET_COVERAGE_HPP */
