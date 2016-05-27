#ifdef USE_OPENCV

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <boost/array.hpp>
#include <boost/foreach.hpp>
#include <boost/ref.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/static_assert.hpp>

#include <algorithm>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/util/detectnet_coverage.hpp"

using namespace cv;  // NOLINT(build/namespaces)
using boost::array;
using boost::scoped_ptr;
using boost::reference_wrapper;
using boost::ref;

#define foreach_ BOOST_FOREACH
#define TRANSFORMED_LABEL_AT(coverageClass, _x, _y, _z) \
transformedLabels[(coverageClass)->gridROI_.area() * (_z) \
                + (_y) * (coverageClass)->gridROI_.width \
                + (_x)]

namespace caffe {


template<typename Dtype>
void zeroCoverage(vector<reference_wrapper<Dtype> >* coverages) {
  foreach_(reference_wrapper<Dtype>& coverage, *coverages) {
    coverage.get() = 0.0;
  }
}


template<typename Dtype>
CoverageGenerator<Dtype>::CoverageGenerator(
    const DetectNetGroundTruthParameter& param
) : param_(param),
    imageROI_(0, 0, (Dtype) param.image_size_x(), (Dtype) param.image_size_y()),
    gridROI_(
      imageROI_.tl()   * (Dtype)(1.0 / param.stride()),
      imageROI_.size() * (Dtype)(1.0 / param.stride())),
    gridBoxArea_(this->param_.stride() * this->param_.stride()),
    minObjNorm_(this->param_.obj_norm() ? 0.0 : 1.0),
    labels_(this->assignLabels(param)),
    label_size_(findNumLabels(labels_) + TRANSFORMED_LABEL_SIZE) {}


template<typename Dtype>
CoverageGenerator<Dtype>* CoverageGenerator<Dtype>::create(
    const DetectNetGroundTruthParameter& param
) {
  switch (param.coverage_type()) {
    default:
      LOG(WARNING)
          << "Unknown coverage type specified \""
          << param.coverage_type()
          << "\", defaulting to rectangular";
    case DetectNetGroundTruthParameter_CoverageType_RECTANGULAR:
      return new RectangularCoverageGenerator<Dtype>(param);
  }
}


template<typename Dtype>
size_t CoverageGenerator<Dtype>::findNumLabels(
    const LabelMap& labels
) const {
  size_t max_keyvalue = 0;
  foreach_(const LabelMap::value_type& keypair, labels) {
    max_keyvalue = std::max(max_keyvalue, keypair.second);
  }
  return max_keyvalue + 1;
}


template<typename Dtype>
typename CoverageGenerator<Dtype>::LabelMap
CoverageGenerator<Dtype>::assignLabels(
    const DetectNetGroundTruthParameter& g_param_
) const {
  LabelMap result;
  for (size_t iClass = 0; iClass != g_param_.object_class_size(); ++iClass) {
    const DetectNetGroundTruthParameter_ClassMapping& mapping =
        g_param_.object_class(iClass);
    result[mapping.src()] = mapping.dst();
  }
  // if labels is empty, assign a simple mapping so we always have at least one
  //  class. 1 is a magic number specific to Kitti, implying the car class. We
  //  do this to maintain backward compatibility with singleclass.
  if (result.empty()) {
    result[1] = 0;
  }
  return result;
}


template<typename Dtype>
vector<typename CoverageGenerator<Dtype>::BboxLabel>
CoverageGenerator<Dtype>::pruneBboxes(
    const vector<BboxLabel>& bboxList
) const {
  vector<BboxLabel> result; result.reserve(bboxList.size());

  foreach_(BboxLabel cLabel, bboxList) {
    // crop bounding boxes to the dimensions of the screen:
    if (this->param_.crop_bboxes()) {
      // truncated bbox is the union of bounding box and screen rectangle, and
      //  is always smaller than cLabel.bbox:
      Rectv croppedBbox = cLabel.bbox & this->imageROI_;
      // truncation is the area of the intersection over the whole:
      Dtype truncation = 1.0 - std::max(cLabel.truncated, (Dtype)0.0);
      truncation *=
          croppedBbox.area() /
          std::max(cLabel.bbox.area(), (Dtype)FLT_EPSILON);
      // reset cLabel
      cLabel.truncated = std::max(0.0, std::min(1.0, 1.0 - truncation));
      cLabel.bbox = croppedBbox;
    }

    // exclude all labels which are not of the classes we care about:
    if (labels_.count(size_t(cLabel.classNumber)) == 0) { continue; }

    // exclude bounding boxes which don't exist in the scene at all:
    if (cLabel.bbox.area() < FLT_EPSILON) { continue; }

    result.push_back(cLabel);
  }

  return result;
}


template<typename Dtype>
Scalar CoverageGenerator<Dtype>::bboxToColor(
    const Point2i& tl,
    const Point2i& br
) const {
  // TODO: is there an array of these colors elsewhere?
  //  Exclude green, as that's the bounding box color.
  static const array<Scalar, 18> colors = {{
    Scalar(255, 0, 0),      Scalar(0, 0, 255),
    Scalar(255, 255, 0),    Scalar(255, 0, 255),
    Scalar(0, 255, 255),    Scalar(255, 127, 0),
    Scalar(255, 0, 127),    Scalar(0, 255, 127),
    Scalar(127, 255, 0),    Scalar(127, 0, 255),
    Scalar(0, 127, 255),    Scalar(127, 255, 255),
    Scalar(255, 127, 255),  Scalar(255, 255, 127),
    Scalar(127, 127, 255),  Scalar(127, 255, 127),
    Scalar(255, 127, 127),  Scalar(255, 255, 255)
  }};

  size_t iColor = ((((tl.x) * 1000 + tl.y)*1000 + br.x)*1000 + br.y);

  return colors[iColor % colors.size()];
}

template<typename Dtype>
void CoverageGenerator<Dtype>::clearLabel(Dtype* transformedLabels) const {
  // fill all labels with 0's:
  std::fill(
      transformedLabels + gridROI_.area() * 0,
      transformedLabels + gridROI_.area() * label_size_,
      (Dtype) 0.0);
}


template <typename Dtype>
typename CoverageGenerator<Dtype>::Rectv
CoverageGenerator<Dtype>::coverageBoundingBox(const Rectv& boundingBox) const {
  // find the center of the bbox as the midpoint between its top-left and
  //  bottom-right points:
  Point2v center = (boundingBox.tl() + boundingBox.br()) * 0.5;

  // shrink coverage region by a percentage of the bounding box's size:
  Size2v shrunkSize = boundingBox.size() * (Dtype)this->param_.scale_cvg();

  switch (param_.gridbox_type()) {
    // gridbox_min: ensure coverage region is no larger than the size of the
    //  bounding box, but no smaller than a certain area in # of pixels
    case DetectNetGroundTruthParameter_GridboxType_GRIDBOX_MIN:
    {
      Dtype min_cvg_len = this->param_.min_cvg_len();
      shrunkSize.width = min(
          boundingBox.width,
          max(min_cvg_len, shrunkSize.width));
      shrunkSize.height = min(
          boundingBox.height,
          max(min_cvg_len, shrunkSize.height));
      break;
    }
    // gridbox_max: ensure coverage region is no smaller than a certain area in
    //  # of pixels
    case DetectNetGroundTruthParameter_GridboxType_GRIDBOX_MAX:
    {
      Dtype max_cvg_len = this->param_.max_cvg_len();
      shrunkSize.width = min(max_cvg_len, shrunkSize.width);
      shrunkSize.height = min(max_cvg_len, shrunkSize.height);
      break;
    }
  }

  // create a coverage region which defines the above shrunken size, centered
  //  at the center of the bounding box:
  Rectv coverage(
      (center + Point2v(shrunkSize * (Dtype)0.5)),
      (center - Point2v(shrunkSize * (Dtype)0.5)));

  return coverage;
}

template<typename Dtype>
Rect CoverageGenerator<Dtype>::imageRectToGridRect(const Rectv& area) const {
  float stride = param_.stride();

  // define a truncated rectangle which encompasses all gridspaces covered by
  //  this bounding box:
  Point tl(floor(area.tl().x / stride), floor(area.tl().y / stride));
  Point br(ceil(area.br().x / stride), ceil(area.br().y / stride));
  Rect g_area(tl, br);

  // bound truncated rectangle by size of the gridbox area:
  g_area &= gridROI_;

  return g_area;
}


BOOST_STATIC_ASSERT(CoverageGenerator<float>::TRANSFORMED_LABEL_SIZE == 8);

template<typename Dtype>
TransformedLabel_<Dtype>::TransformedLabel_(
    const CoverageGenerator<Dtype>& cgen,
    Dtype* transformedLabels,
    size_t g_x,
    size_t g_y
):  // foreground:
    foreground(TRANSFORMED_LABEL_AT(&cgen, g_x, g_y, 0)),
    // top left corner x,y:
    topLeft_x(TRANSFORMED_LABEL_AT(&cgen, g_x, g_y, 1)),
    topLeft_y(TRANSFORMED_LABEL_AT(&cgen, g_x, g_y, 2)),
    // bottom right corner x,y:
    bottomRight_x(TRANSFORMED_LABEL_AT(&cgen, g_x, g_y, 3)),
    bottomRight_y(TRANSFORMED_LABEL_AT(&cgen, g_x, g_y, 4)),
    // inverse size:
    dimension_w(TRANSFORMED_LABEL_AT(&cgen, g_x, g_y, 5)),
    dimension_h(TRANSFORMED_LABEL_AT(&cgen, g_x, g_y, 6)),
    // obj_norm
    obj_norm(TRANSFORMED_LABEL_AT(&cgen, g_x, g_y, 7)),
    coverages() {
  size_t coverageDimensions =
      cgen.label_size_ - CoverageGenerator<Dtype>::TRANSFORMED_LABEL_SIZE;
  coverages.reserve(coverageDimensions);
  for (size_t iCoverage = 0;
      iCoverage != coverageDimensions;
      ++iCoverage) {
    Dtype& coverage = TRANSFORMED_LABEL_AT(
        &cgen,
        g_x,
        g_y,
        CoverageGenerator<Dtype>::TRANSFORMED_LABEL_SIZE + iCoverage);

    coverages.push_back(ref(coverage));
  }
}


template <typename Dtype>
TransformedLabel_<Dtype> CoverageGenerator<Dtype>::transformedLabelAt(
    Dtype* transformedLabels,
    size_t g_x,
    size_t g_y
) const {
  return TransformedLabel_<Dtype>(*this, transformedLabels, g_x, g_y);
}

template<typename Dtype>
Dtype CoverageGenerator<Dtype>::objectNormValue(
    const CoverageRegion& coverageRegion
) const {
  return this->gridBoxArea_ / coverageRegion.area();
}


template <typename Dtype>
void CoverageGenerator<Dtype>::generate(
    Dtype* transformedLabels,
    const vector <BboxLabel>& _bboxList
) const {
  // ignore all label types we don't care about:
  const vector<BboxLabel> bboxList(this->pruneBboxes(_bboxList));

  // clear out transformed_label, things may remain inside from the last batch
  this->clearLabel(transformedLabels);

  // foreach bbox in list:
  foreach_(const BboxLabel& label, bboxList) {
    Rectv bbox(label.bbox);

    // Define the area in which we intend to mark as containing an object to
    //  the network:
    Rectv coverage(this->coverageBoundingBox(bbox));

    // coverage region is implementation specific and defined by extending
    //  classes, but must fit within coverage Rectf:
    scoped_ptr<CoverageRegion> coverageRegion(
        this->coverageRegion(coverage));
    Dtype dObjNormValue = objectNormValue(*coverageRegion);

    // This Rect includes all gridspaces which overlap the coverage rectangle:
    Rect g_coverage(this->imageRectToGridRect(coverage));

    for (size_t g_y = g_coverage.tl().y; g_y < g_coverage.br().y; g_y++) {
      for (size_t g_x = g_coverage.tl().x; g_x < g_coverage.br().x; g_x++) {
        // the current gridbox:
        Rectv gridBox(
          Point2v(g_x, g_y) * (Dtype)this->param_.stride(),
          Size2v(this->param_.stride(), this->param_.stride()));

        // the amount of the gridbox covered by the coverage region:
        Dtype cvgArea =
            coverageRegion->intersectionArea(gridBox)
          / this->gridBoxArea_;

        TransformedLabel tLabel(this->transformedLabelAt(
            transformedLabels,
            g_x,
            g_y));

        if (cvgArea > FLT_EPSILON) {
          zeroCoverage(&tLabel.coverages);
          Dtype& tLabel_coverages =
              tLabel.coverages[labels_.at(size_t(label.classNumber))];
          // coverage (clamped from 0..1)
          tLabel_coverages = 1.0;

          // bbox
          tLabel.topLeft_x     = bbox.tl().x - gridBox.tl().x;
          tLabel.topLeft_y     = bbox.tl().y - gridBox.tl().y;
          tLabel.bottomRight_x = bbox.br().x - gridBox.tl().x;
          tLabel.bottomRight_y = bbox.br().y - gridBox.tl().y;

          // bbox dimensions
          tLabel.dimension_w = 1.0 / bbox.width;
          tLabel.dimension_h = 1.0 / bbox.height;

          // obj_norm
          tLabel.obj_norm = std::max(this->minObjNorm_, dObjNormValue);

          // foreground:
          tLabel.foreground = 1.0;
        }
      }  // foreach x
    }  // foreach y
  }
}


INSTANTIATE_CLASS(CoverageGenerator);
INSTANTIATE_CLASS(RectangularCoverageGenerator);

}  // namespace caffe

#undef TRANSFORMED_LABEL_AT
#undef foreach_

#endif  // USE_OPENCV
