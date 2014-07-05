// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_OBJECT_DETECTOR_HPP_
#define CAFFE_OBJECT_DETECTOR_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/objdetect/rect.hpp"
#include "caffe/objdetect/regions_merger.hpp"
#include "caffe/objdetect/roi_generator.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
using std::vector;

template <typename Dtype>
class ObjectDetector {
 public:
  explicit ObjectDetector(const ObjectDetectorParameter& param) :
    object_detector_param_(param) {}
  virtual ~ObjectDetector() {}
  virtual void detect(const Blob<Dtype>& image,
                      vector<Rect>* object_regions) = 0;
 protected:
  ObjectDetectorParameter object_detector_param_;
};

template <typename Dtype>
class GenericCNNObjectDetector : public ObjectDetector<Dtype> {
 public:
  explicit GenericCNNObjectDetector(const ObjectDetectorParameter& param);
  virtual ~GenericCNNObjectDetector() {}
  virtual void detect(const Blob<Dtype>& image, vector<Rect>* object_regions);
 protected:
  Net<float> net_;
  shared_ptr<ROIGenerator<Dtype> > roi_generator_;
  shared_ptr<RegionsMerger> regions_merger_;
};

template <typename Dtype>
ObjectDetector<Dtype>* GetObjectDetector(const ObjectDetectorParameter& param);

}  // namespace caffe

#endif  // CAFFE_OBJECT_DETECTOR_HPP_
