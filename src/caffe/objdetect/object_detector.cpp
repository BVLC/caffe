// Copyright 2014 BVLC and contributors.

#include "caffe/objdetect/object_detector.hpp"

#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <fstream>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
SpatialPyramidPoolingNetObjectDetector<Dtype>::
SpatialPyramidPoolingNetObjectDetector(
    const ObjectDetectorParameter& param) : ObjectDetector<Dtype>(param),
    net_(param.spatial_pyramid_pooling_net_param().net_proto()),
    roi_generator_(
        GetROIGenerator<Dtype>(param.spatial_pyramid_pooling_net_param(
            ).roi_generator_param())),
    regions_merger_(
        GetRegionsMerger(param.spatial_pyramid_pooling_net_param(
            ).regions_merger_param())) {
  net_.CopyTrainedLayersFrom(
      param.spatial_pyramid_pooling_net_param().pretrained_model_file());
}

template <typename Dtype>
void SpatialPyramidPoolingNetObjectDetector<Dtype>::detect(
    const Blob<Dtype>& image, vector<Rect>* object_regions) {
//  vector<Rect> candidate_bboxes;
//  roi_generator_->generate(image, &candidate_bboxes);
////  const shared_ptr<MemoryDataLayer<float> > memory_data_layer =
////        boost::static_pointer_cast<MemoryDataLayer<float> >(
////            caffe_test_net.layer_by_name("data"));
//  const size_t num_candidate_bboxes = candidate_bboxes.size();
//  const size_t num_batches = ceil(num_candidate_bboxes / batch_size);
//  for (size_t i = 0; i < num_batches; ++i) {
////    preprocess images
////    memory_data_layer.add(images);
//    float loss;
//    vector<Blob<float>* > dummy_bottom_vec;
//    const vector<Blob<float>*>& result = net_.Forward(dummy_bottom_vec, &loss);
//    const int top_k = result[0]->count() / result[0]->num();
//    vector<float> confidences(result[0]->cpu_data(),
//                              result[0]->cpu_data() + top_k);
//  }
//  regions_merger_.merge(candidate_bboxes, confidences, object_regions);
}

INSTANTIATE_CLASS(SpatialPyramidPoolingNetObjectDetector);

template <typename Dtype>
ObjectDetector<Dtype>* GetObjectDetector(
    const ObjectDetectorParameter& param) {
  switch (param.type()) {
  case ObjectDetectorParameter_ObjectDetectorType_SPATIAL_PYRAMID_POOLING_NET:
    return new SpatialPyramidPoolingNetObjectDetector<Dtype>(param);
  default:
    LOG(FATAL) << "Unknown ObjectDetector type " << param.type();
  }
}
template <>
ObjectDetector<float>* GetObjectDetector<float>(
    const ObjectDetectorParameter& param);
template <>
ObjectDetector<double>* GetObjectDetector<double>(
    const ObjectDetectorParameter& param);

}  // namespace caffe
