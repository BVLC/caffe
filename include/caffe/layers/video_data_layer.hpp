#ifndef CAFFE_VIDEO_DATA_LAYER_HPP_
#define CAFFE_VIDEO_DATA_LAYER_HPP_

#ifdef USE_OPENCV
#if OPENCV_VERSION == 3
#include <opencv2/videoio.hpp>
#else
#include <opencv2/opencv.hpp>
#endif  // OPENCV_VERSION == 3

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

/**
 * @brief Provides data to the Net from webcam or video files.
 *
 * TODO(weiliu89): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class VideoDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit VideoDataLayer(const LayerParameter& param);
  virtual ~VideoDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "VideoData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  VideoDataParameter_VideoType video_type_;
  cv::VideoCapture cap_;

  int skip_frames_;

  int total_frames_;
  int processed_frames_;
  vector<int> top_shape_;
};

}  // namespace caffe
#endif  // USE_OPENCV

#endif  // CAFFE_VIDEO_DATA_LAYER_HPP_
