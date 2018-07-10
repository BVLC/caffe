#ifdef USE_OPENCV
#ifndef CAFFE_CONNECTED_COMPONENT_LAYER_HPP_
#define CAFFE_CONNECTED_COMPONENT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


/**
 * @brief Computes a connected components map from a segmentation map.
 */
template<typename Dtype, typename MItype, typename MOtype>
class ConnectedComponentLayer : public Layer<Dtype, MItype, MOtype> {
 public:
  explicit ConnectedComponentLayer(const LayerParameter& param)
    : Layer<Dtype, MItype, MOtype>(param) {
  }

  virtual void LayerSetUp(const vector<Blob<MItype>*>& bottom,
                          const vector<Blob<MOtype>*>& top);

  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
                       const vector<Blob<MOtype>*>& top);

  virtual inline int_tp ExactNumBottomBlobs() const {
    return 1;
  }

  virtual inline int_tp ExactNumTopBlobs() const {
    return 1;
  }

  virtual inline const char* type() const {
    return "ConnectedComponent";
  }

 protected:
    virtual void Forward_cpu(const vector<Blob<MItype>*>& bottom,
                             const vector<Blob<MOtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<MOtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<MItype>*>& bottom);

 private:
     cv::Mat FindBlobs(const int maxlabel, const cv::Mat &input);
};

}  // namespace caffe

#endif  // CAFFE_CONNECTED_COMPONENT_LAYER_HPP_
#endif  // USE_OPENCV
