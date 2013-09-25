// Copyright 2013 Yangqing Jia

#ifndef CAFFE_LAYER_H_
#define CAFFE_LAYER_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

using std::map;
using std::vector;
using std::string;

namespace caffe {

template <typename Dtype>
class Net {
 public:
  explicit Net(const NetParameter& param);
  ~Net();
  void Forward(const vector<Blob<Dtype*>> & bottom,
      vector<Blob<Dtype*>* top);
  Dtype Backward(const vector<Blob<Dtype*>> & bottom,
      vector<Blob<Dtype*>* top);

  // For an already initialized net, CopyTrainedLayersFrom() copies the already
  // trained layers from another net parameter instance.
  void CopyTrainedLayersFrom(const NetParameter& param);
  // Writes the net to a proto.
  void ToProto(NetParameter* param, bool write_diff = false);

 protected:
  // Individual layers in the net
  vector<shared_ptr<Layer<Dtype> > > layers_;
  vector<shared_ptr<Layer<Dtype> > > layer_names_;
  // bottom_vecs stores the vectors containing the input for each layer
  vector<vector<Blob<Dtype>*> > bottom_vecs_;
  // top_vecs stores the vectors containing the output for each layer
  vector<vector<Blob<Dtype>* > top_vecs_;
  // blobs stores the blobs that store intermediate results between the
  // layers.
  vector<shared_ptr<Blob<Dtype> > blobs_;
  vector<shared_ptr<Blob<Dtype> > blob_names_;
};


}  // namespace caffe

#endif  // CAFFE_LAYER_H_
