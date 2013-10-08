// Copyright 2013 Yangqing Jia

#include <cuda_runtime.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>

#include <cstring>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/optimization/solver.hpp"

using namespace caffe;

int main(int argc, char** argv) {
  cudaSetDevice(1);
  Caffe::set_mode(Caffe::GPU);

  NetParameter net_param;
  ReadProtoFromBinaryFile(argv[1], &net_param);
  BlobProto input_blob_proto;
  ReadProtoFromBinaryFile(argv[2], &input_blob_proto);
  shared_ptr<Blob<float> > input_blob(new Blob<float>());
  input_blob->FromProto(input_blob_proto);

  vector<Blob<float>* > input_vec;
  input_vec.push_back(input_blob.get());
  // For implementational reasons, we need to first set up the net, and
  // then copy the trained parameters.
  shared_ptr<Net<float> > caffe_net(new Net<float>(net_param, input_vec));
  caffe_net->CopyTrainedLayersFrom(net_param);

  // Run the network without training.
  LOG(ERROR) << "Performing Forward";
  caffe_net->Forward(input_vec);

  // Dump results.
  return 0;
}
