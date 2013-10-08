// Copyright 2013 Yangqing Jia
//
// This program takes in a trained network and an input blob, and then dumps
// all the intermediate blobs produced by the net to individual binary
// files stored in protobuffer binary formats.
// Usage:
//    dump_network input_net_param trained_net_param input_blob output_prefix

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
  Caffe::set_mode(Caffe::CPU);
  Caffe::set_phase(Caffe::TEST);

  NetParameter net_param;
  NetParameter trained_net_param;
  ReadProtoFromTextFile(argv[1], &net_param);
  ReadProtoFromBinaryFile(argv[2], &trained_net_param);
  BlobProto input_blob_proto;
  ReadProtoFromBinaryFile(argv[3], &input_blob_proto);
  shared_ptr<Blob<float> > input_blob(new Blob<float>());
  input_blob->FromProto(input_blob_proto);

  vector<Blob<float>* > input_vec;
  input_vec.push_back(input_blob.get());
  // For implementational reasons, we need to first set up the net, and
  // then copy the trained parameters.
  shared_ptr<Net<float> > caffe_net(new Net<float>(net_param, input_vec));
  caffe_net->CopyTrainedLayersFrom(trained_net_param);

  // Run the network without training.
  LOG(ERROR) << "Performing Forward";
  caffe_net->Forward(input_vec);

  // Now, let's dump all the layers
  string output_prefix(argv[4]);
  const vector<string>& blob_names = caffe_net->blob_names();
  const vector<shared_ptr<Blob<float> > >& blobs = caffe_net->blobs();
  for (int blobid = 0; blobid < caffe_net->blobs().size(); ++blobid) {
    // Serialize blob
    LOG(ERROR) << "Dumping " << blob_names[blobid];
    BlobProto output_blob_proto;
    blobs[blobid]->ToProto(&output_blob_proto);
    WriteProtoToBinaryFile(output_blob_proto, output_prefix + blob_names[blobid]);
  }

  // Dump results.
  return 0;
}
