// Copyright 2013 Yangqing Jia
//
// This program takes in a trained network and an input blob, and then dumps
// all the intermediate blobs produced by the net to individual binary
// files stored in protobuffer binary formats.
// Usage:
//    dump_network input_net_param trained_net_param input_blob output_prefix 0/1
// if input_net_param is 'none', we will directly load the network from
// trained_net_param. If the last argv is 1, we will do a forward-backward pass
// before dumping everyting, and also dump the who network.

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
#include "caffe/solver.hpp"

using namespace caffe;

int main(int argc, char** argv) {
  cudaSetDevice(1);
  Caffe::set_mode(Caffe::GPU);
  Caffe::set_phase(Caffe::TEST);

  NetParameter net_param;
  NetParameter trained_net_param;

  if (strcmp(argv[1], "none") == 0) {
    // We directly load the net param from trained file
    ReadProtoFromBinaryFile(argv[2], &net_param);
  } else {
    ReadProtoFromTextFile(argv[1], &net_param);
  }
  ReadProtoFromBinaryFile(argv[2], &trained_net_param);

  vector<Blob<float>* > input_vec;
  if (strcmp(argv[3], "none") != 0) {
    BlobProto input_blob_proto;
    ReadProtoFromBinaryFile(argv[3], &input_blob_proto);
    shared_ptr<Blob<float> > input_blob(new Blob<float>());
    input_blob->FromProto(input_blob_proto);
    input_vec.push_back(input_blob.get());
  }

  shared_ptr<Net<float> > caffe_net(new Net<float>(net_param));
  caffe_net->CopyTrainedLayersFrom(trained_net_param);

  string output_prefix(argv[4]);
  // Run the network without training.
  LOG(ERROR) << "Performing Forward";
  caffe_net->Forward(input_vec);
  if (argc > 4 && strcmp(argv[4], "1")) {
    LOG(ERROR) << "Performing Backward";
    caffe_net->Backward();
    // Dump the network
    NetParameter output_net_param;
    caffe_net->ToProto(&output_net_param, true);
    WriteProtoToBinaryFile(output_net_param, output_prefix + output_net_param.name());
  }
  // Now, let's dump all the layers

  const vector<string>& blob_names = caffe_net->blob_names();
  const vector<shared_ptr<Blob<float> > >& blobs = caffe_net->blobs();
  for (int blobid = 0; blobid < caffe_net->blobs().size(); ++blobid) {
    // Serialize blob
    LOG(ERROR) << "Dumping " << blob_names[blobid];
    BlobProto output_blob_proto;
    blobs[blobid]->ToProto(&output_blob_proto);
    WriteProtoToBinaryFile(output_blob_proto, output_prefix + blob_names[blobid]);
  }

  return 0;
}
