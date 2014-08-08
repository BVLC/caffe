// This program takes in a trained network and an input blob, and then dumps
// all the intermediate blobs produced by the net to individual binary
// files stored in protobuffer binary formats.
// Usage:
//    dump_network input_net_param trained_net_param
//        input_blob output_prefix 0/1
// if input_net_param is 'none', we will directly load the network from
// trained_net_param. If the last argv is 1, we will do a forward-backward pass
// before dumping everyting, and also dump the who network.

#include <string>
#include <vector>

#include "fcntl.h"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  Caffe::set_mode(Caffe::GPU);
  Caffe::set_phase(Caffe::TEST);

  shared_ptr<Net<float> > caffe_net;
  if (strcmp(argv[1], "none") == 0) {
    // We directly load the net param from trained file
    caffe_net.reset(new Net<float>(argv[2]));
  } else {
    caffe_net.reset(new Net<float>(argv[1]));
  }
  caffe_net->CopyTrainedLayersFrom(argv[2]);

  vector<Blob<float>* > input_vec;
  shared_ptr<Blob<float> > input_blob(new Blob<float>());
  if (strcmp(argv[3], "none") != 0) {
    BlobProto input_blob_proto;
    ReadProtoFromBinaryFile(argv[3], &input_blob_proto);
    input_blob->FromProto(input_blob_proto);
    input_vec.push_back(input_blob.get());
  }

  string output_prefix(argv[4]);
  // Run the network without training.
  LOG(ERROR) << "Performing Forward";
  caffe_net->Forward(input_vec);
  if (argc > 5 && strcmp(argv[5], "1") == 0) {
    LOG(ERROR) << "Performing Backward";
    Caffe::set_phase(Caffe::TRAIN);
    caffe_net->Backward();
    // Dump the network
    NetParameter output_net_param;
    caffe_net->ToProto(&output_net_param, true);
    WriteProtoToBinaryFile(output_net_param,
        output_prefix + output_net_param.name());
  }
  // Now, let's dump all the layers

  const vector<string>& blob_names = caffe_net->blob_names();
  const vector<shared_ptr<Blob<float> > >& blobs = caffe_net->blobs();
  for (int blobid = 0; blobid < caffe_net->blobs().size(); ++blobid) {
    // Serialize blob
    LOG(ERROR) << "Dumping " << blob_names[blobid];
    BlobProto output_blob_proto;
    blobs[blobid]->ToProto(&output_blob_proto);
    WriteProtoToBinaryFile(output_blob_proto,
        output_prefix + blob_names[blobid]);
  }

  return 0;
}
