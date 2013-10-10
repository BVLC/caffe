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
  cudaSetDevice(0);
  Caffe::set_mode(Caffe::GPU);
  Caffe::set_phase(Caffe::TRAIN);

  NetParameter net_param;
  ReadProtoFromTextFile(argv[1],
      &net_param);
  vector<Blob<float>*> bottom_vec;
  Net<float> caffe_net(net_param, bottom_vec);

  // Run the network without training.
  LOG(ERROR) << "Performing Forward";
  caffe_net.Forward(bottom_vec);
  LOG(ERROR) << "Performing Backward";
  LOG(ERROR) << "Initial loss: " << caffe_net.Backward();

  /*
  // Now, let's dump all the layers
  string output_prefix("alexnet_initial_dump_");
  const vector<string>& blob_names = caffe_net.blob_names();
  const vector<shared_ptr<Blob<float> > >& blobs = caffe_net.blobs();
  for (int blobid = 0; blobid < caffe_net.blobs().size(); ++blobid) {
    // Serialize blob
    LOG(ERROR) << "Dumping " << blob_names[blobid];
    BlobProto output_blob_proto;
    blobs[blobid]->ToProto(&output_blob_proto);
    WriteProtoToBinaryFile(output_blob_proto, output_prefix + blob_names[blobid]);
  }
  */

  SolverParameter solver_param;
  solver_param.set_base_lr(0.01);
  solver_param.set_display(1);
  solver_param.set_max_iter(60000);
  solver_param.set_lr_policy("fixed");
  solver_param.set_momentum(0.9);
  solver_param.set_weight_decay(0.0005);
  solver_param.set_snapshot(5000);
  solver_param.set_snapshot_prefix("alexnet");

  LOG(ERROR) << "Starting Optimization";
  SGDSolver<float> solver(solver_param);
  solver.Solve(&caffe_net);
  LOG(ERROR) << "Optimization Done.";

  // Run the network after training.
  LOG(ERROR) << "Performing Forward";
  caffe_net.Forward(bottom_vec);
  LOG(ERROR) << "Performing Backward";
  float loss = caffe_net.Backward();
  LOG(ERROR) << "Final loss: " << loss;

  return 0;
}
