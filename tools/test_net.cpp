// Copyright 2014 BVLC and contributors.
//
// This is a simple script that allows one to quickly test a network whose
// structure is specified by text format protocol buffers, and whose parameter
// are loaded from a pre-trained network.
// Usage:
//    test_net net_proto pretrained_net_proto iterations [CPU/GPU]

#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>

#include "caffe/caffe.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  if (argc < 4 || argc > 5) {
    LOG(ERROR) << "test_net net_proto pretrained_net_proto iterations "
        << "[CPU/GPU]";
    return 1;
  }

  cudaSetDevice(0);
  Caffe::set_phase(Caffe::TEST);

  if (argc == 5 && strcmp(argv[4], "GPU") == 0) {
    LOG(ERROR) << "Using GPU";
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  Net<float> caffe_test_net(argv[1]);
  NetParameter trained_net_param;
  ReadProtoFromBinaryFile(argv[2], &trained_net_param);
  caffe_test_net.CopyTrainedLayersFrom(trained_net_param);

  int total_iter = atoi(argv[3]);
  LOG(ERROR) << "Running " << total_iter << "Iterations.";

  double test_accuracy = 0;
  vector<Blob<float>*> dummy_blob_input_vec;
  for (int i = 0; i < total_iter; ++i) {
    const vector<Blob<float>*>& result =
        caffe_test_net.Forward(dummy_blob_input_vec);
    test_accuracy += result[0]->cpu_data()[0];
    LOG(ERROR) << "Batch " << i << ", accuracy: " << result[0]->cpu_data()[0];
  }
  test_accuracy /= total_iter;
  LOG(ERROR) << "Test accuracy:" << test_accuracy;

  return 0;
}
