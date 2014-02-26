// Copyright 2013 Yangqing Jia

#include <ctime>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "fcntl.h"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/solver.hpp"

using boost::shared_ptr;

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  int total_iter = 50;

  if (argc < 2) {
    LOG(ERROR) << "net_speed_benchmark net_proto [iterations=50] [CPU/GPU] "
        << "[Device_id=0]";
    return 0;
  }

  if (argc >=3) {
    total_iter = atoi(argv[2]);
  }

  LOG(ERROR) << "Testing for " << total_iter << "Iterations.";

  if (argc >= 4 && strcmp(argv[3], "GPU") == 0) {
    LOG(ERROR) << "Using GPU";
    uint device_id = 0;
    if (argc >= 5 && strcmp(argv[3], "GPU") == 0) {
      device_id = atoi(argv[4]);
    }
    LOG(ERROR) << "Using Device_id=" << device_id;
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  Caffe::set_phase(Caffe::TRAIN);
  NetParameter net_param;
  ReadProtoFromTextFile(argv[1],
      &net_param);
  Net<float> caffe_net(net_param);

  // Run the network without training.
  LOG(ERROR) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  caffe_net.Forward(vector<Blob<float>*>());
  LOG(ERROR) << "Performing Backward";
  LOG(ERROR) << "Initial loss: " << caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  LOG(ERROR) << "*** Benchmark begins ***";
  clock_t forward_start = clock();
  for (int i = 0; i < layers.size(); ++i) {
    const string& layername = layers[i]->layer_param().name();
    clock_t start = clock();
    for (int j = 0; j < total_iter; ++j) {
      layers[i]->Forward(bottom_vecs[i], &top_vecs[i]);
    }
    LOG(ERROR) << layername << "\tforward: "
        << static_cast<float>(clock() - start) / CLOCKS_PER_SEC
        << " seconds.";
  }
  LOG(ERROR) << "Forward pass: "
      << static_cast<float>(clock() - forward_start) / CLOCKS_PER_SEC
      << " seconds.";
  clock_t backward_start = clock();
  for (int i = layers.size() - 1; i >= 0; --i) {
    const string& layername = layers[i]->layer_param().name();
    clock_t start = clock();
    for (int j = 0; j < total_iter; ++j) {
      layers[i]->Backward(top_vecs[i], true, &bottom_vecs[i]);
    }
    LOG(ERROR) << layername << "\tbackward: "
        << static_cast<float>(clock() - start) / CLOCKS_PER_SEC
        << " seconds.";
  }
  LOG(ERROR) << "Backward pass: "
      << static_cast<float>(clock() - backward_start) / CLOCKS_PER_SEC
      << " seconds.";
  LOG(ERROR) << "Total Time: "
      << static_cast<float>(clock() - forward_start) / CLOCKS_PER_SEC
      << " seconds.";
  LOG(ERROR) << "*** Benchmark ends ***";
  return 0;
}
