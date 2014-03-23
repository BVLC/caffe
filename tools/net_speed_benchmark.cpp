// Copyright 2014 BVLC and contributors.

#include <cuda_runtime.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>

#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/solver.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  int total_iter = 50;
  if (argc < 2 || argc > 5) {
    LOG(ERROR) << "net_speed_benchmark net_proto [iterations=50]"
        " [CPU/GPU] [Device_id=0]";
    return 1;
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
  Net<float> caffe_net(argv[1]);

  // Run the network without training.
  LOG(ERROR) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  float initial_loss;
  caffe_net.Forward(vector<Blob<float>*>(), &initial_loss);
  LOG(ERROR) << "Initial loss: " << initial_loss;
  LOG(ERROR) << "Performing Backward";
  caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  LOG(ERROR) << "*** Benchmark begins ***";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  forward_timer.Start();
  Timer timer;
  for (int i = 0; i < layers.size(); ++i) {
    const string& layername = layers[i]->layer_param().name();
    timer.Start();
    for (int j = 0; j < total_iter; ++j) {
      layers[i]->Forward(bottom_vecs[i], &top_vecs[i]);
    }
    LOG(ERROR) << layername << "\tforward: " << timer.MilliSeconds() <<
        " milli seconds.";
  }
  LOG(ERROR) << "Forward pass: " << forward_timer.MilliSeconds() <<
      " milli seconds.";
  Timer backward_timer;
  backward_timer.Start();
  for (int i = layers.size() - 1; i >= 0; --i) {
    const string& layername = layers[i]->layer_param().name();
    timer.Start();
    for (int j = 0; j < total_iter; ++j) {
      layers[i]->Backward(top_vecs[i], true, &bottom_vecs[i]);
    }
    LOG(ERROR) << layername << "\tbackward: "
        << timer.MilliSeconds() << " milli seconds.";
  }
  LOG(ERROR) << "Backward pass: " << backward_timer.MilliSeconds() <<
      " milli seconds.";
  LOG(ERROR) << "Total Time: " << total_timer.MilliSeconds() <<
      " milli seconds.";
  LOG(ERROR) << "*** Benchmark ends ***";
  return 0;
}
