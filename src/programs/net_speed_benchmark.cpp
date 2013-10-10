// Copyright 2013 Yangqing Jia

#include <cuda_runtime.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>

#include <cstring>
#include <ctime>

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
  Caffe::set_phase(Caffe::TRAIN);
  int repeat = 100;

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

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  LOG(ERROR) << "*** Benchmark begins ***";
  for (int i = 0; i < layers.size(); ++i) {
    const string& layername = layers[i]->layer_param().name();
    clock_t start = clock();
    for (int j = 0; j < repeat; ++j) {
      layers[i]->Forward(bottom_vecs[i], &top_vecs[i]);
    }
    LOG(ERROR) << layername << "\tforward: "
        << float(clock() - start) / CLOCKS_PER_SEC << " seconds.";
  }
  for (int i = layers.size() - 1; i >= 0; --i) {
    const string& layername = layers[i]->layer_param().name();
    clock_t start = clock();
    for (int j = 0; j < repeat; ++j) {
      layers[i]->Backward(top_vecs[i], true, &bottom_vecs[i]);
    }
    LOG(ERROR) << layername << "\tbackward: "
        << float(clock() - start) / CLOCKS_PER_SEC << " seconds.";
  }
  LOG(ERROR) << "*** Benchmark ends ***";
  return 0;
}
