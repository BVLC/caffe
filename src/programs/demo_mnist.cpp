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
  ReadProtoFromTextFile("caffe/test/data/lenet.prototxt",
      &net_param);
  vector<Blob<float>*> bottom_vec;
  Net<float> caffe_net(net_param, bottom_vec);

  // Run the network without training.
  LOG(ERROR) << "Performing Forward";
  caffe_net.Forward(bottom_vec);
  LOG(ERROR) << "Performing Backward";
  LOG(ERROR) << "Initial loss: " << caffe_net.Backward();

  SolverParameter solver_param;
  solver_param.set_base_lr(0.01);
  solver_param.set_display(0);
  solver_param.set_max_iter(6000);
  solver_param.set_lr_policy("inv");
  solver_param.set_gamma(0.0001);
  solver_param.set_power(0.75);
  solver_param.set_momentum(0.9);

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

  NetParameter trained_net_param;
  caffe_net.ToProto(&trained_net_param);

  NetParameter traintest_net_param;
  ReadProtoFromTextFile("caffe/test/data/lenet_traintest.prototxt",
      &traintest_net_param);
  Net<float> caffe_traintest_net(traintest_net_param, bottom_vec);
  caffe_traintest_net.CopyTrainedLayersFrom(trained_net_param);

  // Test run
  double train_accuracy = 0;
  int batch_size = traintest_net_param.layers(0).layer().batchsize();
  for (int i = 0; i < 60000 / batch_size; ++i) {
    const vector<Blob<float>*>& result =
        caffe_traintest_net.Forward(bottom_vec);
    train_accuracy += result[0]->cpu_data()[0];
  }
  train_accuracy /= 60000 / batch_size;
  LOG(ERROR) << "Train accuracy:" << train_accuracy;

  NetParameter test_net_param;
  ReadProtoFromTextFile("caffe/test/data/lenet_test.prototxt", &test_net_param);
  Net<float> caffe_test_net(test_net_param, bottom_vec);
  caffe_test_net.CopyTrainedLayersFrom(trained_net_param);

  // Test run
  double test_accuracy = 0;
  batch_size = test_net_param.layers(0).layer().batchsize();
  for (int i = 0; i < 10000 / batch_size; ++i) {
    const vector<Blob<float>*>& result =
        caffe_test_net.Forward(bottom_vec);
    test_accuracy += result[0]->cpu_data()[0];
  }
  test_accuracy /= 10000 / batch_size;
  LOG(ERROR) << "Test accuracy:" << test_accuracy;

  return 0;
}
