// Copyright 2013 Yangqing Jia
// This example shows how to run a modified version of LeNet using Caffe.

#include <cuda_runtime.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>

#include <cstring>
#include <iostream>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/solver.hpp"

using namespace caffe;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cout << "Usage:" << std::endl;
    std::cout << "demo_mnist.bin train_file test_file [CPU/GPU]" << std::endl;
    return 0;
  }
  google::InitGoogleLogging(argv[0]);
  Caffe::DeviceQuery();

  if (argc == 4 && strcmp(argv[3], "GPU") == 0) {
    LOG(ERROR) << "Using GPU";
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  // Start training
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

  SolverParameter solver_param;
  // Solver Parameters are hard-coded in this case, but you can write a
  // SolverParameter protocol buffer to specify all these values.
  solver_param.set_base_lr(0.001);
  solver_param.set_display(100);
  solver_param.set_max_iter(5000);
  solver_param.set_lr_policy("inv");
  solver_param.set_gamma(0.0001);
  solver_param.set_power(0.75);
  solver_param.set_momentum(0.9);
  solver_param.set_weight_decay(0.0005);

  LOG(ERROR) << "Starting Optimization";
  SGDSolver<float> solver(solver_param);
  solver.Solve(&caffe_net);
  LOG(ERROR) << "Optimization Done.";

  // Write the trained network to a NetParameter protobuf. If you are training
  // the model and saving it for later, this is what you want to serialize and
  // store.
  NetParameter trained_net_param;
  caffe_net.ToProto(&trained_net_param);

  // Now, let's starting doing testing.
  Caffe::set_phase(Caffe::TEST);

  // Using the testing data to test the accuracy.
  NetParameter test_net_param;
  ReadProtoFromTextFile(argv[2], &test_net_param);
  Net<float> caffe_test_net(test_net_param, bottom_vec);
  caffe_test_net.CopyTrainedLayersFrom(trained_net_param);

  double test_accuracy = 0;
  int batch_size = test_net_param.layers(0).layer().batchsize();
  for (int i = 0; i < 10000 / batch_size; ++i) {
    const vector<Blob<float>*>& result =
        caffe_test_net.Forward(bottom_vec);
    test_accuracy += result[0]->cpu_data()[0];
  }
  test_accuracy /= 10000 / batch_size;
  LOG(ERROR) << "Test accuracy:" << test_accuracy;

  return 0;
}
