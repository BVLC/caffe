// Copyright 2013 Yangqing Jia

#include <cuda_runtime.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <gtest/gtest.h>

#include <cstring>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/optimization/solver.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class MNISTSolverTest : public ::testing::Test {};

typedef ::testing::Types<float> Dtypes;
TYPED_TEST_CASE(MNISTSolverTest, Dtypes);

TYPED_TEST(MNISTSolverTest, TestSolve) {
  Caffe::set_mode(Caffe::GPU);

  NetParameter net_param;
  ReadProtoFromTextFile("caffe/test/data/lenet.prototxt",
      &net_param);
  vector<Blob<TypeParam>*> bottom_vec;
  Net<TypeParam> caffe_net(net_param, bottom_vec);

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
  SGDSolver<TypeParam> solver(solver_param);
  solver.Solve(&caffe_net);
  LOG(ERROR) << "Optimization Done.";

  // Run the network after training.
  LOG(ERROR) << "Performing Forward";
  caffe_net.Forward(bottom_vec);
  LOG(ERROR) << "Performing Backward";
  TypeParam loss = caffe_net.Backward();
  LOG(ERROR) << "Final loss: " << loss;
  EXPECT_LE(loss, 0.5);

  NetParameter trained_net_param;
  caffe_net.ToProto(&trained_net_param);
  // LOG(ERROR) << "Writing to disk.";
  // WriteProtoToBinaryFile(trained_net_param,
  //     "caffe/test/data/lenet_trained.prototxt");

  NetParameter traintest_net_param;
  ReadProtoFromTextFile("caffe/test/data/lenet_traintest.prototxt",
      &traintest_net_param);
  Net<TypeParam> caffe_traintest_net(traintest_net_param, bottom_vec);
  caffe_traintest_net.CopyTrainedLayersFrom(trained_net_param);

  // Test run
  double train_accuracy = 0;
  int batch_size = traintest_net_param.layers(0).layer().batchsize();
  for (int i = 0; i < 60000 / batch_size; ++i) {
    const vector<Blob<TypeParam>*>& result =
        caffe_traintest_net.Forward(bottom_vec);
    train_accuracy += result[0]->cpu_data()[0];
  }
  train_accuracy /= 60000 / batch_size;
  LOG(ERROR) << "Train accuracy:" << train_accuracy;
  EXPECT_GE(train_accuracy, 0.98);

  NetParameter test_net_param;
  ReadProtoFromTextFile("caffe/test/data/lenet_test.prototxt", &test_net_param);
  Net<TypeParam> caffe_test_net(test_net_param, bottom_vec);
  caffe_test_net.CopyTrainedLayersFrom(trained_net_param);

  // Test run
  double test_accuracy = 0;
  batch_size = test_net_param.layers(0).layer().batchsize();
  for (int i = 0; i < 10000 / batch_size; ++i) {
    const vector<Blob<TypeParam>*>& result =
        caffe_test_net.Forward(bottom_vec);
    test_accuracy += result[0]->cpu_data()[0];
  }
  test_accuracy /= 10000 / batch_size;
  LOG(ERROR) << "Test accuracy:" << test_accuracy;
  EXPECT_GE(test_accuracy, 0.98);
}

}  // namespace caffe
