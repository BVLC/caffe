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
class SolverTest : public ::testing::Test {};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(SolverTest, Dtypes);

TYPED_TEST(SolverTest, TestSolve) {
  Caffe::set_mode(Caffe::GPU);

  NetParameter net_param;
  ReadProtoFromTextFile("caffe/test/data/linear_regression.prototxt",
      &net_param);
  // check if things are right
  EXPECT_EQ(net_param.layers_size(), 3);
  EXPECT_EQ(net_param.input_size(), 0);
  vector<Blob<TypeParam>*> bottom_vec;
  Net<TypeParam> caffe_net(net_param, bottom_vec);
  EXPECT_EQ(caffe_net.layer_names().size(), 3);
  EXPECT_EQ(caffe_net.blob_names().size(), 3);

  // Run the network without training.
  LOG(ERROR) << "Performing Forward";
  caffe_net.Forward(bottom_vec);
  LOG(ERROR) << "Performing Backward";
  LOG(ERROR) << "Initial loss: " << caffe_net.Backward();

  SolverParameter solver_param;
  solver_param.set_base_lr(0.1);
  solver_param.set_display(0);
  solver_param.set_max_iter(100);
  solver_param.set_lr_policy("inv");
  solver_param.set_gamma(1.);
  solver_param.set_power(0.75);
  solver_param.set_momentum(0.9);

  LOG(ERROR) << "Starting Optimization";
  SGDSolver<TypeParam> solver(solver_param);
  solver.Solve(&caffe_net);
  LOG(ERROR) << "Optimization Done.";
  LOG(ERROR) << "Weight: " << caffe_net.params()[0]->cpu_data()[0] << ", "
      << caffe_net.params()[0]->cpu_data()[1];
  LOG(ERROR) << "Bias: " << caffe_net.params()[1]->cpu_data()[0];

  EXPECT_GE(caffe_net.params()[0]->cpu_data()[0], 0.3);
  EXPECT_LE(caffe_net.params()[0]->cpu_data()[0], 0.35);

  EXPECT_GE(caffe_net.params()[0]->cpu_data()[1], 0.3);
  EXPECT_LE(caffe_net.params()[0]->cpu_data()[1], 0.35);

  EXPECT_GE(caffe_net.params()[1]->cpu_data()[0], -0.01);
  EXPECT_LE(caffe_net.params()[1]->cpu_data()[0], 0.01);
}

}  // namespace caffe
