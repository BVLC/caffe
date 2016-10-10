/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/sgd_solvers.hpp"
#include "caffe/solver.hpp"

#include "caffe/test/test_caffe_main.hpp"

using std::ostringstream;

namespace caffe {

template <typename TypeParam>
class SolverTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  virtual void InitSolverFromProtoString(const string& proto) {
    SolverParameter param;
    CHECK(google::protobuf::TextFormat::ParseFromString(proto, &param));
    // Set the solver_mode according to current Caffe::mode.
    switch (Caffe::mode()) {
      case Caffe::CPU:
        param.set_solver_mode(SolverParameter_SolverMode_CPU);
        break;
      case Caffe::GPU:
        param.set_solver_mode(SolverParameter_SolverMode_GPU);
        break;
      default:
        LOG(FATAL) << "Unknown Caffe mode: " << Caffe::mode();
    }
    solver_.reset(new SGDSolver<Dtype>(param));
  }

  shared_ptr<Solver<Dtype> > solver_;
};

TYPED_TEST_CASE(SolverTest, TestDtypesAndDevices);

TYPED_TEST(SolverTest, TestInitTrainTestNets) {
  const string& proto =
     "test_interval: 10 "
     "test_iter: 10 "
     "test_state: { stage: 'with-softmax' }"
     "test_iter: 10 "
     "test_state: {}"
     "net_param { "
     "  name: 'TestNetwork' "
     "  layer { "
     "    name: 'data' "
     "    type: 'DummyData' "
     "    dummy_data_param { "
     "      shape { "
     "        dim: 5 "
     "        dim: 2 "
     "        dim: 3 "
     "        dim: 4 "
     "      } "
     "      shape { "
     "        dim: 5 "
     "      } "
     "    } "
     "    top: 'data' "
     "    top: 'label' "
     "  } "
     "  layer { "
     "    name: 'innerprod' "
     "    type: 'InnerProduct' "
     "    inner_product_param { "
     "      num_output: 10 "
     "    } "
     "    bottom: 'data' "
     "    top: 'innerprod' "
     "  } "
     "  layer { "
     "    name: 'accuracy' "
     "    type: 'Accuracy' "
     "    bottom: 'innerprod' "
     "    bottom: 'label' "
     "    top: 'accuracy' "
     "    exclude: { phase: TRAIN } "
     "  } "
     "  layer { "
     "    name: 'loss' "
     "    type: 'SoftmaxWithLoss' "
     "    bottom: 'innerprod' "
     "    bottom: 'label' "
     "    include: { phase: TRAIN } "
     "    include: { phase: TEST stage: 'with-softmax' } "
     "  } "
     "} ";
  this->InitSolverFromProtoString(proto);
  ASSERT_TRUE(this->solver_->net() != NULL);
  EXPECT_TRUE(this->solver_->net()->has_layer("loss"));
  EXPECT_FALSE(this->solver_->net()->has_layer("accuracy"));
  ASSERT_EQ(2, this->solver_->test_nets().size());
  EXPECT_TRUE(this->solver_->test_nets()[0]->has_layer("loss"));
  EXPECT_TRUE(this->solver_->test_nets()[0]->has_layer("accuracy"));
  EXPECT_FALSE(this->solver_->test_nets()[1]->has_layer("loss"));
  EXPECT_TRUE(this->solver_->test_nets()[1]->has_layer("accuracy"));
}

}  // namespace caffe
