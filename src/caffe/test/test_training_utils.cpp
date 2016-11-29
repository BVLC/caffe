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

#include "boost/algorithm/string.hpp"
#include "boost/lexical_cast.hpp"

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

#include "caffe/solver.hpp"
#include "caffe/test/test_caffe_main.hpp"

#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

#include "caffe/training_utils.hpp"

namespace caffe {

class TrainingUtilsTest : public ::testing::Test {
 protected:
  TrainingUtilsTest()
    : topology_file_name_(
    CMAKE_SOURCE_DIR "caffe/test/test_data/test_topology.prototxt") {
    }

  string topology_file_name_;
};

TEST_F(TrainingUtilsTest, GetStagesFromFlags) {
  string stages_flag = "stage1,stage2, stage3";

  EXPECT_EQ(get_stages_from_flags(stages_flag).size(), 3);
}

TEST_F(TrainingUtilsTest, SetSolverParamsFromFlags) {
  caffe::SolverParameter solver_param;
  string solver;
  string engine = "new engine";
  string stages = "stage1,stage2";
  int level = 1;

  use_flags(&solver_param, solver, engine, level, stages);

  EXPECT_EQ(solver_param.engine(), engine);
  EXPECT_EQ(solver_param.train_state().stage_size(), 2);
  EXPECT_EQ(solver_param.train_state().level(), 1);
}

TEST_F(TrainingUtilsTest, MultiphaseTrainNegativeCpuMode) {
  caffe::MultiPhaseSolverParameter multi_solver_param;
  caffe::SolverBatchSizePair* solver_param_pair =
    multi_solver_param.add_params_pair();

  solver_param_pair->mutable_solver_params()->set_solver_mode(
    caffe::SolverParameter_SolverMode_GPU);

  solver_param_pair->mutable_solver_params()->set_net(
    this->topology_file_name_);
  EXPECT_EQ(multiphase_train(&multi_solver_param, "", "", 0, ""), -1);
}

}  // namespace caffe
