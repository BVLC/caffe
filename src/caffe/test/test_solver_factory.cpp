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

#include <map>
#include <string>

#include "boost/scoped_ptr.hpp"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/solver.hpp"
#include "caffe/solver_factory.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class SolverFactoryTest : public MultiDeviceTest<TypeParam> {
 protected:
  SolverParameter simple_solver_param() {
    const string solver_proto =
        "train_net_param { "
        "  layer { "
        "    name: 'data' type: 'DummyData' top: 'data' "
        "    dummy_data_param { shape { dim: 1 } } "
        "  } "
        "} ";
    SolverParameter solver_param;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        solver_proto, &solver_param));
    return solver_param;
  }
};

TYPED_TEST_CASE(SolverFactoryTest, TestDtypesAndDevices);

TYPED_TEST(SolverFactoryTest, TestCreateSolver) {
  typedef typename TypeParam::Dtype Dtype;
  typename SolverRegistry<Dtype>::CreatorRegistry& registry =
      SolverRegistry<Dtype>::Registry();
  shared_ptr<Solver<Dtype> > solver;
  SolverParameter solver_param = this->simple_solver_param();
  for (typename SolverRegistry<Dtype>::CreatorRegistry::iterator iter =
       registry.begin(); iter != registry.end(); ++iter) {
    solver_param.set_type(iter->first);
    solver.reset(SolverRegistry<Dtype>::CreateSolver(solver_param));
    EXPECT_EQ(iter->first, solver->type());
  }
}

}  // namespace caffe
