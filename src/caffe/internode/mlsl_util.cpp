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

#ifdef USE_MLSL

#include <glog/logging.h>
#include <stdlib.h>
#include <cassert>
#include <stdexcept>
#include <string>
#include "caffe/internode/mlsl_util.hpp"

namespace caffe {
namespace internode {

MLSL::Distribution *data_parallelism;
MLSL::Distribution *model_parallelism;
//MLSL::Distribution *hybrid_parallelism;
bool isDistributionsInited = false;

void mlsl_init(int argc, char** argv) {
  LOG(INFO) << "MLSL init";
  MLSL::Init(&argc, &argv);
}

void mlsl_finalize() {
  LOG(INFO) << "MLSL finalize";

  if (isDistributionsInited) {
      isDistributionsInited = false;
      delete data_parallelism;
      delete model_parallelism;
      //delete hybrid_parallelism;
  }

  MLSL::Finalize();
}

void mlsl_init_distributions() {

    if (!isDistributionsInited) {
        isDistributionsInited = true;
        data_parallelism = new MLSL::Distribution(MLSL::GetNumNodes(), 1);
        model_parallelism = new MLSL::Distribution(1, MLSL::GetNumNodes());
    }
}

}  // namespace internode
}  // namespace caffe

#endif /* USE_MLSL */

