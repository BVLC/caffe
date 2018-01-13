/*
All modification made by Intel Corporation: © 2018 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md

A part of the code referenced BVLC CAFFE ristretto branch
For the original code go to https://github.com/pmgysel/caffe 

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

#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/upgrade_proto.hpp"

#include "caffe/quant/quantization.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(weights, "",
    "The trained weights.");
DEFINE_string(trimming_mode, "",
    "Available options: dynamic_fixed_point.");
DEFINE_string(model_quantized, "",
    "The output path of the quantized net");
DEFINE_int32(iterations, 50,
    "Optional: The number of iterations to run.");
DEFINE_double(error_margin, 2,
    "Optional: the allowed accuracy drop in %");
DEFINE_int32(score_number, 0,
    "Optional: The score number to run.");
DEFINE_string(scaling, "single",
    "The scaling model: single or multiple");
DEFINE_int32(detection, 0,
    "Optional: classfication or object detection");
DEFINE_int32(power, 1,
    "Optional: power of two or floating");

// A simple registry for caffe commands.

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available sampling actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Quantize FP32 to INT8
int quantize(){
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
  CHECK_GT(FLAGS_model_quantized.size(), 0) << "Need network description "
      "output path.";
  CHECK_GT(FLAGS_trimming_mode.size(), 0) << "Need trimming mode.";
  Quantization* q = new Quantization(FLAGS_model, FLAGS_weights,
      FLAGS_model_quantized, FLAGS_iterations, FLAGS_trimming_mode,
      FLAGS_error_margin, FLAGS_score_number, FLAGS_scaling, FLAGS_detection, FLAGS_power);
  q->QuantizeNet();
  delete q;
  return 0;
}
RegisterBrewFunction(quantize);

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Set version
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: sample <command> <args>\n\n"
      "commands:\n"
      "  quantize        Trim 32bit floating point net\n");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2) {
      return GetBrewFunction(caffe::string(argv[1]))();
  } else {
      gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/sample");
  }
}
