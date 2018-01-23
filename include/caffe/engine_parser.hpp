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

#ifndef CAFFE_MKLDNN_ENGINES_HPP_
#define CAFFE_MKLDNN_ENGINES_HPP_

#include <glog/logging.h>
#include <cstring>
#include <string>
#include <vector>

#ifdef MKLDNN_SUPPORTED
#include "caffe/mkldnn_base.hpp"
#endif

namespace caffe {
static const char* supportedEngines[] =
    {"CAFFE", "CUDNN", "MKL2017", "MKLDNN"};
class EngineParser {
 public:
  explicit EngineParser(const std::string subEngineString) {
    parse(subEngineString.c_str());
    // Check for wrong engine name
    validateEngine();
  }

  bool isEngine(const char* name) const {
    return (engineName == name);
  }

  unsigned getNumberOfSubEngines() const {
    return subEngines.size();
  }

#ifdef MKLDNN_SUPPORTED
  engine& getMKLDNNSubEngine(unsigned engineIndex) const {
    CHECK(engineIndex < getNumberOfSubEngines());
    const char *engineName = subEngines[engineIndex].c_str();

    if (!strcmp(engineName, "CPU"))
      return CpuEngine::Instance().get_engine();

#ifdef FPGA_ENABLED
    if (!strcmp(engineName, "FPGA"))
      return FPGAEngine::Instance().get_engine();
#endif

#ifdef DLA_ENABLED
    if (!strcmp(engineName, "DLA"))
      return DLAEngine::Instance().get_engine();
#endif

    LOG(FATAL) << "EngineParser: Unknown subengine: " << engineName;
    // should never be here. it's used to eliminate a build warning #1011: missing return statement at end of non-void function.
    return CpuEngine::Instance().get_engine(); 
  }
#endif

 private:
  std::string engineName;
  std::vector<std::string> subEngines;

  bool parse(const char *subEngineString) {
    // Ignore whitespaces
    subEngineString = parseWhitespaces(subEngineString);

    // Extract engine identifier. It can be empty at this point
    const char *beginOfIdentifier = subEngineString;
    subEngineString = parseIdentifier(subEngineString);
    engineName.assign(beginOfIdentifier, subEngineString - beginOfIdentifier);

    // Ignore whitespaces
    subEngineString = parseWhitespaces(subEngineString);

    // String termination is allowed at this place
    if (!*subEngineString)
        return true;

    // Otherwise colon must be specified and engine identifier cannot be empty
    if (!engineName.length() ||  (*subEngineString != ':')
            ||  (*(subEngineString+1) == '\0'))
        LOG(FATAL) << "Wrong engine specification";

    // Process sub engines
    subEngineString++;
    while (true) {
      // Ignore separators
      subEngineString = parseSeparators(subEngineString);

      // String termination is allowed at this place
      if (!*subEngineString)
          return true;

      // Extract sub engine identifier
      const char *beginOfIdentifier = subEngineString;
      subEngineString = parseIdentifier(subEngineString);

      // Identifier can not be empty nor contain invalid characters
      if (beginOfIdentifier == subEngineString)
          return false;

      // Collect all valid sub engine names
      std::string subEngineName;
      subEngineName.assign(beginOfIdentifier,
              subEngineString - beginOfIdentifier);
      subEngines.push_back(subEngineName);
    }
  }

  void validateEngine() {
#ifndef USE_CUDNN
    if (engineName == "CUDNN")
        LOG(FATAL) << "Support for CUDNN is not enabled";
#endif
#ifndef MKL2017_SUPPORTED
    if (engineName == "MKL2017")
        LOG(FATAL) << "Support for MKL2017 is not enabled";
#endif
#ifndef MKLDNN_SUPPORTED
    if (engineName == "MKLDNN")
        LOG(FATAL) << "Support for MKLDNN is not enabled";
#endif
    for (unsigned i = 0;
         i < sizeof(supportedEngines)/sizeof(supportedEngines[0]); i++ )
        if (supportedEngines[i] == engineName) {
            if (subEngines.size() > 0 && engineName != "MKLDNN")
              LOG(FATAL) << "Engine " << engineName
                         << " does not support subengines";
            return;
        }
    LOG(FATAL) << "Unknown engine: " << engineName;
  }

  const char *parseWhitespaces(const char *subEngineString) const {
    while (isspace(*subEngineString))
      subEngineString++;

    return subEngineString;
  }

  const char *parseSeparators(const char *subEngineString) const {
    while (isspace(*subEngineString) || (*subEngineString == ',')
            || (*subEngineString == ';'))
      subEngineString++;

    return subEngineString;
  }

  const char *parseIdentifier(const char *subEngineString) const {
    if (!isalpha(*subEngineString) && (*subEngineString != '_'))
      return subEngineString;

    do {
      subEngineString++;
    } while (isalnum(*subEngineString) || (*subEngineString == '_'));

    return subEngineString;
  }
};
}  // namespace caffe
#endif
