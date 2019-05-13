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

#ifndef CAFFE_UTIL_CPU_INFO_HPP
#define CAFFE_UTIL_CPU_INFO_HPP

#include <boost/thread/thread.hpp>
#if defined(_MSC_EXTENSIONS)
  #define NOMINMAX
  #include <windows.h>
  #include <malloc.h>    
  #include <tchar.h>
#else
  #include <sched.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>


namespace caffe {
namespace cpu {

struct Processor {
  unsigned processor;
  unsigned physicalId;
  unsigned siblings;
  unsigned coreId;
  unsigned cpuCores;
  unsigned speedMHz;
  size_t mask;
  Processor();
};

#if defined(_MSC_EXTENSIONS)
// 2 processor group should be enough for current h/w.
// But here we reserve 4 processor group for future extension.
struct cpu_set_t {
  size_t Mask[4];
};
#endif

class CpuInfoInterface {
 public:
  virtual ~CpuInfoInterface() {}
  virtual const char *getFirstLine() = 0;
  virtual const char *getNextLine() = 0;
};

class CpuInfo : public CpuInfoInterface {
 public:
  CpuInfo();
  explicit CpuInfo(const char *content);
  virtual ~CpuInfo();

  virtual const char *getFirstLine();
  virtual const char *getNextLine();

 private:
  const char *fileContentBegin;
  const char *fileContentEnd;
  const char *currentLine;

  void loadContentFromFile(const char *fileName);
  void loadContent(std::string &content);
  void parseLines(char *content);
};

class CollectionInterface {
 public:
  virtual ~CollectionInterface() {}
  virtual unsigned getProcessorSpeedMHz() = 0;
  virtual unsigned getTotalNumberOfSockets() = 0;
  virtual unsigned getTotalNumberOfCpuCores() = 0;
  virtual unsigned getNumberOfProcessors() = 0;
  virtual const Processor &getProcessor(unsigned processorId) = 0;
};

class Collection : public CollectionInterface {
 public:
  explicit Collection(CpuInfoInterface *cpuInfo);

  virtual unsigned getProcessorSpeedMHz();
  virtual unsigned getTotalNumberOfSockets();
  virtual unsigned getTotalNumberOfCpuCores();
  virtual unsigned getNumberOfProcessors();
  virtual const Processor &getProcessor(unsigned processorId);

 private:
#if !defined(_MSC_EXTENSIONS)
  void parseCpuInfoLine(const char *cpuInfoLine);
  void parseValue(const char *fieldName, const char *valueString);
  void appendNewProcessor();
  bool beginsWith(const char *lineBuffer, const char *text) const;
  unsigned parseInteger(const char *text) const;
  unsigned extractSpeedFromModelName(const char *text) const;
#endif

  void collectBasicCpuInformation();
  void updateCpuInformation(const Processor &processor,
    unsigned numberOfUniquePhysicalId);

  CpuInfoInterface &cpuInfo;
  unsigned totalNumberOfSockets;
  unsigned totalNumberOfCpuCores;
  std::vector<Processor> processors;
  Processor *currentProcessor;

  Collection(const Collection &collection);
  Collection &operator =(const Collection &collection);

  void parseCpuInfo();
};

#ifdef _OPENMP

class OpenMpManager {
 public:
  static void setGpuEnabled();
  static void setGpuDisabled();

  static void bindCurrentThreadToNonPrimaryCoreIfPossible();

  static void bindOpenMpThreads();
  static void printVerboseInformation();

  static bool isMajorThread(boost::thread::id currentThread);
  static unsigned getProcessorSpeedMHz();
  static unsigned getNumaNode();

 private:
  boost::thread::id mainThreadId;
  Collection &collection;

  bool isGpuEnabled;
  bool isAnyOpenMpEnvVarSpecified;
  cpu_set_t currentCpuSet;
  cpu_set_t currentCoreSet;

  explicit OpenMpManager(Collection *collection);
  OpenMpManager(const OpenMpManager &openMpManager);
  OpenMpManager &operator =(const OpenMpManager &openMpManager);
  static OpenMpManager &getInstance();

  void getOpenMpEnvVars();
  void getCurrentCpuSet();
  void getDefaultCpuSet(cpu_set_t *defaultCpuSet);
  void getCurrentCoreSet();

  void selectAllCoreCpus(cpu_set_t *set, unsigned physicalCoreId);
  unsigned getPhysicalCoreId(unsigned logicalCoreId);

  bool isThreadsBindAllowed();
  void setOpenMpThreadNumberLimit();
  void bindCurrentThreadToLogicalCoreCpu(unsigned logicalCoreId);
  void bindCurrentThreadToLogicalCoreCpus(unsigned logicalCoreId);
};

#endif  // _OPENMP

}  // namespace cpu

}  // namespace caffe

#endif  // CAFFE_UTIL_CPU_INFO_HPP
