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

#include "gtest/gtest.h"

#include "caffe/util/cpu_info.hpp"

namespace caffe {
namespace cpu {

class CpuInfoContent {
 public:
  CpuInfoContent(const char *modelName,
    int numberOfSockets, int coresPerSocket, int threadsPerCore) {

    const int contentLength = 1 * 1024 * 1024;
    content = new char[contentLength];

    char *contentPosition = content;
    char *contentEnd = &content[contentLength];

    int processorId = 0;
    for (int socketId = 0; socketId < numberOfSockets; socketId++) {
      for (int threadId = 0; threadId < threadsPerCore; threadId++) {
        for (int coreId = 0; coreId < coresPerSocket; coreId++) {
          contentPosition += snprintf(
            contentPosition,
            contentEnd - contentPosition,
            "processor       : %i\n"
            "model name      : %s\n"
            "physical id     : %i\n"
            "siblings        : %i\n"
            "core id         : %i\n"
            "cpu cores       : %i\n"
            "\n",
            processorId++,
            modelName,
            socketId,
            coresPerSocket * threadsPerCore,
            coreId,
            coresPerSocket);
        }
      }
    }
  }

  ~CpuInfoContent() {
    delete [] content;
  }

  const char *getContent() const {
    return content;
  }

 private:
  char *content;
};

TEST(CpuInfo, isProcessorStructureInitialized) {
  Processor processor;
  EXPECT_EQ(processor.processor, 0);
  EXPECT_EQ(processor.physicalId, 0);
  EXPECT_EQ(processor.siblings, 0);
  EXPECT_EQ(processor.coreId, 0);
  EXPECT_EQ(processor.cpuCores, 0);
  EXPECT_EQ(processor.speedMHz, 0);
}

TEST(CpuInfo, testCpuInfoForEmptyInput) {
  CpuInfo cpuInfo("");
  EXPECT_STREQ(cpuInfo.getFirstLine(), NULL);
  EXPECT_STREQ(cpuInfo.getNextLine(), NULL);
  EXPECT_STREQ(cpuInfo.getNextLine(), NULL);
}

TEST(CpuInfo, testCpuInfoForSingleCharacterInput) {
  CpuInfo cpuInfo("c");
  EXPECT_STREQ(cpuInfo.getFirstLine(), "c");
  EXPECT_STREQ(cpuInfo.getNextLine(), NULL);
  EXPECT_STREQ(cpuInfo.getNextLine(), NULL);
}

TEST(CpuInfo, testCpuInfoForSingleLineInput) {
  CpuInfo cpuInfo("First line");
  EXPECT_STREQ(cpuInfo.getFirstLine(), "First line");
  EXPECT_STREQ(cpuInfo.getNextLine(), NULL);
  EXPECT_STREQ(cpuInfo.getNextLine(), NULL);
}

TEST(CpuInfo, testCpuInfoForMultiLineInput) {
  CpuInfo cpuInfo("First line\nSecond line\nThird line");
  EXPECT_STREQ(cpuInfo.getFirstLine(), "First line");
  EXPECT_STREQ(cpuInfo.getNextLine(), "Second line");
  EXPECT_STREQ(cpuInfo.getNextLine(), "Third line");
  EXPECT_STREQ(cpuInfo.getNextLine(), NULL);
  EXPECT_STREQ(cpuInfo.getNextLine(), NULL);
}

TEST(CpuInfo, testCpuInfoForEmptyLinesInput) {
  CpuInfo cpuInfo("\nSecond line\nThird line\n\nFifth line\n\n");
  EXPECT_STREQ(cpuInfo.getFirstLine(), "");
  EXPECT_STREQ(cpuInfo.getNextLine(), "Second line");
  EXPECT_STREQ(cpuInfo.getNextLine(), "Third line");
  EXPECT_STREQ(cpuInfo.getNextLine(), "");
  EXPECT_STREQ(cpuInfo.getNextLine(), "Fifth line");
  EXPECT_STREQ(cpuInfo.getNextLine(), "");
  EXPECT_STREQ(cpuInfo.getNextLine(), NULL);
  EXPECT_STREQ(cpuInfo.getNextLine(), NULL);
}

TEST(CpuInfo, testCollectionForEmptyInput) {
  CpuInfo cpuInfo("");
  Collection collection(&cpuInfo);
  EXPECT_EQ(collection.getProcessorSpeedMHz(), 0);
  EXPECT_EQ(collection.getTotalNumberOfSockets(), 0);
  EXPECT_EQ(collection.getTotalNumberOfCpuCores(), 0);
  EXPECT_EQ(collection.getNumberOfProcessors(), 0);
}

TEST(CpuInfo, testCollectionForSingleSocketSingleCoreSingleThread) {
  CpuInfoContent cpuInfoContent("xxx", 1, 1, 1);
  CpuInfo cpuInfo(cpuInfoContent.getContent());
  Collection collection(&cpuInfo);
  EXPECT_EQ(collection.getProcessorSpeedMHz(), 0);
  EXPECT_EQ(collection.getTotalNumberOfSockets(), 1);
  EXPECT_EQ(collection.getTotalNumberOfCpuCores(), 1);
  EXPECT_EQ(collection.getNumberOfProcessors(), 1);
}

TEST(CpuInfo, testCollectionForMultipleSockets) {
  CpuInfoContent cpuInfoContent("xxx", 4, 1, 1);
  CpuInfo cpuInfo(cpuInfoContent.getContent());
  Collection collection(&cpuInfo);
  EXPECT_EQ(collection.getProcessorSpeedMHz(), 0);
  EXPECT_EQ(collection.getTotalNumberOfSockets(), 4);
  EXPECT_EQ(collection.getTotalNumberOfCpuCores(), 4);
  EXPECT_EQ(collection.getNumberOfProcessors(), 4);
}

TEST(CpuInfo, testCollectionForMultipleCores) {
  CpuInfoContent cpuInfoContent("xxx", 1, 8, 1);
  CpuInfo cpuInfo(cpuInfoContent.getContent());
  Collection collection(&cpuInfo);
  EXPECT_EQ(collection.getProcessorSpeedMHz(), 0);
  EXPECT_EQ(collection.getTotalNumberOfSockets(), 1);
  EXPECT_EQ(collection.getTotalNumberOfCpuCores(), 8);
  EXPECT_EQ(collection.getNumberOfProcessors(), 8);
}

TEST(CpuInfo, testCollectionForMultithreading) {
  CpuInfoContent cpuInfoContent("xxx", 1, 1, 2);
  CpuInfo cpuInfo(cpuInfoContent.getContent());
  Collection collection(&cpuInfo);
  EXPECT_EQ(collection.getProcessorSpeedMHz(), 0);
  EXPECT_EQ(collection.getTotalNumberOfSockets(), 1);
  EXPECT_EQ(collection.getTotalNumberOfCpuCores(), 1);
  EXPECT_EQ(collection.getNumberOfProcessors(), 2);
}

TEST(CpuInfo, testCollectionForMultipleCoresWithMultithreading) {
  CpuInfoContent cpuInfoContent("xxx", 1, 4, 2);
  CpuInfo cpuInfo(cpuInfoContent.getContent());
  Collection collection(&cpuInfo);
  EXPECT_EQ(collection.getProcessorSpeedMHz(), 0);
  EXPECT_EQ(collection.getTotalNumberOfSockets(), 1);
  EXPECT_EQ(collection.getTotalNumberOfCpuCores(), 4);
  EXPECT_EQ(collection.getNumberOfProcessors(), 8);
}

TEST(CpuInfo, testCollectionForMultipleSocketsMultipleCoresWithMultithreading) {
  CpuInfoContent cpuInfoContent("xxx", 2, 18, 2);
  CpuInfo cpuInfo(cpuInfoContent.getContent());
  Collection collection(&cpuInfo);
  EXPECT_EQ(collection.getProcessorSpeedMHz(), 0);
  EXPECT_EQ(collection.getTotalNumberOfSockets(), 2);
  EXPECT_EQ(collection.getTotalNumberOfCpuCores(), 36);
  EXPECT_EQ(collection.getNumberOfProcessors(), 72);
}

TEST(CpuInfo, testCollectionForSpeed) {
  CpuInfoContent cpuInfoContent("Intel(R) Xeon(R) CPU E5-2699 v4 @ 2.20GHz",
                                 2,
                                 22,
                                 2);
  CpuInfo cpuInfo(cpuInfoContent.getContent());
  Collection collection(&cpuInfo);
  EXPECT_EQ(collection.getProcessorSpeedMHz(), 2200);
  EXPECT_EQ(collection.getTotalNumberOfSockets(), 2);
  EXPECT_EQ(collection.getTotalNumberOfCpuCores(), 44);
  EXPECT_EQ(collection.getNumberOfProcessors(), 88);
}

TEST(CpuInfo, testCollectionForSpeedInGhz1) {
  CpuInfoContent cpuInfoContent("xxx @ 4.80GHz", 1, 1, 1);
  CpuInfo cpuInfo(cpuInfoContent.getContent());
  Collection collection(&cpuInfo);
  EXPECT_EQ(collection.getProcessorSpeedMHz(), 4800);
}

TEST(CpuInfo, testCollectionForSpeedInGhz2) {
  CpuInfoContent cpuInfoContent("xxx @ 400 GHz", 1, 1, 1);
  CpuInfo cpuInfo(cpuInfoContent.getContent());
  Collection collection(&cpuInfo);
  EXPECT_EQ(collection.getProcessorSpeedMHz(), 400000);
}

TEST(CpuInfo, testCollectionForSpeedInMhz1) {
  CpuInfoContent cpuInfoContent("xxx @ 400 MHz", 1, 1, 1);
  CpuInfo cpuInfo(cpuInfoContent.getContent());
  Collection collection(&cpuInfo);
  EXPECT_EQ(collection.getProcessorSpeedMHz(), 400);
}

TEST(CpuInfo, testCollectionForSpeedInMhz2) {
  CpuInfoContent cpuInfoContent("xxx @ 2400 MHz", 1, 1, 1);
  CpuInfo cpuInfo(cpuInfoContent.getContent());
  Collection collection(&cpuInfo);
  EXPECT_EQ(collection.getProcessorSpeedMHz(), 2400);
}

TEST(CpuInfo, testCollectionForSpeedRecognitionGhz) {
  CpuInfoContent cpuInfoContent("xxx @ 2.4", 1, 1, 1);
  CpuInfo cpuInfo(cpuInfoContent.getContent());
  Collection collection(&cpuInfo);
  EXPECT_EQ(collection.getProcessorSpeedMHz(), 2400);
}

TEST(CpuInfo, testCollectionForSpeedRecognitionMhz) {
  CpuInfoContent cpuInfoContent("xxx @ 2400", 1, 1, 1);
  CpuInfo cpuInfo(cpuInfoContent.getContent());
  Collection collection(&cpuInfo);
  EXPECT_EQ(collection.getProcessorSpeedMHz(), 2400);
}

}  // namespace cpu
}  // namespace caffe

