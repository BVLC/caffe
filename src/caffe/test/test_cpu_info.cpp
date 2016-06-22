#include "gtest/gtest.h"

#include "caffe/util/cpu_info.hpp"

namespace caffe {
namespace cpu {

TEST(CpuInfo, isProcessorStructureInitialized) {
  Processor processor;
  EXPECT_EQ(processor.processor, 0);
  EXPECT_EQ(processor.physicalId, 0);
  EXPECT_EQ(processor.siblings, 0);
  EXPECT_EQ(processor.coreId, 0);
  EXPECT_EQ(processor.cpuCores, 0);
  EXPECT_EQ(processor.speedMHz, 0);
}

}  // namespace cpu
}  // namespace caffe

