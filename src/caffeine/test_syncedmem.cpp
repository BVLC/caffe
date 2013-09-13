#include "gtest/gtest.h"
#include "caffeine/syncedmem.hpp"

namespace caffeine {

class SyncedMemoryTest : public ::testing::Test {};

TEST_F(SyncedMemoryTest, TestInitialization) {
  SyncedMemory mem(10);
  EXPECT_EQ(mem.head(), SyncedMemory::UNINITIALIZED);
}

TEST_F(SyncedMemoryTest, TestAllocation) {
  SyncedMemory mem(10);
  EXPECT_NE(mem.cpu_data(), (void*)NULL);
  EXPECT_NE(mem.gpu_data(), (void*)NULL);
}

}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}