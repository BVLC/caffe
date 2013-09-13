#include <cstring>
#include <cuda_runtime.h>

#include "gtest/gtest.h"
#include "caffeine/common.hpp"
#include "caffeine/syncedmem.hpp"


namespace caffeine {

class SyncedMemoryTest : public ::testing::Test {};

TEST_F(SyncedMemoryTest, TestInitialization) {
  SyncedMemory mem(10);
  EXPECT_EQ(mem.head(), SyncedMemory::UNINITIALIZED);
  EXPECT_EQ(mem.size(), 10);
  SyncedMemory* p_mem = new SyncedMemory(10 * sizeof(float));
  EXPECT_EQ(p_mem->size(), 10 * sizeof(float));
  delete p_mem;
}

TEST_F(SyncedMemoryTest, TestAllocation) {
  SyncedMemory mem(10);
  EXPECT_TRUE(mem.cpu_data());
  EXPECT_TRUE(mem.gpu_data());
  EXPECT_TRUE(mem.mutable_cpu_data());
  EXPECT_TRUE(mem.mutable_gpu_data());
}

TEST_F(SyncedMemoryTest, TestCPUWrite) {
  SyncedMemory mem(10);
  void* cpu_data = mem.mutable_cpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::HEAD_AT_CPU);
  memset(cpu_data, 1, mem.size());
  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ(((char*)cpu_data)[i], 1);
  }
  const void* gpu_data = mem.gpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::SYNCED);
}

TEST_F(SyncedMemoryTest, TestGPUWrite) {
  SyncedMemory mem(10);
  void* gpu_data = mem.mutable_gpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::HEAD_AT_GPU);
  CUDA_CHECK(cudaMemset(gpu_data, 1, mem.size()));
  const void* cpu_data = mem.cpu_data();
  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ(((char*)cpu_data)[i], 1);
  }
  EXPECT_EQ(mem.head(), SyncedMemory::SYNCED);
}

}
