// Copyright 2013 Yangqing Jia

#include <cstring>
#include <cuda_runtime.h>

#include "gtest/gtest.h"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

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
  // check if values are the same
  char* recovered_value = new char[10];
  cudaMemcpy((void*)recovered_value, gpu_data, 10, cudaMemcpyDeviceToHost);
  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ(((char*)recovered_value)[i], 1);
  }
  // do another round
  cpu_data = mem.mutable_cpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::HEAD_AT_CPU);
  memset(cpu_data, 2, mem.size());
  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ(((char*)cpu_data)[i], 2);
  }
  gpu_data = mem.gpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::SYNCED);
  // check if values are the same
  cudaMemcpy((void*)recovered_value, gpu_data, 10, cudaMemcpyDeviceToHost);
  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ(((char*)recovered_value)[i], 2);
  }
  delete[] recovered_value;
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

  gpu_data = mem.mutable_gpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::HEAD_AT_GPU);
  CUDA_CHECK(cudaMemset(gpu_data, 2, mem.size()));
  cpu_data = mem.cpu_data();
  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ(((char*)cpu_data)[i], 2);
  }
  EXPECT_EQ(mem.head(), SyncedMemory::SYNCED);
}

}
