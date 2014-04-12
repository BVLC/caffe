// Copyright 2014 BVLC and contributors.

#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

class SyncedMemoryTest : public ::testing::Test {};

TEST_F(SyncedMemoryTest, TestInitialization) {
  SyncedMemory empty;
  EXPECT_EQ(empty.head(), SyncedMemory::UNINITIALIZED);
  EXPECT_EQ(empty.size(), 0);
  EXPECT_EQ(empty.capacity(), 0);
  SyncedMemory mem(10);
  EXPECT_EQ(mem.head(), SyncedMemory::UNINITIALIZED);
  EXPECT_EQ(mem.size(), 10);
  EXPECT_EQ(mem.capacity(), 10);
  SyncedMemory* p_mem = new SyncedMemory(10 * sizeof(float));
  EXPECT_EQ(p_mem->size(), 10 * sizeof(float));
  EXPECT_EQ(p_mem->capacity(), 10 * sizeof(float));
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
    EXPECT_EQ((static_cast<char*>(cpu_data))[i], 1);
  }
  const void* gpu_data = mem.gpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::SYNCED);
  // check if values are the same
  char* recovered_value = new char[10];
  cudaMemcpy(static_cast<void*>(recovered_value), gpu_data, 10,
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((static_cast<char*>(recovered_value))[i], 1);
  }
  // do another round
  cpu_data = mem.mutable_cpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::HEAD_AT_CPU);
  memset(cpu_data, 2, mem.size());
  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((static_cast<char*>(cpu_data))[i], 2);
  }
  gpu_data = mem.gpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::SYNCED);
  // check if values are the same
  cudaMemcpy(static_cast<void*>(recovered_value), gpu_data, 10,
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((static_cast<char*>(recovered_value))[i], 2);
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
    EXPECT_EQ((static_cast<const char*>(cpu_data))[i], 1);
  }
  EXPECT_EQ(mem.head(), SyncedMemory::SYNCED);

  gpu_data = mem.mutable_gpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::HEAD_AT_GPU);
  CUDA_CHECK(cudaMemset(gpu_data, 2, mem.size()));
  cpu_data = mem.cpu_data();
  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((static_cast<const char*>(cpu_data))[i], 2);
  }
  EXPECT_EQ(mem.head(), SyncedMemory::SYNCED);
}

TEST_F(SyncedMemoryTest, TestResize) {
  SyncedMemory mem;
  mem.resize(20);
  EXPECT_EQ(mem.head(), SyncedMemory::UNINITIALIZED);
  EXPECT_EQ(mem.size(), 20);
  EXPECT_EQ(mem.capacity(), 20);
  mem.resize(0);
  EXPECT_EQ(mem.head(), SyncedMemory::UNINITIALIZED);
  EXPECT_EQ(mem.size(), 0);
  EXPECT_EQ(mem.capacity(), 20);
  mem.resize(5, 123);
  const void* cpu_data = mem.cpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::HEAD_AT_CPU);
  EXPECT_EQ(mem.size(), 5);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(123, (static_cast<const uint8_t*>(cpu_data))[i]);
  }
  mem.resize(30, 234);
  const void* gpu_data = mem.gpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::SYNCED);
  EXPECT_EQ(mem.size(), 30);
  EXPECT_EQ(mem.capacity(), 30);
  uint8_t* recovered_value = new uint8_t[30];
  cudaMemcpy(static_cast<void*>(recovered_value), gpu_data, 30,
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(123, (static_cast<const uint8_t*>(recovered_value))[i]);
  }
  for (int i = 5; i < 30; ++i) {
    EXPECT_EQ(234, (static_cast<const uint8_t*>(recovered_value))[i]);
  }
}

TEST_F(SyncedMemoryTest, TestReserve) {
  SyncedMemory mem(10);
  mem.reserve(20);
  EXPECT_EQ(mem.head(), SyncedMemory::UNINITIALIZED);
  EXPECT_EQ(mem.size(), 10);
  EXPECT_EQ(mem.capacity(), 20);
  mem.reserve(0);
  EXPECT_EQ(mem.head(), SyncedMemory::UNINITIALIZED);
  EXPECT_EQ(mem.size(), 10);
  EXPECT_EQ(mem.capacity(), 20);
  const void* gpu_data = mem.gpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::HEAD_AT_GPU);
  EXPECT_EQ(mem.size(), 10);
  EXPECT_EQ(mem.capacity(), 20);
  uint8_t* recovered_value = new uint8_t[10];
  cudaMemcpy(static_cast<void*>(recovered_value), gpu_data, 10,
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(0, (static_cast<const uint8_t*>(recovered_value))[i]);
  }
  mem.reserve(30);
  const void* cpu_data = mem.cpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::SYNCED);
  EXPECT_EQ(mem.size(), 10);
  EXPECT_EQ(mem.capacity(), 30);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(0, (static_cast<const uint8_t*>(cpu_data))[i]);
  }
}

}  // namespace caffe
