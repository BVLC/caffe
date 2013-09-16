#include <cstring>
#include <cuda_runtime.h>

#include "gtest/gtest.h"
#include "caffeine/common.hpp"

namespace caffeine {

class CommonTest : public ::testing::Test {};

TEST_F(CommonTest, TestCublasHandler) {
  int cuda_device_id;
  CUDA_CHECK(cudaGetDevice(&cuda_device_id));
  EXPECT_TRUE(Caffeine::cublas_handle());
}

TEST_F(CommonTest, TestVslStream) {
  EXPECT_TRUE(Caffeine::vsl_stream());
}

TEST_F(CommonTest, TestBrewMode) {
 EXPECT_EQ(Caffeine::mode(), Caffeine::CPU);
 Caffeine::set_mode(Caffeine::GPU);
 EXPECT_EQ(Caffeine::mode(), Caffeine::GPU);
}

TEST_F(CommonTest, TestPhase) {
 EXPECT_EQ(Caffeine::phase(), Caffeine::TRAIN);
 Caffeine::set_phase(Caffeine::TEST);
 EXPECT_EQ(Caffeine::phase(), Caffeine::TEST);
}

}
