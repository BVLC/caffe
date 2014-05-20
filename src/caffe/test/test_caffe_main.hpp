// Copyright 2014 BVLC and contributors.

// The main caffe test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.
#ifndef CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
#define CAFFE_TEST_TEST_CAFFE_MAIN_HPP_

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <cstdio>

using std::cout;
using std::endl;

int main(int argc, char** argv);

#endif  // CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
