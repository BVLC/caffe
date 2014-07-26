// Copyright 2014 BVLC and contributors.

#include "caffe/caffe.hpp"

int main(int argc, char** argv) {
  LOG(FATAL) << "Deprecated. Use caffe.bin speedtest --net_proto_file=... "
             "[--run_iterations=50] [--speedtest_with_gpu] [--device_id=0]";
  return 0;
}
