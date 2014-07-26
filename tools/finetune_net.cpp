// Copyright 2014 BVLC and contributors.

#include "caffe/caffe.hpp"

int main(int argc, char** argv) {
  LOG(FATAL) << "Deprecated. Use caffe.bin train --solver_proto_file=... "
                "[--pretrained_net_file=...] instead.";
  return 0;
}
