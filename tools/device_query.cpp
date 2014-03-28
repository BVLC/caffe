// Copyright 2014 BVLC and contributors.


#include "caffe/common.hpp"
#include "caffe/net.hpp"


using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  if (argc > 2) {
    LOG(ERROR) << "device_query [device_id=0]";
    return 1;
  }
  if (argc == 2) {
    LOG(INFO) << "Querying device_id=" << argv[1];
    Caffe::SetDevice(atoi(argv[1]));
    Caffe::DeviceQuery();
  } else {
    Caffe::DeviceQuery();
  }
  return 0;
}
