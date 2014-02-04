// Copyright 2014 Sergio Guadarrama


#include "caffe/common.hpp"
#include "caffe/net.hpp"


using namespace caffe;

int main(int argc, char** argv) {
  if (argc > 2) {
    LOG(ERROR) << "device_query [device_id=0]";
    return 0;
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
