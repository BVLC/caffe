#include "caffe/common.hpp"

int main(int argc, char** argv) {
  LOG(FATAL) << "Deprecated. Use caffe.bin devicequery "
                "[--device_id=0] instead.";
  return 0;
}
