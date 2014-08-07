#include "caffe/caffe.hpp"

int main(int argc, char** argv) {
  LOG(FATAL) << "Deprecated. Use caffe.bin test --model=... "
      "--weights=... instead.";
  return 0;
}
