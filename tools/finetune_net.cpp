#include "caffe/caffe.hpp"

int main(int argc, char** argv) {
  LOG(FATAL) << "Deprecated. Use caffe train --solver=... "
                "[--weights=...] instead.";
  return 0;
}
