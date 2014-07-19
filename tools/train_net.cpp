#include "caffe/caffe.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {

  LOG(FATAL) << "Deprecated. Use caffe train --solver=... "
                "[--snapshot=...] instead.";
  return 0;
}
