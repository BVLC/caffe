#ifndef CAFFE_UTIL_NCCL_H_
#define CAFFE_UTIL_NCCL_H_
#ifdef USE_NCCL

#include <nccl.h>

#include "caffe/common.hpp"

#define NCCL_CHECK(condition) \
{ \
  ncclResult_t result = condition; \
  CHECK_EQ(result, ncclSuccess) << " " \
    << ncclGetErrorString(result); \
}

namespace caffe {

namespace nccl {

template <typename Dtype> class dataType;

template<> class dataType<float> {
 public:
  static const ncclDataType_t type = ncclFloat;
};
template<> class dataType<double> {
 public:
  static const ncclDataType_t type = ncclDouble;
};

}  // namespace nccl

}  // namespace caffe

#endif  // end USE_NCCL

#endif  // CAFFE_UTIL_NCCL_H_
