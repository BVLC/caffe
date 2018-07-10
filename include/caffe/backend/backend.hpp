#ifndef CAFFE_BACKEND_BACKEND_HPP_
#define CAFFE_BACKEND_BACKEND_HPP_

#include <cstddef>
#include <type_traits>

#include "caffe/common.hpp"
#include "caffe/definitions.hpp"
#include "caffe/util/half_fp.hpp"

namespace caffe {

enum Backend {
  BACKEND_CUDA,
  BACKEND_OPENCL,
  BACKEND_HIP,
  BACKEND_CPU
};

inline string backend_name(Backend backend) {
  switch(backend) {
    case BACKEND_CPU:
      return "cpu";
    case BACKEND_CUDA:
      return "cuda";
    case BACKEND_OPENCL:
      return "opencl";
    case BACKEND_HIP:
      return "hip";
    default:
      return "unknown";
  }
}

typedef tuple<string, string, uint64_t>              KernelArg;
typedef vector<KernelArg>                            KernelArgs;

const uint64_t KERNEL_ARG_NONE          =            0ULL       ;
const uint64_t KERNEL_ARG_CONST         =            1ULL <<   0;
const uint64_t KERNEL_ARG_GLOBAL_MEM    =            1ULL <<   1;
const uint64_t KERNEL_ARG_LOCAL_MEM     =            1ULL <<   2;
const uint64_t KERNEL_ARG_MEM_OFFSET    =            1ULL <<   3;
const uint64_t KERNEL_ARG_RESTRICT      =            1ULL <<   4;

enum KernelHintOption {
  KERNEL_HINT_WORK_GROUP_X,
  KERNEL_HINT_WORK_GROUP_Y,
  KERNEL_HINT_WORK_GROUP_Z,
  KERNEL_HINT_VEC_TYPE,
  KERNEL_HINT_MIN_BLOCKS_PER_MP,
  KERNEL_REQD_WORK_GROUP_X,
  KERNEL_REQD_WORK_GROUP_Y,
  KERNEL_REQD_WORK_GROUP_Z
};

typedef tuple<KernelHintOption, string>              KernelHint;
typedef vector<KernelHint>                           KernelHints;



}  // namespace caffe

#endif  // CAFFE_BACKEND_BACKEND_HPP_
