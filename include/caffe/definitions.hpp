#ifndef CAFFE_DEFINITIONS_HPP_
#define CAFFE_DEFINITIONS_HPP_

#include <stdint.h>

// Types used for parameters, offset computations and so on
#define int_tp int64_t
#define uint_tp uint64_t

// Definitions used to cast the types above as needed
#define int_tpc long long  // NOLINT
#define uint_tpc unsigned long long  // NOLINT

#endif /* CAFFE_DEFINITIONS_HPP_ */
