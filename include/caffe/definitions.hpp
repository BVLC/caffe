#ifndef CAFFE_DEFINITIONS_HPP_
#define CAFFE_DEFINITIONS_HPP_

#include <stdint.h>


#ifdef USE_INDEX_64
// Types used for parameters, offset computations and so on
#define int_tp int64_t
#define uint_tp uint64_t

// Definitions used to cast the types above as needed
#define int_tpc long long  // NOLINT
#define uint_tpc unsigned long long  // NOLINT
#else
// Types used for parameters, offset computations and so on
#define int_tp int32_t
#define uint_tp uint32_t

// Definitions used to cast the types above as needed
#define int_tpc int  // NOLINT
#define uint_tpc unsigned int  // NOLINT
#endif

enum DataType {
  FP16,
  FP32,
  FP64,
  INT8,
  INT16,
  INT32,
  INT64,
  UINT8,
  UINT16,
  UINT32,
  UINT64
};


#endif /* CAFFE_DEFINITIONS_HPP_ */
