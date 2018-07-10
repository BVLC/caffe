#ifndef CAFFE_DEFINITIONS_HPP_
#define CAFFE_DEFINITIONS_HPP_

#include <cstddef>
#include <cstdio>
#include <math.h>
#include <climits>
#include <cmath>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <memory>
#ifdef USE_OPENMP
#include <omp.h>
#endif  // USE_OPENMP
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>  // pair
#include <vector>
#include <boost/variant.hpp>


#include "caffe/trait_helper.hpp"
#include "caffe/util/half_fp.hpp"
#include "caffe/util/inline_math.hpp"


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

#ifndef CAFFE_MALLOC_PAGE_ALIGN
#define CAFFE_MALLOC_PAGE_ALIGN 4096
#endif  // CAFFE_MALLOC_PAGE_ALIGN

#ifndef CAFFE_MALLOC_CACHE_ALIGN
#define CAFFE_MALLOC_CACHE_ALIGN 64
#endif  // CAFFE_MALLOC_CACHE_ALIGN

#ifndef CAFFE_OMP_BYTE_STRIDE
#define CAFFE_OMP_BYTE_STRIDE 8
#endif  // CAFFE_OMP_BYTE_STRIDE

namespace caffe {

// Common functions and classes from std and boost that Caffe often uses.
using std::fstream;
using std::ios;
using std::isnan;
using std::isinf;
using std::is_same;
using std::iterator;
using std::make_pair;
using std::map;
using std::ostringstream;
using std::pair;
using std::tuple;
using std::string;
using std::stringstream;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::static_pointer_cast;
using boost::variant;

}


#endif /* CAFFE_DEFINITIONS_HPP_ */
