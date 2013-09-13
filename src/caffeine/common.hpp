#ifndef CAFFEINE_COMMON_HPP_
#define CAFFEINE_COMMON_HPP_

#include <iostream>

#include <boost/shared_ptr.hpp>
#include <glog/logging.h>

#include "driver_types.h"

namespace caffeine {
  using boost::shared_ptr;
}

static std::ostream nullout(0);

#define CUDA_CHECK(condition) \
    CHECK((condition) == cudaSuccess)

#endif  // CAFFEINE_COMMON_HPP_
