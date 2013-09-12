#ifndef CAFFEINE_COMMON_HPP_
#define CAFFEINE_COMMON_HPP_

#include <iostream>

#include <boost/shared_ptr.hpp>

#include "driver_types.h"

namespace caffeine {
  using boost::shared_ptr;
}

static std::ostream nullout(0);

// TODO(Yangqing): make a better logging scheme
#define LOG_IF(condition) \
    ((condition) != cudaSuccess) ? nullout : std::cout

#define CUDA_CHECK(condition) \
    LOG_IF(condition) << "Check failed: " #condition " "
    
#endif  // CAFFEINE_COMMON_HPP_