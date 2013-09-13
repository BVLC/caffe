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
    !(condition) ? nullout : std::cout

#define CHECK(condition) \
    LOG_IF(condition) << "Check failed: " #condition " "

#ifndef NDEBUG

#define DCHECK(condition) CHECK(condition)

#else

#define DCHECK(condition)

#endif  // NDEBUG


#define CUDA_CHECK(condition) \
    CUDA_LOG_IF(condition) << "Check failed: " #condition " "


// TODO(Yangqing): make a better logging scheme
#define CUDA_LOG_IF(condition) \
    ((condition) != cudaSuccess) ? nullout : std::cout

#define CUDA_CHECK(condition) \
    CUDA_LOG_IF(condition) << "Check failed: " #condition " "

#ifndef NDEBUG

#define CUDA_DCHECK(condition) CUDA_CHECK(condition)

#else

#define CUDA_DCHECK(condition)

#endif  // NDEBUG

#endif  // CAFFEINE_COMMON_HPP_