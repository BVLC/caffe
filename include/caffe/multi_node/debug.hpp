
#ifndef MULTI_NODE_DEBUG_HPP_
#define MULTI_NODE_DEBUG_HPP_

#include <glog/logging.h>

// logging for multi node debug
#define MLOG(level) LOG_IF(level, MULTI_NODE_DEBUG)

#endif

