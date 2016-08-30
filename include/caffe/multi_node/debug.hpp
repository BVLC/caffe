
#ifndef MULTI_NODE_DEBUG_HPP_
#define MULTI_NODE_DEBUG_HPP_

#include <glog/logging.h>

// disable multi node debug log by default
#ifndef MULTI_NODE_DEBUG
#define MULTI_NODE_DEBUG 0
#endif

// logging for multi node debug
#define MLOG(level) LOG_IF(level, MULTI_NODE_DEBUG)

#endif

