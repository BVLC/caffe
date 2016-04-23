#pragma once

#if defined(_MSC_VER)

typedef unsigned int uint;

#define __builtin_popcount  __popcnt 
#define __builtin_popcountl __popcnt64

#define snprintf    _snprintf_s

#define NOMINMAX

#include <process.h>
#define getpid  _getpid

#endif

