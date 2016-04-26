#pragma once

#if defined(_MSC_VER)

typedef unsigned int uint;

#define snprintf    _snprintf_s

#define NOMINMAX

#include <process.h>
#define getpid  _getpid

#endif

