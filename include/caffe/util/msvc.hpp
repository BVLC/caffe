#ifndef CAFFE_UTIL_MSVC_H_
#define CAFFE_UTIL_MSVC_H_

#ifdef _MSC_VER
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#if _MSC_VER >= 1400  // VC 8.0 and later deprecate snprintf and _snprintf.
# define snprintf _snprintf_s
#else
# define snprintf _snprintf
#endif 

#define getpid _getpid

#define  mkdir(str, mode) _mkdir(str)

#define __builtin_popcount __popcnt 
#define __builtin_popcountl __popcnt

#endif // _MSC_VER


#endif