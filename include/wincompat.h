#if !defined _HEADER_WIN_COMPAT_20140627_INCLUDED_
#define _HEADER_WIN_COMPAT_20140627_INCLUDED_

typedef unsigned int uint;
#define snprintf _snprintf
#define round(x) ((x)>=0?(long)((x)+0.5):(long)((x)-0.5))
#include <process.h>
#define getpid _getpid
//#define signbit(x) ((x)<0?true:false)

#endif //_HEADER_WIN_COMPAT_20140627_INCLUDED_
