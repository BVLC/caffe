#ifndef INCLUDE_COMPAT_COMPAT_H_
#define INCLUDE_COMPAT_COMPAT_H_

#ifdef _MSC_VER

#include <fcntl.h>
#include <io.h>
#include <process.h>
#include "signal_compat.h"

#define getpid() _getpid()

#else  // not _MSC_VER

#ifndef O_BINARY
#ifdef _O_BINARY
#define O_BINARY _O_BINARY
#else
#define O_BINARY 0     // If this isn't defined, the platform doesn't need it.
#endif
#endif

#endif

#endif  // INCLUDE_COMPAT_COMPAT_H_
