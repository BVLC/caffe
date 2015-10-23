#pragma once

#ifdef _MSC_VER

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#define snprintf _snprintf_s
#define getpid _getpid

#define mkdir(str, mode) _mkdir(str)

#define __builtin_popcount __popcnt
#define __builtin_popcountl __popcnt64

static const char letters[] =
"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

/* Generate a temporary file name based on TMPL.  TMPL must match the
rules for mk[s]temp (i.e. end in "XXXXXX").  The name constructed
does not exist at the time of the call to mkstemp.  TMPL is
overwritten with the result.  */
int mkstemp(char *tmpl);
#endif  // _MSC_VER