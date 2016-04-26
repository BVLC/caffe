#pragma once

#if defined(_MSC_VER)

#define NOMINMAX
#include <direct.h>
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>

#define mkdir(X, Y) _mkdir(X)

#include <boost/filesystem.hpp>
using ::boost::filesystem::path;

inline int open(const char *filename, int oflag, int pmode) {
    return _open(filename, oflag, pmode);
}

inline int close(int fd) {
    return _close(fd);
}

inline int mkstemp(char *temp) {
    _mktemp_s(temp, strlen(temp) + 1);
    return open(temp, _O_CREAT | _O_TEMPORARY, _S_IREAD | _S_IWRITE);
}

inline char* mkdtemp(char *temp) {
    _mktemp_s(temp, strlen(temp) + 1);
    return (0 != mkdir(temp, 0700)) ? NULL : temp;
}

#endif

