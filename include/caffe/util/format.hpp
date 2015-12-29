#ifndef CAFFE_UTIL_FORMAT_H_
#define CAFFE_UTIL_FORMAT_H_

#include <iomanip>  // NOLINT(readability/streams)
#include <sstream>  // NOLINT(readability/streams)
#include <string>

namespace caffe {

inline std::string format_int(int n, int numberOfLeadingZeros = 0 ) {
  std::ostringstream s;
  s << std::setw(numberOfLeadingZeros) << std::setfill('0') << n;
  return s.str();
}

}

#endif   // CAFFE_UTIL_FORMAT_H_
