#ifndef CAFFE_UTIL_SEARCH_PATH_H_
#define CAFFE_UTIL_SEARCH_PATH_H_

#include <string>
#include <vector>

namespace caffe {

/**
 * @brief Parses a list of colon separated paths as a vector of paths.
 */
std::vector<std::string> ParseSearchPath(
  std::string const & search_path
);

}  // namespace caffe

#endif   // CAFFE_UTIL_SEARCH_PATH_H_
