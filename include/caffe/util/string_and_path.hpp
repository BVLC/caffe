#ifndef CAFFE_UTIL_STRING_AND_PATH_HPP_
#define CAFFE_UTIL_STRING_AND_PATH_HPP_

#include "caffe/common.hpp"
#include "caffe/definitions.hpp"

namespace caffe {

bool is_forbidden_path_char(char c) {
    static std::string forbiddenChars("\\/:?\"<>|");
    return std::string::npos != forbiddenChars.find(c);
}

string sanitize_path_string(string path) {
  std::string path_copy(path);
  std::replace_if(path_copy.begin(), path_copy.end(), is_forbidden_path_char,
                  ' ');
  return path_copy;
}


}  // namespace caffe


#endif  // CAFFE_UTIL_STRING_AND_PATH_HPP_
