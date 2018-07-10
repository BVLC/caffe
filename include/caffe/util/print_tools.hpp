#ifndef CAFFE_UTIL_PRINT_TOOLS_HPP_
#define CAFFE_UTIL_PRINT_TOOLS_HPP_


#include "caffe/common.hpp"


namespace caffe {

template<typename Dtype>
inline void print_matrix(Dtype* data, size_t rows, size_t cols,
                         bool col_major=false, size_t char_limit = 8) {
  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < cols; ++c) {
      std::stringstream ss;
      if (col_major) {
        ss << std::right << std::setw(char_limit)
           << std::setprecision(char_limit)
           << std::setfill(' ')
           << static_cast<float>(data[c + cols * r]);
      } else {
        ss << std::right << std::setw(char_limit)
           << std::setprecision(char_limit)
           << std::setfill(' ')
           << static_cast<float>(data[r + rows * c]);
      }
      std::cout << ss.str().substr(0, char_limit);
      if (c == cols - 1) {
        std::cout << std::endl;
      } else {
        std::cout << "  ";
      }
    }
  }
}


}  // namespace caffe


#endif  //CAFFE_UTIL_PRINT_TOOLS_HPP_
