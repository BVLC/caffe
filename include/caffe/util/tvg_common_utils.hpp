
#ifndef CAFFE_TVG_COMMON_UTILS_HPP
#define CAFFE_TVG_COMMON_UTILS_HPP

#include <string>
#include "caffe/blob.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {

  namespace CommonUtils {


    template<typename Dtype>
    void read_into_the_diagonal(const std::string & source, caffe::Blob<Dtype> & blob) {

      const int height = blob.height();
      Dtype * data = blob.mutable_cpu_data();

      caffe::caffe_set(blob.count(), Dtype(0.), data);

      std::stringstream iss;
      iss.clear();
      iss << source;
      std::string token;

      for (int i = 0; i < height; ++i) {

        if (std::getline(iss, token, ' ')) {
          data[i * height + i] = std::stof(token);
        } else {
          throw std::runtime_error(
                  "A malformed string! >" + source + "<. Couldn't read " + std::to_string(height) + " values.");
        }
      }
    }

  }
}


#endif //CAFFE_TVG_COMMON_UTILS_HPP

