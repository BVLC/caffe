//
// Created by sadeep on 14/10/15.
//

#ifndef CAFFE_TVG_COMMON_UTILS_HPP
#define CAFFE_TVG_COMMON_UTILS_HPP

#include <string>
#include "caffe/blob.hpp"

namespace tvg {

  namespace CommonUtils {

    template<typename Dtype>
    void read_into_array(const int N, const std::string & source, Dtype * target) {

      std::stringstream iss;
      iss.clear();
      iss << source;
      std::string token;

      for (int i = 0; i < N; ++i) {

        if (std::getline(iss, token, ' ')) {
          target[i] = std::stof(token);
        } else {
          throw std::runtime_error(
                  "A malformed string! >" + source + "<. Couldn't read " + std::to_string(N) + " values.");
        }
      }
    }

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

      /*
       * Saves a blob to file
       * Assumes N = 1
       * Saves in the following format WxHxC (displays channel first)
       */
      template<typename Dtype>
      void save_blob_to_file(const caffe::Blob<Dtype> & blob, const std::string& filename)
      {

        if ( blob.num() != 1) { return; }
        const Dtype * data = blob.cpu_data();

        std::ofstream fs(filename.c_str());

        for (size_t h = 0; h < blob.height(); ++h){
          for (size_t w = 0; w < blob.width(); ++w){
            for (size_t c = 0; c < blob.channels(); ++c){
              size_t index =  (c * blob.height() + h) * blob.width() + w;
              fs << data[index] << ( (c+1) % blob.channels() == (0) ? '\n' : ',');
            }
          }
        }
      }

      template<typename Dtype>
      void save_stat_potentials_to_file(std::vector<std::vector<std::vector<Dtype> > > & stat_potentials, const std::string& filename)
      {

        std::ofstream fs(filename.c_str());

        for (size_t layer = 0; layer < stat_potentials.size(); ++layer){
          for (size_t segment = 0; segment < stat_potentials[layer].size(); ++segment){
            for (size_t i = 0; i < stat_potentials[layer][segment].size(); ++i){
              fs << stat_potentials[layer][segment][i] << ( (i+1) % stat_potentials[layer][segment].size() == (0) ? '\n' : ',');
            }
          }
        }
      }

  }
}


#endif //CAFFE_TVG_COMMON_UTILS_HPP
