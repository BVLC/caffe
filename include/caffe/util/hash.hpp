#ifndef CAFFE_UTIL_HASH_HPP_
#define CAFFE_UTIL_HASH_HPP_

#include "caffe/definitions.hpp"

namespace caffe {

size_t generate_hash(string text);

size_t generate_hash(vector<string> text);

string hash_hex_string(size_t hash);

}  // namespace caffe


#endif /* CAFFE_UTIL_HASH_HPP_ */
