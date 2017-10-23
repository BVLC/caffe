#include <unordered_set>
#include <iomanip>

#include "caffe/util/hash.hpp"

namespace caffe {

  size_t generate_hash(string text) {
    return std::hash<string>{}(text);
  }

  size_t generate_hash(vector<string> text) {
    size_t h;
    for(int i = 0; i < text.size(); ++i) {
      size_t hn = std::hash<std::string>{}(text[i]);
      if (i == 0) {
        h = hn;
      } else {
        h = hn ^ (h << 1);
      }
    }
    return h;
  }

  string hash_hex_string(size_t hash) {
    stringstream ss;
    ss << std::hex << hash;
    return ss.str();
  }

}
