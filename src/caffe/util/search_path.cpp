#include "caffe/util/search_path.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace caffe {

std::vector<std::string> ParseSearchPath(
  std::string const & search_path
) {
  std::vector<std::string> result;

  std::string::const_iterator start = search_path.begin();
  while (true) {
    std::string::const_iterator i = std::find(start, search_path.end(), ':');
    result.push_back(std::string(start, i));
    if (i == search_path.end()) break;
    start = i + 1;
  }

  return result;
}

}  // namespace caffe
