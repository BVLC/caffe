#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <sstream>  // NOLINT(readability/streams)
#include <string>

#include "caffe/util/indexed_data.hpp"

namespace caffe {

template <typename Dtype>
SimpleIndexedTextFile<Dtype>::SimpleIndexedTextFile(
    const std::string& file_name) {
  std::ifstream input(file_name.c_str());
  CHECK(input) << "Fail to open " << file_name << " for reading";

  std::string line;
  Dtype tmp;
  while (std::getline(input, line)) {
    // indices[i] is the start point for line i
    indices_.push_back(data_.size());
    std::istringstream ss(line);
    while (ss >> tmp)
      data_.push_back(tmp);
  }
  indices_.push_back(data_.size());
}

template <typename Dtype>
index_type SimpleIndexedTextFile<Dtype>::read(
    index_type index, Dtype* out, index_type length) {
  if (index >= data_.size())
    return 0;

  std::size_t start = indices_[index];
  std::size_t finish = std::min<std::size_t>(start + length,
                                             indices_[index + 1]);
  std::copy(&data_[start], &data_[finish], out);
  return indices_[index + 1] - indices_[index];
}

INSTANTIATE_CLASS(SimpleIndexedTextFile);
}  // namespace caffe
