#include "caffe/util/indexed_data.hpp"

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <sstream>  // NOLINT(readability/streams)
#include <string>

namespace caffe {

template <typename Dtype>
SimpleIndexedTextFile<Dtype>::SimpleIndexedTextFile(
    const std::string& source_file) {
  std::ifstream input(source_file.c_str());
  CHECK(input) << "Fail to open " << source_file << " for reading";

  std::string line;
  Dtype tmp;
  while (std::getline(input, line)) {
    // indices[i] is the start point for line i
    this->indices_.push_back(this->data_.size());
    std::istringstream ss(line);
    while (ss >> tmp)
      this->data_.push_back(tmp);
  }
  this->indices_.push_back(this->data_.size());
}

template <typename Dtype>
index_type LinearIndexedStorage<Dtype>::read(
    index_type index, Dtype* out, index_type length) {
  if (index >= data_.size())
    return 0;

  if (length > 0) {
    std::size_t start = indices_[index];
    std::size_t finish = std::min<std::size_t>(start + length,
                                               indices_[index + 1]);
    std::copy(&data_[start], &data_[finish], out);
  }
  return indices_[index + 1] - indices_[index];
}

INSTANTIATE_CLASS(SimpleIndexedTextFile);

template <typename Dtype>
IndexedBinaryFiles<Dtype>::IndexedBinaryFiles(
    const std::string& source_file) {
  std::ifstream input(source_file.c_str());
  CHECK(input) << "Fail to open " << source_file << " for reading";

  std::string fn;
  while (std::getline(input, fn))
    file_names_.push_back(fn);
}

template <typename Dtype>
index_type IndexedBinaryFiles<Dtype>::read(
    index_type index, Dtype* out, index_type length) {
  if (index >= file_names_.size())
    return 0;

  std::ifstream input(file_names_[index].c_str(), std::ios_base::binary);
  CHECK(input) << "Fail to open " << file_names_[index] << " for reading";
  input.seekg(0, std::ios_base::end);
  index_type actual_length = input.tellg() / sizeof(Dtype);

  if (length > 0) {
    input.seekg(0);
    input.read(reinterpret_cast<char*>(out),
               std::min(length, actual_length) * sizeof(Dtype));
  }
  return actual_length;
}

INSTANTIATE_CLASS(IndexedBinaryFiles);

template<typename Dtype>
shared_ptr<IndexedDataReader<Dtype> > IndexedDataReader<Dtype>::make_reader(
    IndirectionParameter::IndirectionSourceType type,
    const std::string& source_file) {
  shared_ptr<IndexedDataReader<Dtype> > result;

  switch (type) {
  case IndirectionParameter_IndirectionSourceType_SIMPLE_TEXT:
    result.reset(new SimpleIndexedTextFile<Dtype>(source_file));
    break;
  case IndirectionParameter_IndirectionSourceType_INDEXED_BINARY:
    result.reset(new IndexedBinaryFiles<Dtype>(source_file));
    break;
  default:
    LOG(FATAL) << "Unkown IndirectionParameter type " << static_cast<int>(type);
  }

  return result;
}

// Explicit instantiation of factory function
template
shared_ptr<IndexedDataReader<float> > IndexedDataReader<float>::make_reader(
    IndirectionParameter::IndirectionSourceType type,
    const std::string& source_file);

template
shared_ptr<IndexedDataReader<double> > IndexedDataReader<double>::make_reader(
    IndirectionParameter::IndirectionSourceType type,
    const std::string& source_file);
}  // namespace caffe
