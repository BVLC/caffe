#include "caffe/util/indexed_data.hpp"

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <sstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/util/io.hpp"

template <typename InputIterator, typename OutputIterator,
          typename InputLength, typename OutputLength>
static void copy(InputIterator ibegin, InputLength ilen,
                 OutputIterator obegin, OutputLength olen) {
  if (sizeof(ilen) > sizeof(olen))
    std::copy(ibegin, ibegin + std::min<InputLength>(ilen, olen), obegin);
  else
    std::copy(ibegin, ibegin + std::min<OutputLength>(ilen, olen), obegin);
}

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
index_type SimpleIndexedTextFile<Dtype>::read(
    index_type index, Dtype* out, index_type length) {
  if (index >= this->size())
    return 0;

  std::size_t actual_length = indices_[index + 1] - indices_[index];
  if (length > 0) {
    std::size_t start = indices_[index];
    copy(&data_[start], actual_length, out, length);
  }
  return actual_length;
}

template <typename Dtype>
index_type SimpleIndexedTextFile<Dtype>::size() const {
  return static_cast<index_type>(indices_.size()) - 1;
}

INSTANTIATE_CLASS(SimpleIndexedTextFile);

template <typename Dtype>
IndexedFileStorage<Dtype>::IndexedFileStorage(
    const std::string& source_file) {
  std::ifstream input(source_file.c_str());
  CHECK(input) << "Fail to open " << source_file << " for reading";

  std::string fn;
  while (std::getline(input, fn))
    file_names_.push_back(fn);
}

template <typename Dtype>
index_type IndexedFileStorage<Dtype>::read(
    index_type index, Dtype* out, index_type length) {
  if (index >= file_names_.size())
    return 0;

  return read(file_names_[index], out, length);
}

template <typename Dtype>
index_type IndexedBinaryFiles<Dtype>::read(
    const std::string& file_name, Dtype* out, index_type length) {
  std::ifstream input(file_name.c_str(), std::ios_base::binary);
  CHECK(input) << "Fail to open " << file_name << " for reading";
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

template <typename Dtype>
index_type IndexedBlobProtos<Dtype>::read(
    const std::string& file_name, Dtype* out, index_type length) {
  BlobProto blob;
  ReadProtoFromBinaryFileOrDie(file_name, &blob);

  if (length > 0) {
    copy(blob.data().data(), blob.data().size(), out, length);
  }

  return blob.data_size();
}

INSTANTIATE_CLASS(IndexedBlobProtos);

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
  case IndirectionParameter_IndirectionSourceType_INDEXED_BLOB:
    result.reset(new IndexedBlobProtos<Dtype>(source_file));
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

template <typename Dtype>
void IndexedCompleteInMemoryCache<Dtype>::set_underlying_reader(
    shared_ptr<IndexedDataReader<Dtype> > reader) {
  for (index_type i = 0; i < reader->size(); ++i) {
    index_type last = data_.size();
    data_.resize(last + this->data_length());
    CHECK_EQ(reader->read(i, &data_[last], this->data_length()),
        this->data_length()) << "Error reading data from underlying storage";
  }
}

template <typename Dtype>
index_type IndexedCompleteInMemoryCache<Dtype>::read(
    index_type index, Dtype* out, index_type length) {
  if (index >= data_.size())
    return 0;

  if (length > 0) {
    std::size_t start = this->data_length() * index;
    copy(&data_[start], this->data_length(), out, length);
  }
  return this->data_length();
}

INSTANTIATE_CLASS(IndexedCompleteInMemoryCache);

template <typename Dtype>
shared_ptr<IndexedDataReader<Dtype> >
IndexedCachedDataReader<Dtype>::make_cached_reader(
    IndirectionParameter::IndirectionCacheType type,
    shared_ptr<IndexedDataReader<Dtype> > reader,
    index_type data_length,
    index_type cache_block_size,
    index_type cache_block_num) {
  shared_ptr<IndexedCachedDataReader<Dtype> > result;

  switch (type) {
  case IndirectionParameter_IndirectionCacheType_NONE:
    return reader;
  case IndirectionParameter_IndirectionCacheType_WHOLE:
    result.reset(new IndexedCompleteInMemoryCache<Dtype>(data_length));
    break;
  case IndirectionParameter_IndirectionCacheType_CLOCK:
    result.reset(new IndexedBlockCache_Clock<Dtype>(
                   data_length, cache_block_size, cache_block_num));
    break;
  default:
    LOG(FATAL) << "Unknown cache type " << static_cast<int>(type);
  }

  result->set_underlying_reader(reader);
  return result;
}

// Explicit instantiation of factory function
template
shared_ptr<IndexedDataReader<float> >
IndexedCachedDataReader<float>::make_cached_reader(
    IndirectionParameter::IndirectionCacheType type,
    shared_ptr<IndexedDataReader<float> > reader,
    index_type data_length,
    index_type cache_block_size,
    index_type cache_block_num);

template
shared_ptr<IndexedDataReader<double> >
IndexedCachedDataReader<double>::make_cached_reader(
    IndirectionParameter::IndirectionCacheType type,
    shared_ptr<IndexedDataReader<double> > reader,
    index_type data_length,
    index_type cache_block_size,
    index_type cache_block_num);

template <typename Dtype>
struct IndexedBlockCache<Dtype>::block {
  std::vector<Dtype> data;
  index_type start_index, end_index;
  bool accessed;
  bool valid;

  block() : start_index(-1), end_index(-1), accessed(false), valid(false) {}
};

template <typename Dtype>
IndexedBlockCache<Dtype>::IndexedBlockCache(index_type length,
                                            index_type block_size,
                                            index_type block_num)
  : IndexedCachedDataReader<Dtype>(length),
    block_size_(block_size), block_num_(block_num) {
  blocks_ = new block[block_num];
  for (index_type i = 0; i < block_num; ++i)
    blocks_[i].data.resize(block_size * length);
}

template <typename Dtype>
IndexedBlockCache<Dtype>::~IndexedBlockCache() {
  delete[] blocks_;
}

template <typename Dtype>
index_type IndexedBlockCache<Dtype>::read(index_type index,
                               Dtype* out,
                               index_type length) {
  if (!reader_)
    return 0;

  index_type block_index = index / block_size();
  block* b = query_block(block_index);

  if (!b) {
    b = find_victim();
    mapping_.erase(b->start_index / block_size());
    fill_block(b, block_index);
    b->valid = true;
    mapping_.insert(std::make_pair(block_index, b));
  }

  b->accessed = true;
  mark_block_as_accessed(b);
  if (length > 0 && index >= b->start_index
      && index < b->end_index) {
    Dtype* begin = &b->data[(index - b->start_index) * this->data_length()];
    copy(begin, this->data_length(), out, length);
  }
  return (index >= b->start_index && index < b->end_index) ?
        this->data_length() : 0;
}

template <typename Dtype>
typename IndexedBlockCache<Dtype>::block*
IndexedBlockCache<Dtype>::query_block(index_type i) const {
  typedef boost::unordered_map<index_type, block*> map_type;
  typedef typename map_type::const_iterator iterator_type;
  iterator_type it = mapping_.find(i);
  return it == mapping_.end() ? NULL : it->second;
}

template <typename Dtype>
void IndexedBlockCache<Dtype>::fill_block(
    typename IndexedBlockCache<Dtype>::block* b, index_type block_index) {
  b->start_index = block_index * block_size();
  b->end_index = std::min<index_type>(b->start_index + block_size(),
                                      reader_->size());

  for (index_type i = b->start_index; i < b->end_index; ++i) {
    Dtype* buffer = &b->data[(i - b->start_index) * this->data_length()];
    CHECK_EQ(reader_->read(i, buffer, this->data_length()), this->data_length())
        << "Error reading from underlying reader";
  }
}

template <typename Dtype>
typename IndexedBlockCache_Clock<Dtype>::block*
IndexedBlockCache_Clock<Dtype>::find_victim() {
  while (this->blocks_[hand_].accessed) {
    this->blocks_[hand_].accessed = false;
    hand_ = (hand_ + 1) % this->block_number();
  }
  block* result = this->blocks_ + hand_;
  hand_ = (hand_ + 1) % this->block_number();
  return result;
}

INSTANTIATE_CLASS(IndexedBlockCache_Clock);

}  // namespace caffe
