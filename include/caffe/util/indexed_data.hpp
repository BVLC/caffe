#ifndef CAFFE_UTIL_INDEXED_DATA_H_
#define CAFFE_UTIL_INDEXED_DATA_H_

#include <boost/unordered_map.hpp>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Integer type used for indexing. Can be redefined to support
 *        64-bit indices
 */
typedef int index_type;

/**
 * @brief An abstract class for retrieving data array
 *        by index. Used by IndirectionLayer to convert indices
 *        into blobs.
 */
template <typename Dtype>
class IndexedDataReader {
 public:
  IndexedDataReader() {}
  virtual ~IndexedDataReader() {}

  /**
   * @brief Retrieve the data.
   * @param index The index of the data
   * @param out The caller allocated storage to write data into
   * @param length The length of the caller allocated array
   * @return The actual length of the data, which could be larger or
   *         smaller than the length parameter
   *
   * This function shall be stateless instead of stream-like. That is,
   * calling it twice with the same arguments shall return the same
   * data, provided the underlying storage do not mutate in the meantime.
   *
   * It is not marked const because implementations may cache responses.
   *
   * When length equals zero, out can be a null pointer; otherwise, a null
   * out will cause undefined behavior.
   */
  virtual index_type read(index_type index,
          Dtype* out, index_type length) = 0;

  /**
   * @brief The number of entries in the storage.
   *        Valid index is in the range [0, size()).
   */
  virtual index_type size() const = 0;

  /**
   * @brief Factory function for creating subtypes of IndexedDataReader
   */
  static shared_ptr<IndexedDataReader<Dtype> > make_reader(
      IndirectionParameter::IndirectionSourceType type,
      const std::string& source_file);

  DISABLE_COPY_AND_ASSIGN(IndexedDataReader);
};

/**
 * @brief A cache class of IndexedDataReader.
 */
template <typename Dtype>
class IndexedDataReadCache: public IndexedDataReader<Dtype> {
 private:
  index_type length_;

 public:
  /**
   * @brief Constructor
   * @param The length of each data array
   *
   * The cache only works with readers whose each data array has the same
   * length, and has no gaps in indices
   */
  explicit IndexedDataReadCache(index_type length): length_(length)
  {}

  virtual void set_underlying_reader(
      shared_ptr<IndexedDataReader<Dtype> > reader) = 0;

  index_type data_length() const { return length_; }

  static shared_ptr<IndexedDataReader<Dtype> > make_cached_reader(
      IndirectionParameter::IndirectionCacheType type,
      shared_ptr<IndexedDataReader<Dtype> > reader,
      index_type data_length,
      index_type cache_block_size,
      index_type cache_block_num);
};

template <typename Dtype>
class LinearIndexedStorage : public IndexedDataReader<Dtype> {
 protected:
  std::vector<Dtype> data_;
  std::vector<std::size_t> indices_;

 public:
  virtual index_type read(index_type index,
        Dtype* out, index_type length);

  virtual index_type size() const { return data_.size(); }
};

/**
 * @brief The simplest indexed data storage backed by a text file
 *        where each line consists of numbers separated by whitespace
 */
template <typename Dtype>
class SimpleIndexedTextFile
  : public LinearIndexedStorage<Dtype> {
 public:
    explicit SimpleIndexedTextFile(const std::string& source_file);
};

template <typename Dtype>
class IndexedFileStorage: public IndexedDataReader<Dtype> {
 private:
  std::vector<std::string> file_names_;

 protected:
  virtual index_type read(const std::string& file,
                          Dtype* out, index_type length) = 0;

 public:
  explicit IndexedFileStorage(const std::string& source_file);

  virtual index_type read(index_type index,
        Dtype* out, index_type length);

  virtual index_type size() const { return file_names_.size(); }
};

/**
 * @brief A indexed data storage where each line of source file points
 *        to a binary file of Dtype array in machine byte order
 */
template <typename Dtype>
class IndexedBinaryFiles : public IndexedFileStorage<Dtype> {
 public:
  explicit IndexedBinaryFiles(const std::string& source_file)
    : IndexedFileStorage<Dtype>(source_file) {}

 protected:
  virtual index_type read(const std::string& file,
        Dtype* out, index_type length);
};

/**
 * @brief A indexed data storage where each line of source file points
 *        to a BlobProto file
 */
template <typename Dtype>
class IndexedBlobProtos : public IndexedFileStorage<Dtype> {
 public:
  explicit IndexedBlobProtos(const std::string& source_file)
    : IndexedFileStorage<Dtype>(source_file) {}

 protected:
  virtual index_type read(const std::string& file,
        Dtype* out, index_type length);
};

template <typename Dtype>
class IndexedCompleteInMemoryCache : public IndexedDataReadCache<Dtype> {
 private:
  std::vector<Dtype> data_;

 public:
  explicit IndexedCompleteInMemoryCache(index_type length)
    : IndexedDataReadCache<Dtype>(length) {}

  virtual void set_underlying_reader(
      shared_ptr<IndexedDataReader<Dtype> > reader);

  virtual index_type size() const { return data_.size(); }

  virtual index_type read(index_type index,
          Dtype* out, index_type length);
};

template <typename Dtype>
class IndexedBlockCache : public IndexedDataReadCache<Dtype> {
 protected:
  struct block;
  block* blocks_;

 private:
  shared_ptr<IndexedDataReader<Dtype> > reader_;
  boost::unordered_map<index_type, block*> mapping_;
  index_type block_size_;
  index_type block_num_;

  void fill_block(block* b, index_type block_index);

 public:
  explicit IndexedBlockCache(index_type length, index_type block_size,
                             index_type block_num);

  virtual ~IndexedBlockCache();

  virtual void set_underlying_reader(
      shared_ptr<IndexedDataReader<Dtype> > reader) {
    reader_ = reader;
  }

  virtual index_type size() const { return reader_ ? reader_->size() : 0; }

  virtual index_type read(index_type index,
          Dtype* out, index_type length);

  index_type block_size() const { return block_size_; }

  index_type block_number() const { return block_num_; }

 protected:
  /**
   * @brief Query the mapping of blocks
   * @param i The <emph>block index</emph>, equal to data index divided by
   *          block_size
   * @return The mapped cached block; NULL on failure
   */
  block* query_block(index_type i) const;

  /**
   * @brief Subclasses should override this method if it wants to record
   *        any metadata for cache replacement algorithm
   */
  virtual void mark_block_as_accessed(block* b) = 0;

  /**
   * @brief Subclasses should implement this method
   * @return A block in the array blocks_ to reject
   */
  virtual block* find_victim() = 0;
};

template <typename Dtype>
class IndexedBlockCache_Clock : public IndexedBlockCache<Dtype> {
 private:
  index_type hand_;

  typedef IndexedBlockCache<Dtype> parent_type;
  typedef typename parent_type::block block;

 public:
  explicit IndexedBlockCache_Clock(index_type length, index_type block_size,
                                   index_type block_num)
    : IndexedBlockCache<Dtype>(length, block_size, block_num), hand_(0) {}

 protected:
  virtual void mark_block_as_accessed(block* b) {}
  virtual block* find_victim();
};
}  // namespace caffe

#endif    // CAFFE_UTIL_INDEXED_DATA_H_
