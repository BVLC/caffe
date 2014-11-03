#ifndef CAFFE_UTIL_INDEXED_DATA_H_
#define CAFFE_UTIL_INDEXED_DATA_H_

#include <caffe/common.hpp>

namespace caffe {

/**
 * @brief An abstract class for retrieving data array
 *        by index. Used by IndirectionLayer to convert indices
 *        into blobs.
 */
template <typename Dtype>
class IndexedDataReader {
 public:
  IndexedDataReader() {}
  virtual IndexedDataReader() {}

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
  virtual uint32_t retrieve(uint32_t index,
          Dtype* out, uint32_t length) = 0;
};


/**
 * @brief An abstract class for writing into floating point array storage.
 */
template <typename Dtype>
class IndexedDataWriter {
 public:
  IndexedDataWriter() {}
  virtual ~IndexedDataWriter() {}

  /**
   * @brief Write the data to the storage.
   * @return Whether the write succeeded
   *
   * in can never be null.
   */
  virtual bool write(uint32_t index, const Dtype* in,
          uint32_t length) = 0;
};


/**
 * @brief A cache class of IndexedDataReader.
 */
template <typename Dtype>
class IndexedDataReadCache: public IndexedDataReader {
 private:
  shared_ptr<IndexedDataReader> reader_;
  uint32_t length_;

 public:
  /**
   * @brief Constructor
   * @param reader The underlying reader
   * @param The length of each data array
   *
   * The cache only works with readers whose data array has the same
   * length, and has no gaps in indices
   */
  explicit IndexedDataReadCache(shared_ptr<IndexedDataReader>
          reader, uint32_t length): reader_(reader), length_(length)
  {}

  uint32_t data_length() const { return length_; }
};
}  // namespace caffe

#endif    // CAFFE_UTIL_INDEXED_DATA_H_
