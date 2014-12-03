#ifndef CAFFE_DATA_SOURCES_HPP_
#define CAFFE_DATA_SOURCES_HPP_

#include <boost/unordered_map.hpp>

#include <algorithm>
#include <vector>

#include "caffe/proto/caffe.pb.h"

namespace caffe {

typedef int index_type;
typedef int length_type;

template <typename Dtype>
class DataSource {
 public:
  explicit DataSource(const DataSourceParameter& param)
    : data_source_param_(param) {}

  const DataSourceParameter& data_source_param() const {
    return data_source_param_;
  }

  virtual ~DataSource() {}

  virtual length_type retrieve(index_type index,
                               Dtype* buffer,
                               length_type buffer_length) = 0;

  /**
   * @brief indices
   * @return a list of indicies that this source accepts
   *
   * This default implementation returns a single {0}.
   *
   * Subclass may allow retrieval of data at indices
   * other than those in the return value of this function.
   * It is especially true for pseudo sources, e.g. ConstantDataSource.
   */
  virtual std::vector<index_type> indices() {
    return std::vector<index_type>(1, 0);
  }

 private:
  DataSourceParameter data_source_param_;
};

template <typename Dtype>
class ConstantDataSource : public DataSource<Dtype> {
 public:
  explicit ConstantDataSource(const DataSourceParameter& param)
    : DataSource<Dtype>(param) {
    constant_ = param.constant_param().constant();
  }

  virtual length_type retrieve(index_type index,
                                 Dtype* buffer,
                                 length_type buffer_length) {
    std::fill(buffer, buffer + buffer_length, constant_);
    return buffer_length;
  }

 private:
  Dtype constant_;
};

template <typename Dtype>
class CSVDataSource : public DataSource<Dtype> {
 public:
  explicit CSVDataSource(const DataSourceParameter& param);

  virtual ~CSVDataSource();

  virtual length_type retrieve(index_type index,
                               Dtype* buffer,
                               length_type buffer_length);

  virtual std::vector<index_type> indices();

 private:
  boost::unordered_map<index_type, std::vector<Dtype> > data_;
};

template <typename Dtype>
DataSource<Dtype>* create_data_source(const DataSourceParameter& param) {
  switch (param.type()) {
  case DataSourceParameter_DataSourceType_CONSTANT:
    return new ConstantDataSource<Dtype>(param);
  case DataSourceParameter_DataSourceType_CSV:
    return new CSVDataSource<Dtype>(param);
  default:
    LOG(FATAL) << "Unknown or unimplemented data source type "
               << static_cast<int>(param.type());
  }
  return NULL;
}

}  // namespace caffe

#endif  // CAFFE_DATA_SOURCES_HPP_
