#ifndef CAFFE_DATA_SOURCES_HPP_
#define CAFFE_DATA_SOURCES_HPP_

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

  virtual std::vector<index_type> indices() = 0;

 private:
  DataSourceParameter data_source_param_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_SOURCES_HPP_
