#include <stdlib.h>

#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_sources.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class CSVDataSourceTest : public ::testing::Test {
 protected:
  std::map<index_type, std::vector<Dtype> > data_;

 protected:
  void FillRandom(int rows, int columns) {
    data_.clear();

    rng_t& engine = *caffe_rng();

    boost::uniform_int<index_type> index_dist(rows / 2, 2 * rows + 2);
    boost::uniform_real<Dtype> value_dist;

    while (data_.size() < rows) {
      index_type index = index_dist(engine);
      std::vector<Dtype>* buffer = &data_[index];
      if (!buffer->empty())
        continue;
      for (int j = 0; j < columns; ++j)
        buffer->push_back(value_dist(engine));
    }
  }

  void ToCSV(const char* filename) {
    std::ofstream output(filename);
    CHECK(output) << "Failed to open " << filename << " for writing";

    typedef typename std::map<index_type, std::vector<Dtype> >::const_iterator
        iterator_type;

    for (iterator_type it = data_.begin(); it != data_.end(); ++it) {
      output << it->first << ", ";
      const std::vector<Dtype>& values = it->second;
      CHECK_GT(values.size(), 0);
      for (int j = 0; j + 1 < values.size(); ++j) {
        output << values[j] << ", ";
      }
      output << values.back() << '\n';
    }
  }
};

TYPED_TEST_CASE(CSVDataSourceTest, TestDtypes);

TYPED_TEST(CSVDataSourceTest, TestReading) {
  typedef TypeParam Dtype;

  char temp_file_name[] = "/tmp/caffe_CSVDataSourceTest.XXXXXX";
  int fd = mkstemp(temp_file_name);
  CHECK_GT(fd, 0) << "Failed to create temporary file";
  close(fd);

  const int rows = 100, columns = 20;
  this->FillRandom(rows, columns);
  this->ToCSV(temp_file_name);

  DataSourceParameter param;
  param.set_type(caffe::DataSourceParameter_DataSourceType_CSV);
  param.set_filename(temp_file_name);

  DataSource<Dtype>* source = new CSVDataSource<Dtype>(param);

  std::vector<Dtype> buffer(columns);
  typedef typename std::map<index_type, std::vector<Dtype> >::const_iterator
                                                    iterator_type;

  for (iterator_type it = this->data_.begin(); it != this->data_.end(); ++it) {
    CHECK_EQ(source->retrieve(it->first, &buffer[0], columns),
             columns);
    for (int j = 0; j < columns; ++j) {
      CHECK_NEAR(it->second[j], buffer[j], 0.001);
    }
  }
}
}  // namespace caffe
