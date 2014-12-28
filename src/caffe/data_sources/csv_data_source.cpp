#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>
#include <boost/unordered_map.hpp>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_sources.hpp"

namespace caffe {

template <typename Dtype>
CSVDataSource<Dtype>::CSVDataSource(const DataSourceParameter& param)
  : DataSource<Dtype>(param) {
  std::ifstream input(param.filename().c_str());
  CHECK(input) << "Opening file " << param.filename() << " failed";

  std::string line;
  typedef boost::char_separator<char> separator_type;
  typedef boost::tokenizer<separator_type> tokenizer_type;
  typedef typename tokenizer_type::iterator iterator_type;

  try {
    separator_type separator(", \t");
    while (std::getline(input, line)) {
      tokenizer_type tokenizer(line, separator);
      iterator_type it = tokenizer.begin();
      if (it != tokenizer.end()) {
        index_type index = boost::lexical_cast<index_type>(*it);
        std::vector<Dtype>* current_data = &data_[index];
        CHECK(current_data->empty()) << "Duplicate index " << index;

        while (++it != tokenizer.end()) {
          current_data->push_back(boost::lexical_cast<Dtype>(*it));
        }
      }
    }
  } catch (const boost::bad_lexical_cast& e) {
    LOG(FATAL) << e.what();
  }
}

template <typename Dtype>
CSVDataSource<Dtype>::~CSVDataSource() {}

template <typename Dtype>
length_type CSVDataSource<Dtype>::retrieve(index_type index,
                                           Dtype* buffer,
                                           length_type buffer_length) {
  const std::vector<Dtype>& current_data = data_[index];
  length_type copy_length =
      std::min<length_type>(current_data.size(), buffer_length);
  std::copy(current_data.begin(), current_data.begin() + copy_length, buffer);
  return current_data.size();
}

template <typename Dtype>
std::vector<index_type> CSVDataSource<Dtype>::indices() {
  std::vector<index_type> result;
  result.reserve(data_.size());
  for (typename boost::unordered_map<index_type, std::vector<Dtype> >::iterator
       it = data_.begin(); it != data_.end(); ++it) {
    result.push_back(it->first);
  }
  return result;
}

INSTANTIATE_CLASS(CSVDataSource);

}  // namespace caffe
