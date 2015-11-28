#ifndef CAFFE_DATASET_FACTORY_H_
#define CAFFE_DATASET_FACTORY_H_

#include <string>

#include "caffe/common.hpp"
#include "caffe/dataset.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename K, typename V>
shared_ptr<Dataset<K, V> > DatasetFactory(const DataParameter_DB& type);

template <typename K, typename V>
shared_ptr<Dataset<K, V> > DatasetFactory(const string& type);

}  // namespace caffe

#endif  // CAFFE_DATASET_FACTORY_H_
