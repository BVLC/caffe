#ifndef CAFFE_DATABASE_FACTORY_H_
#define CAFFE_DATABASE_FACTORY_H_

#include <string>

#include "caffe/common.hpp"
#include "caffe/database.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename K, typename V>
shared_ptr<Database<K, V> > DatabaseFactory(const DataParameter_DB& type);

template <typename K, typename V>
shared_ptr<Database<K, V> > DatabaseFactory(const string& type);

}  // namespace caffe

#endif  // CAFFE_DATABASE_FACTORY_H_
