#ifndef CAFFE_DATABASE_FACTORY_H_
#define CAFFE_DATABASE_FACTORY_H_

#include <string>

#include "caffe/common.hpp"
#include "caffe/database.hpp"

namespace caffe {

shared_ptr<Database> DatabaseFactory(const DataParameter_DB& type);
shared_ptr<Database> DatabaseFactory(const string& type);

}  // namespace caffe

#endif  // CAFFE_DATABASE_FACTORY_H_
