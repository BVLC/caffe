#include <boost/assign.hpp>
#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include "caffe/multinode/BlobComms.hpp"

namespace caffe {
namespace {

using ::testing::_;
using ::testing::Return;
using ::testing::Test;
using ::testing::StrictMock;
using ::testing::Mock;
using ::testing::InSequence;
using boost::assign::list_of;


}  // namespace
}  // namespace caffe
