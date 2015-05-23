#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"

#include "caffe/test/test_caffe_main.hpp"

using std::ostringstream;

namespace caffe {

template <typename TypeParam>
class SolverTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  virtual void InitSolverFromProtoString(const string& proto) {
    SolverParameter param;
    CHECK(google::protobuf::TextFormat::ParseFromString(proto, &param));
    // Set the solver_mode according to current Caffe::mode.
    switch (Caffe::mode()) {
      case Caffe::CPU:
        param.set_solver_mode(SolverParameter_SolverMode_CPU);
        break;
      case Caffe::GPU:
        param.set_solver_mode(SolverParameter_SolverMode_GPU);
        break;
      default:
        LOG(FATAL) << "Unknown Caffe mode: " << Caffe::mode();
    }
    solver_.reset(new SGDSolver<Dtype>(param));
  }

  shared_ptr<Solver<Dtype> > solver_;
};

TYPED_TEST_CASE(SolverTest, TestDtypesAndDevices);

TYPED_TEST(SolverTest, TestInitTrainTestNets) {
  const string& proto =
     "test_interval: 10 "
     "test_iter: 10 "
     "test_state: { stage: 'with-softmax' }"
     "test_iter: 10 "
     "test_state: {}"
     "net_param { "
     "  name: 'TestNetwork' "
     "  layer { "
     "    name: 'data' "
     "    type: 'DummyData' "
     "    dummy_data_param { "
     "      shape { "
     "        dim: 5 "
     "        dim: 2 "
     "        dim: 3 "
     "        dim: 4 "
     "      } "
     "      shape { "
     "        dim: 5 "
     "      } "
     "    } "
     "    top: 'data' "
     "    top: 'label' "
     "  } "
     "  layer { "
     "    name: 'innerprod' "
     "    type: 'InnerProduct' "
     "    inner_product_param { "
     "      num_output: 10 "
     "    } "
     "    bottom: 'data' "
     "    top: 'innerprod' "
     "  } "
     "  layer { "
     "    name: 'accuracy' "
     "    type: 'Accuracy' "
     "    bottom: 'innerprod' "
     "    bottom: 'label' "
     "    top: 'accuracy' "
     "    exclude: { phase: TRAIN } "
     "  } "
     "  layer { "
     "    name: 'loss' "
     "    type: 'SoftmaxWithLoss' "
     "    bottom: 'innerprod' "
     "    bottom: 'label' "
     "    include: { phase: TRAIN } "
     "    include: { phase: TEST stage: 'with-softmax' } "
     "  } "
     "} ";
  this->InitSolverFromProtoString(proto);
  ASSERT_TRUE(this->solver_->net() != NULL);
  EXPECT_TRUE(this->solver_->net()->has_layer("loss"));
  EXPECT_FALSE(this->solver_->net()->has_layer("accuracy"));
  ASSERT_EQ(2, this->solver_->test_nets().size());
  EXPECT_TRUE(this->solver_->test_nets()[0]->has_layer("loss"));
  EXPECT_TRUE(this->solver_->test_nets()[0]->has_layer("accuracy"));
  EXPECT_FALSE(this->solver_->test_nets()[1]->has_layer("loss"));
  EXPECT_TRUE(this->solver_->test_nets()[1]->has_layer("accuracy"));
}

}  // namespace caffe
