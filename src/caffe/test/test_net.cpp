// Copyright 2014 BVLC and contributors.

#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/text_format.h"

#include "gtest/gtest.h"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

using std::ostringstream;

namespace caffe {

template <typename TypeParam>
class NetTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  NetTest() : seed_(1701) {}

  virtual void InitNetFromProtoString(const string& proto) {
    NetParameter param;
    CHECK(google::protobuf::TextFormat::ParseFromString(proto, &param));
    net_.reset(new Net<Dtype>(param));
  }

  virtual void InitTinyNet(const bool force_backward = false) {
    string proto =
        "name: 'TinyTestNetwork' "
        "layers: { "
        "  name: 'data' "
        "  type: DUMMY_DATA "
        "  dummy_data_param { "
        "    num: 5 "
        "    channels: 2 "
        "    height: 3 "
        "    width: 4 "
        "    num: 5 "
        "    channels: 1 "
        "    height: 1 "
        "    width: 1 "
        "    data_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "  } "
        "  top: 'data' "
        "  top: 'label' "
        "} "
        "layers: { "
        "  name: 'innerproduct' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 1000 "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "    bias_filler { "
        "      type: 'constant' "
        "      value: 0 "
        "    } "
        "  } "
        "  blobs_lr: 1. "
        "  blobs_lr: 2. "
        "  weight_decay: 1. "
        "  weight_decay: 0. "
        "  bottom: 'data' "
        "  top: 'innerproduct' "
        "} "
        "layers: { "
        "  name: 'loss' "
        "  type: SOFTMAX_LOSS "
        "  bottom: 'innerproduct' "
        "  bottom: 'label' "
        "  top: 'top_loss' "
        "} ";
    if (force_backward) {
      proto += "force_backward: true ";
    }
    InitNetFromProtoString(proto);
  }

  virtual void InitTinyNetEuclidean(const bool force_backward = false) {
    string proto =
        "name: 'TinyTestEuclidLossNetwork' "
        "layers: { "
        "  name: 'data' "
        "  type: DUMMY_DATA "
        "  dummy_data_param { "
        "    num: 5 "
        "    channels: 2 "
        "    height: 3 "
        "    width: 4 "
        "    num: 5 "
        "    channels: 1 "
        "    height: 1 "
        "    width: 1 "
        "    data_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "  } "
        "  top: 'data' "
        "  top: 'label' "
        "} "
        "layers: { "
        "  name: 'innerproduct' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 1 "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "    bias_filler { "
        "      type: 'constant' "
        "      value: 0 "
        "    } "
        "  } "
        "  blobs_lr: 1. "
        "  blobs_lr: 2. "
        "  weight_decay: 1. "
        "  weight_decay: 0. "
        "  bottom: 'data' "
        "  top: 'innerproduct' "
        "} "
        "layers: { "
        "  name: 'loss' "
        "  type: EUCLIDEAN_LOSS "
        "  bottom: 'innerproduct' "
        "  bottom: 'label' "
        "} ";
    if (force_backward) {
      proto += "force_backward: true ";
    }
    InitNetFromProtoString(proto);
  }

  virtual void InitTrickyNet() {
    const string& proto =
        "name: 'TrickyTestNetwork' "
        "layers: { "
        "  name: 'data' "
        "  type: DUMMY_DATA "
        "  dummy_data_param { "
        "    num: 5 "
        "    channels: 2 "
        "    height: 3 "
        "    width: 4 "
        "    num: 5 "
        "    channels: 1 "
        "    height: 1 "
        "    width: 1 "
        "    data_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "  } "
        "  top: 'data' "
        "  top: 'label' "
        "} "
        "layers: { "
        "  name: 'innerproduct' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 1000 "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "    bias_filler { "
        "      type: 'constant' "
        "      value: 0 "
        "    } "
        "  } "
        "  blobs_lr: 1. "
        "  blobs_lr: 2. "
        "  weight_decay: 1. "
        "  weight_decay: 0. "
        "  bottom: 'data' "
        "  top: 'transformed_data' "
        "} "
        "layers: { "
        "  name: 'innerproduct' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 1 "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "    bias_filler { "
        "      type: 'constant' "
        "      value: 0 "
        "    } "
        "  } "
        "  blobs_lr: 1. "
        "  blobs_lr: 2. "
        "  weight_decay: 1. "
        "  weight_decay: 0. "
        "  bottom: 'label' "
        "  top: 'transformed_label' "
        "} "
        "layers: { "
        "  name: 'loss' "
        "  type: SOFTMAX_LOSS "
        "  bottom: 'transformed_data' "
        "  bottom: 'transformed_label' "
        "} ";
    InitNetFromProtoString(proto);
  }

  virtual void InitUnsharedWeightsNet(const bool bias_term = false,
      const Dtype blobs_lr_w1 = 1, const Dtype blobs_lr_b1 = 2,
      const Dtype blobs_lr_w2 = 1, const Dtype blobs_lr_b2 = 2) {
    ostringstream proto;
    proto <<
        "name: 'UnsharedWeightsNetwork' "
        "layers: { "
        "  name: 'data' "
        "  type: DUMMY_DATA "
        "  dummy_data_param { "
        "    num: 5 "
        "    channels: 2 "
        "    height: 3 "
        "    width: 4 "
        "    data_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "  } "
        "  top: 'data' "
        "} "
        "layers: { "
        "  name: 'innerproduct1' "
       "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: " << bias_term <<
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 10 "
        "    } "
        "  } "
        "  param: 'unsharedweights1' ";
    if (bias_term) {
      proto << "  param: '' ";
    }
    proto <<
        "  blobs_lr: " << blobs_lr_w1;
    if (bias_term) {
      proto << "  blobs_lr: " << blobs_lr_b1;
    }
    proto <<
        "  bottom: 'data' "
        "  top: 'innerproduct1' "
        "} "
        "layers: { "
        "  name: 'innerproduct2' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: " << bias_term <<
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 10 "
        "    } "
        "  } "
        "  param: 'unsharedweights2' ";
    if (bias_term) {
      proto << "  param: '' ";
    }
    proto <<
        "  bottom: 'data' "
        "  blobs_lr: " << blobs_lr_w2;
    if (bias_term) {
      proto << "  blobs_lr: " << blobs_lr_b2;
    }
    proto <<
        "  top: 'innerproduct2' "
        "} "
        "layers: { "
        "  name: 'loss' "
        "  type: EUCLIDEAN_LOSS "
        "  bottom: 'innerproduct1' "
        "  bottom: 'innerproduct2' "
        "} ";
    InitNetFromProtoString(proto.str());
  }

  virtual void InitSharedWeightsNet() {
    const string& proto =
        "name: 'SharedWeightsNetwork' "
        "layers: { "
        "  name: 'data' "
        "  type: DUMMY_DATA "
        "  dummy_data_param { "
        "    num: 5 "
        "    channels: 2 "
        "    height: 3 "
        "    width: 4 "
        "    data_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "  } "
        "  top: 'data' "
        "} "
        "layers: { "
        "  name: 'innerproduct1' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 10 "
        "    } "
        "  } "
        "  param: 'sharedweights' "
        "  bottom: 'data' "
        "  top: 'innerproduct1' "
        "} "
        "layers: { "
        "  name: 'innerproduct2' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 10 "
        "    } "
        "  } "
        "  param: 'sharedweights' "
        "  bottom: 'data' "
        "  top: 'innerproduct2' "
        "} "
        "layers: { "
        "  name: 'loss' "
        "  type: EUCLIDEAN_LOSS "
        "  bottom: 'innerproduct1' "
        "  bottom: 'innerproduct2' "
        "} ";
    InitNetFromProtoString(proto);
  }

  virtual void InitDiffDataUnsharedWeightsNet() {
    const string& proto =
        "name: 'DiffDataUnsharedWeightsNetwork' "
        "layers: { "
        "  name: 'data' "
        "  type: DUMMY_DATA "
        "  dummy_data_param { "
        "    num: 10 "
        "    channels: 10 "
        "    height: 1 "
        "    width: 1 "
        "    num: 10 "
        "    channels: 10 "
        "    height: 1 "
        "    width: 1 "
        "    data_filler { "
        "      type: 'gaussian' "
        "      std: 10 "
        "    } "
        "  } "
        "  top: 'data1' "
        "  top: 'data2' "
        "} "
        "layers: { "
        "  name: 'innerproduct1' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'constant' "
        "      value: 0.5 "
        "    } "
        "  } "
        "  param: 'unsharedweights1' "
        "  bottom: 'data1' "
        "  top: 'innerproduct1' "
        "} "
        "layers: { "
        "  name: 'innerproduct2' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'constant' "
        "      value: 0.5 "
        "    } "
        "  } "
        "  param: 'unsharedweights2' "
        "  bottom: 'innerproduct1' "
        "  top: 'innerproduct2' "
        "} "
        "layers: { "
        "  name: 'loss' "
        "  type: EUCLIDEAN_LOSS "
        "  bottom: 'data2' "
        "  bottom: 'innerproduct2' "
        "} ";
    InitNetFromProtoString(proto);
  }

  virtual void InitDiffDataSharedWeightsNet() {
    const string& proto =
        "name: 'DiffDataSharedWeightsNetwork' "
        "layers: { "
        "  name: 'data' "
        "  type: DUMMY_DATA "
        "  dummy_data_param { "
        "    num: 10 "
        "    channels: 10 "
        "    height: 1 "
        "    width: 1 "
        "    num: 10 "
        "    channels: 10 "
        "    height: 1 "
        "    width: 1 "
        "    data_filler { "
        "      type: 'gaussian' "
        "      std: 10 "
        "    } "
        "  } "
        "  top: 'data1' "
        "  top: 'data2' "
        "} "
        "layers: { "
        "  name: 'innerproduct1' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'constant' "
        "      value: 0.5 "
        "    } "
        "  } "
        "  param: 'sharedweights' "
        "  bottom: 'data1' "
        "  top: 'innerproduct1' "
        "} "
        "layers: { "
        "  name: 'innerproduct2' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'constant' "
        "      value: 0.5 "
        "    } "
        "  } "
        "  param: 'sharedweights' "
        "  bottom: 'innerproduct1' "
        "  top: 'innerproduct2' "
        "} "
        "layers: { "
        "  name: 'loss' "
        "  type: EUCLIDEAN_LOSS "
        "  bottom: 'data2' "
        "  bottom: 'innerproduct2' "
        "} ";
    InitNetFromProtoString(proto);
  }

  int seed_;
  shared_ptr<Net<Dtype> > net_;
};

TYPED_TEST_CASE(NetTest, TestDtypesAndDevices);

TYPED_TEST(NetTest, TestHasBlob) {
  this->InitTinyNet();
  EXPECT_TRUE(this->net_->has_blob("data"));
  EXPECT_TRUE(this->net_->has_blob("label"));
  EXPECT_TRUE(this->net_->has_blob("innerproduct"));
  EXPECT_FALSE(this->net_->has_blob("loss"));
  EXPECT_TRUE(this->net_->has_blob("top_loss"));
}

TYPED_TEST(NetTest, TestGetBlob) {
  this->InitTinyNet();
  EXPECT_EQ(this->net_->blob_by_name("data"), this->net_->blobs()[0]);
  EXPECT_EQ(this->net_->blob_by_name("label"), this->net_->blobs()[1]);
  EXPECT_EQ(this->net_->blob_by_name("innerproduct"), this->net_->blobs()[2]);
  EXPECT_FALSE(this->net_->blob_by_name("loss"));
  EXPECT_EQ(this->net_->blob_by_name("top_loss"), this->net_->blobs()[3]);
}

TYPED_TEST(NetTest, TestHasLayer) {
  this->InitTinyNet();
  EXPECT_TRUE(this->net_->has_layer("data"));
  EXPECT_TRUE(this->net_->has_layer("innerproduct"));
  EXPECT_TRUE(this->net_->has_layer("loss"));
  EXPECT_FALSE(this->net_->has_layer("label"));
}

TYPED_TEST(NetTest, TestGetLayerByName) {
  this->InitTinyNet();
  EXPECT_EQ(this->net_->layer_by_name("data"), this->net_->layers()[0]);
  EXPECT_EQ(this->net_->layer_by_name("innerproduct"), this->net_->layers()[1]);
  EXPECT_EQ(this->net_->layer_by_name("loss"), this->net_->layers()[2]);
  EXPECT_FALSE(this->net_->layer_by_name("label"));
}

TYPED_TEST(NetTest, TestBottomNeedBackward) {
  this->InitTinyNet();
  const vector<vector<bool> >& bottom_need_backward =
      this->net_->bottom_need_backward();
  EXPECT_EQ(3, bottom_need_backward.size());
  EXPECT_EQ(0, bottom_need_backward[0].size());
  EXPECT_EQ(1, bottom_need_backward[1].size());
  EXPECT_EQ(false, bottom_need_backward[1][0]);
  EXPECT_EQ(2, bottom_need_backward[2].size());
  EXPECT_EQ(true, bottom_need_backward[2][0]);
  EXPECT_EQ(false, bottom_need_backward[2][1]);
}

TYPED_TEST(NetTest, TestBottomNeedBackwardForce) {
  typedef typename TypeParam::Dtype Dtype;
  const bool force_backward = true;
  this->InitTinyNet(force_backward);
  const vector<vector<bool> >& bottom_need_backward =
      this->net_->bottom_need_backward();
  EXPECT_EQ(3, bottom_need_backward.size());
  EXPECT_EQ(0, bottom_need_backward[0].size());
  EXPECT_EQ(1, bottom_need_backward[1].size());
  EXPECT_EQ(true, bottom_need_backward[1][0]);
  EXPECT_EQ(2, bottom_need_backward[2].size());
  EXPECT_EQ(true, bottom_need_backward[2][0]);
  EXPECT_EQ(false, bottom_need_backward[2][1]);
}

TYPED_TEST(NetTest, TestBottomNeedBackwardEuclideanForce) {
  typedef typename TypeParam::Dtype Dtype;
  const bool force_backward = true;
  this->InitTinyNetEuclidean(force_backward);
  const vector<vector<bool> >& bottom_need_backward =
      this->net_->bottom_need_backward();
  EXPECT_EQ(3, bottom_need_backward.size());
  EXPECT_EQ(0, bottom_need_backward[0].size());
  EXPECT_EQ(1, bottom_need_backward[1].size());
  EXPECT_EQ(true, bottom_need_backward[1][0]);
  EXPECT_EQ(2, bottom_need_backward[2].size());
  EXPECT_EQ(true, bottom_need_backward[2][0]);
  EXPECT_EQ(true, bottom_need_backward[2][1]);
}

TYPED_TEST(NetTest, TestBottomNeedBackwardTricky) {
  this->InitTrickyNet();
  const vector<vector<bool> >& bottom_need_backward =
      this->net_->bottom_need_backward();
  EXPECT_EQ(4, bottom_need_backward.size());
  EXPECT_EQ(0, bottom_need_backward[0].size());
  EXPECT_EQ(1, bottom_need_backward[1].size());
  EXPECT_EQ(false, bottom_need_backward[1][0]);
  EXPECT_EQ(1, bottom_need_backward[2].size());
  EXPECT_EQ(false, bottom_need_backward[2][0]);
  EXPECT_EQ(2, bottom_need_backward[3].size());
  EXPECT_EQ(true, bottom_need_backward[3][0]);
  // The label input to the SoftmaxLossLayer should say it "needs backward"
  // since it has weights under it, even though we expect this to cause a crash
  // at training/test time.
  EXPECT_EQ(true, bottom_need_backward[3][1]);
}

TYPED_TEST(NetTest, TestUnsharedWeightsDataNet) {
  typedef typename TypeParam::Dtype Dtype;
  this->InitUnsharedWeightsNet();
  vector<Blob<Dtype>*> bottom;
  Dtype loss;
  this->net_->Forward(bottom, &loss);
  EXPECT_GT(loss, 0);
}

TYPED_TEST(NetTest, TestSharedWeightsDataNet) {
  typedef typename TypeParam::Dtype Dtype;
  this->InitSharedWeightsNet();
  vector<Blob<Dtype>*> bottom;
  Dtype loss;
  this->net_->Forward(bottom, &loss);
  EXPECT_FLOAT_EQ(loss, 0);
}

TYPED_TEST(NetTest, TestUnsharedWeightsDiffNet) {
  typedef typename TypeParam::Dtype Dtype;
  this->InitUnsharedWeightsNet();
  vector<Blob<Dtype>*> bottom;
  Net<Dtype>* net = this->net_.get();
  net->Forward(bottom);
  net->Backward();
  Layer<Dtype>* ip1_layer = net->layer_by_name("innerproduct1").get();
  Layer<Dtype>* ip2_layer = net->layer_by_name("innerproduct2").get();
  const int count = ip1_layer->blobs()[0]->count();
  const Dtype* grad1 = ip1_layer->blobs()[0]->cpu_diff();
  const Dtype* grad2 = ip2_layer->blobs()[0]->cpu_diff();
  for (int i = 0; i < count; ++i) {
    EXPECT_GT(fabs(grad1[i]), 0);
    EXPECT_FLOAT_EQ(-1 * grad1[i], grad2[i]);
  }
}

TYPED_TEST(NetTest, TestSharedWeightsDiffNet) {
  typedef typename TypeParam::Dtype Dtype;
  this->InitSharedWeightsNet();
  vector<Blob<Dtype>*> bottom;
  Net<Dtype>* net = this->net_.get();
  Dtype loss;
  net->Forward(bottom, &loss);
  net->Backward();
  EXPECT_FLOAT_EQ(loss, 0);
  Layer<Dtype>* ip1_layer = net->layer_by_name("innerproduct1").get();
  Layer<Dtype>* ip2_layer = net->layer_by_name("innerproduct2").get();
  const int count = ip1_layer->blobs()[0]->count();
  const Dtype* grad1 = ip1_layer->blobs()[0]->cpu_diff();
  const Dtype* grad2 = ip2_layer->blobs()[0]->cpu_diff();
  for (int i = 0; i < count; ++i) {
    EXPECT_FLOAT_EQ(0, grad1[i]);
    EXPECT_FLOAT_EQ(0, grad2[i]);
  }
}

TYPED_TEST(NetTest, TestSharedWeightsUpdate) {
  typedef typename TypeParam::Dtype Dtype;
  Caffe::set_random_seed(this->seed_);
  this->InitDiffDataSharedWeightsNet();
  vector<Blob<Dtype>*> bottom;
  EXPECT_EQ(this->net_->layer_names()[1], "innerproduct1");
  EXPECT_EQ(this->net_->layer_names()[2], "innerproduct2");
  Blob<Dtype>* ip1_weights = this->net_->layers()[1]->blobs()[0].get();
  Blob<Dtype>* ip2_weights = this->net_->layers()[2]->blobs()[0].get();
  // Check that data blobs of shared weights share the same location in memory.
  EXPECT_EQ(ip1_weights->cpu_data(), ip2_weights->cpu_data());
  // Check that diff blobs of shared weights are at different locations in
  // locations.  (The diffs should be accumulated at update time.)
  EXPECT_NE(ip1_weights->cpu_diff(), ip2_weights->cpu_diff());
  this->net_->Forward(bottom);
  this->net_->Backward();
  // Compute the expected update as the data minus the two diffs.
  Blob<Dtype> shared_params;
  const bool reshape = true;
  const bool copy_diff = false;
  shared_params.CopyFrom(*ip1_weights, copy_diff, reshape);
  shared_params.CopyFrom(*ip1_weights, !copy_diff, reshape);
  const int count = ip1_weights->count();
  // Make sure the diffs are non-trivial.
  for (int i = 0; i < count; ++i) {
    EXPECT_NE(0, ip1_weights->cpu_diff()[i]);
    EXPECT_NE(0, ip2_weights->cpu_diff()[i]);
    EXPECT_NE(ip1_weights->cpu_diff()[i], ip2_weights->cpu_diff()[i]);
  }
  caffe_axpy(count, Dtype(1), ip2_weights->cpu_diff(),
             shared_params.mutable_cpu_diff());
  caffe_axpy(count, Dtype(-1), shared_params.cpu_diff(),
             shared_params.mutable_cpu_data());
  const Dtype* expected_updated_params = shared_params.cpu_data();
  this->net_->Update();
  const Dtype* actual_updated_params = ip1_weights->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(expected_updated_params[i], actual_updated_params[i]);
  }
  // Check that data blobs of shared weights STILL point to the same memory
  // location (because ... who knows).
  EXPECT_EQ(ip1_weights->cpu_data(), ip2_weights->cpu_data());

  Caffe::set_random_seed(this->seed_);
  this->InitDiffDataUnsharedWeightsNet();
  EXPECT_EQ(this->net_->layer_names()[1], "innerproduct1");
  EXPECT_EQ(this->net_->layer_names()[2], "innerproduct2");
  ip1_weights = this->net_->layers()[1]->blobs()[0].get();
  ip2_weights = this->net_->layers()[2]->blobs()[0].get();
  // Check that data and diff blobs of unshared weights are at different
  // locations in memory.
  EXPECT_NE(ip1_weights->cpu_data(), ip2_weights->cpu_data());
  EXPECT_NE(ip1_weights->cpu_diff(), ip2_weights->cpu_diff());
  this->net_->Forward(bottom);
  this->net_->Backward();
  // Compute the expected update.
  Blob<Dtype> unshared_params1;
  unshared_params1.CopyFrom(*ip1_weights, copy_diff, reshape);
  unshared_params1.CopyFrom(*ip1_weights, !copy_diff, reshape);
  Blob<Dtype> unshared_params2;
  unshared_params2.CopyFrom(*ip2_weights, copy_diff, reshape);
  unshared_params2.CopyFrom(*ip2_weights, !copy_diff, reshape);
  // Make sure the diffs are non-trivial and sum to the diff in the shared net.
  for (int i = 0; i < count; ++i) {
    EXPECT_NE(0, ip1_weights->cpu_diff()[i]);
    EXPECT_NE(0, ip2_weights->cpu_diff()[i]);
    EXPECT_NE(ip1_weights->cpu_diff()[i], ip2_weights->cpu_diff()[i]);
    EXPECT_EQ(ip1_weights->cpu_diff()[i] + ip2_weights->cpu_diff()[i],
              shared_params.cpu_diff()[i]);
  }
  caffe_axpy(count, Dtype(-1), ip1_weights->cpu_diff(),
             unshared_params1.mutable_cpu_data());
  caffe_axpy(count, Dtype(-1), ip2_weights->cpu_diff(),
             unshared_params2.mutable_cpu_data());
  const Dtype* expected_updated_params1 = unshared_params1.cpu_data();
  const Dtype* expected_updated_params2 = unshared_params2.cpu_data();
  this->net_->Update();
  const Dtype* actual_updated_params1 = ip1_weights->cpu_data();
  const Dtype* actual_updated_params2 = ip2_weights->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(expected_updated_params1[i], actual_updated_params1[i]);
    EXPECT_EQ(expected_updated_params2[i], actual_updated_params2[i]);
    EXPECT_NE(actual_updated_params1[i], actual_updated_params2[i]);
    EXPECT_NE(expected_updated_params, expected_updated_params1);
  }
}

TYPED_TEST(NetTest, TestParamPropagateDown) {
  typedef typename TypeParam::Dtype Dtype;
  vector<Blob<Dtype>*> bottom;
  const bool kBiasTerm = true;

  // Run the net with all params learned; check that gradients are non-zero.
  Caffe::set_random_seed(this->seed_);
  Dtype blobs_lr_w1 = 1, blobs_lr_w2 = 1, blobs_lr_b1 = 2, blobs_lr_b2 = 2;
  this->InitUnsharedWeightsNet(kBiasTerm, blobs_lr_w1, blobs_lr_w2,
                               blobs_lr_b1, blobs_lr_b2);
  this->net_->Forward(bottom);
  this->net_->Backward();
  const vector<shared_ptr<Blob<Dtype> > >& params = this->net_->params();
  const int num_params = params.size();
  ASSERT_EQ(4, num_params);
  const Dtype kNonZeroTestMin = 1e-3;
  vector<Dtype> param_asums(params.size());
  for (int i = 0; i < num_params; ++i) {
    const Dtype param_asum =
       caffe_cpu_asum(params[i]->count(), params[i]->cpu_diff());
    param_asums[i] = param_asum;
    EXPECT_GT(param_asum, kNonZeroTestMin);
  }

  // Change the learning rates to different non-zero values; should see same
  // gradients.
  Caffe::set_random_seed(this->seed_);
  blobs_lr_w1 *= 2, blobs_lr_w2 *= 2, blobs_lr_b1 *= 2, blobs_lr_b2 *= 2;
  this->InitUnsharedWeightsNet(kBiasTerm, blobs_lr_w1, blobs_lr_w2,
                               blobs_lr_b1, blobs_lr_b2);
  this->net_->Forward(bottom);
  this->net_->Backward();
  const vector<shared_ptr<Blob<Dtype> > >& params2 = this->net_->params();
  ASSERT_EQ(num_params, params2.size());
  for (int i = 0; i < num_params; ++i) {
    const Dtype param_asum =
       caffe_cpu_asum(params2[i]->count(), params2[i]->cpu_diff());
    EXPECT_EQ(param_asum, param_asums[i]);
  }

  // Change a subset of the learning rates to zero; check that we see zero
  // gradients for those.
  Caffe::set_random_seed(this->seed_);
  blobs_lr_w1 = 1, blobs_lr_w2 = 0, blobs_lr_b1 = 0, blobs_lr_b2 = 1;
  this->InitUnsharedWeightsNet(kBiasTerm, blobs_lr_w1, blobs_lr_w2,
                               blobs_lr_b1, blobs_lr_b2);
  this->net_->Forward(bottom);
  this->net_->Backward();
  const vector<shared_ptr<Blob<Dtype> > >& params3 = this->net_->params();
  ASSERT_EQ(num_params, params3.size());
  for (int i = 0; i < num_params; ++i) {
    const Dtype param_asum =
       caffe_cpu_asum(params3[i]->count(), params3[i]->cpu_diff());
    if (i == 1 || i == 2) {
      EXPECT_EQ(0, param_asum);
    } else {
      EXPECT_EQ(param_asum, param_asums[i]);
    }
  }

  // Change the opposite subset of the learning rates to zero.
  Caffe::set_random_seed(this->seed_);
  blobs_lr_w1 = 0, blobs_lr_w2 = 1, blobs_lr_b1 = 1, blobs_lr_b2 = 0;
  this->InitUnsharedWeightsNet(kBiasTerm, blobs_lr_w1, blobs_lr_w2,
                               blobs_lr_b1, blobs_lr_b2);
  this->net_->Forward(bottom);
  this->net_->Backward();
  const vector<shared_ptr<Blob<Dtype> > >& params4 = this->net_->params();
  ASSERT_EQ(num_params, params4.size());
  for (int i = 0; i < num_params; ++i) {
    const Dtype param_asum =
       caffe_cpu_asum(params4[i]->count(), params4[i]->cpu_diff());
    if (i == 0 || i == 3) {
      EXPECT_EQ(0, param_asum);
    } else {
      EXPECT_EQ(param_asum, param_asums[i]);
    }
  }
}

TYPED_TEST(NetTest, TestFromTo) {
  typedef typename TypeParam::Dtype Dtype;
  this->InitTinyNet();

  // Run Forward and Backward, recording the data diff and loss.
  Blob<Dtype> data;
  data.ReshapeLike(*this->net_->blob_by_name("data"));
  this->net_->ForwardPrefilled();
  this->net_->Backward();
  data.CopyFrom(*this->net_->blob_by_name("data"), true, true);
  const Dtype *loss_ptr = this->net_->output_blobs()[0]->cpu_data();
  Dtype loss = *loss_ptr;

  // Check that combining partial Forwards gives the same loss.
  for (int i = 1; i < this->net_->layers().size(); ++i) {
    // Note that we skip layer zero to keep the same data.
    this->net_->ForwardFromTo(1, 1);
    if (i < this->net_->layers().size() - 1) {
      this->net_->ForwardFrom(i + 1);
    }
    EXPECT_EQ(loss, *loss_ptr);
  }

  // Check that combining partial Backwards gives the same data diff.
  for (int i = 1; i < this->net_->layers().size(); ++i) {
    this->net_->BackwardTo(i);
    this->net_->BackwardFrom(i - 1);
    for (int j = 0; j < data.count(); ++j) {
      EXPECT_EQ(data.cpu_diff()[j],
          this->net_->blob_by_name("data")->cpu_diff()[j]);
    }
  }
}

class FilterNetTest : public ::testing::Test {
 protected:
  void RunFilterNetTest(
      const string& input_param_string, const string& filtered_param_string) {
    NetParameter input_param;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        input_param_string, &input_param));
    NetParameter expected_filtered_param;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        filtered_param_string, &expected_filtered_param));
    NetParameter actual_filtered_param;
    Net<float>::FilterNet(input_param, &actual_filtered_param);
    EXPECT_EQ(expected_filtered_param.DebugString(),
        actual_filtered_param.DebugString());
    // Also test idempotence.
    NetParameter double_filtered_param;
    Net<float>::FilterNet(actual_filtered_param, &double_filtered_param);
    EXPECT_EQ(actual_filtered_param.DebugString(),
       double_filtered_param.DebugString());
  }
};

TEST_F(FilterNetTest, TestNoFilter) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'innerprod' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunFilterNetTest(input_proto, input_proto);
}

TEST_F(FilterNetTest, TestFilterLeNetTrainTest) {
  const string& input_proto =
      "name: 'LeNet' "
      "layers { "
      "  name: 'mnist' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "  data_param { "
      "    source: 'mnist-train-leveldb' "
      "    scale: 0.00390625 "
      "    batch_size: 64 "
      "  } "
      "  include: { phase: TRAIN } "
      "} "
      "layers { "
      "  name: 'mnist' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "  data_param { "
      "    source: 'mnist-test-leveldb' "
      "    scale: 0.00390625 "
      "    batch_size: 100 "
      "  } "
      "  include: { phase: TEST } "
      "} "
      "layers { "
      "  name: 'conv1' "
      "  type: CONVOLUTION "
      "  bottom: 'data' "
      "  top: 'conv1' "
      "  blobs_lr: 1 "
      "  blobs_lr: 2 "
      "  convolution_param { "
      "    num_output: 20 "
      "    kernel_size: 5 "
      "    stride: 1 "
      "    weight_filler { "
      "      type: 'xavier' "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "    } "
      "  } "
      "} "
      "layers { "
      "  name: 'ip1' "
      "  type: INNER_PRODUCT "
      "  bottom: 'conv1' "
      "  top: 'ip1' "
      "  blobs_lr: 1 "
      "  blobs_lr: 2 "
      "  inner_product_param { "
      "    num_output: 10 "
      "    weight_filler { "
      "      type: 'xavier' "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "    } "
      "  } "
      "} "
      "layers { "
      "  name: 'accuracy' "
      "  type: ACCURACY "
      "  bottom: 'ip1' "
      "  bottom: 'label' "
      "  top: 'accuracy' "
      "  include: { phase: TEST } "
      "} "
      "layers { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'ip2' "
      "  bottom: 'label' "
      "  top: 'loss' "
      "} ";
  const string input_proto_train = "state: { phase: TRAIN } " + input_proto;
  const string input_proto_test = "state: { phase: TEST } " + input_proto;
  const string& output_proto_train =
      "state: { phase: TRAIN } "
      "name: 'LeNet' "
      "layers { "
      "  name: 'mnist' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "  data_param { "
      "    source: 'mnist-train-leveldb' "
      "    scale: 0.00390625 "
      "    batch_size: 64 "
      "  } "
      "  include: { phase: TRAIN } "
      "} "
      "layers { "
      "  name: 'conv1' "
      "  type: CONVOLUTION "
      "  bottom: 'data' "
      "  top: 'conv1' "
      "  blobs_lr: 1 "
      "  blobs_lr: 2 "
      "  convolution_param { "
      "    num_output: 20 "
      "    kernel_size: 5 "
      "    stride: 1 "
      "    weight_filler { "
      "      type: 'xavier' "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "    } "
      "  } "
      "} "
      "layers { "
      "  name: 'ip1' "
      "  type: INNER_PRODUCT "
      "  bottom: 'conv1' "
      "  top: 'ip1' "
      "  blobs_lr: 1 "
      "  blobs_lr: 2 "
      "  inner_product_param { "
      "    num_output: 10 "
      "    weight_filler { "
      "      type: 'xavier' "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "    } "
      "  } "
      "} "
      "layers { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'ip2' "
      "  bottom: 'label' "
      "  top: 'loss' "
      "} ";
  const string& output_proto_test =
      "state: { phase: TEST } "
      "name: 'LeNet' "
      "layers { "
      "  name: 'mnist' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "  data_param { "
      "    source: 'mnist-test-leveldb' "
      "    scale: 0.00390625 "
      "    batch_size: 100 "
      "  } "
      "  include: { phase: TEST } "
      "} "
      "layers { "
      "  name: 'conv1' "
      "  type: CONVOLUTION "
      "  bottom: 'data' "
      "  top: 'conv1' "
      "  blobs_lr: 1 "
      "  blobs_lr: 2 "
      "  convolution_param { "
      "    num_output: 20 "
      "    kernel_size: 5 "
      "    stride: 1 "
      "    weight_filler { "
      "      type: 'xavier' "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "    } "
      "  } "
      "} "
      "layers { "
      "  name: 'ip1' "
      "  type: INNER_PRODUCT "
      "  bottom: 'conv1' "
      "  top: 'ip1' "
      "  blobs_lr: 1 "
      "  blobs_lr: 2 "
      "  inner_product_param { "
      "    num_output: 10 "
      "    weight_filler { "
      "      type: 'xavier' "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "    } "
      "  } "
      "} "
      "layers { "
      "  name: 'accuracy' "
      "  type: ACCURACY "
      "  bottom: 'ip1' "
      "  bottom: 'label' "
      "  top: 'accuracy' "
      "  include: { phase: TEST } "
      "} "
      "layers { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'ip2' "
      "  bottom: 'label' "
      "  top: 'loss' "
      "} ";
  this->RunFilterNetTest(input_proto_train, output_proto_train);
  this->RunFilterNetTest(input_proto_test, output_proto_test);
}

TEST_F(FilterNetTest, TestFilterOutByStage) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "  include: { stage: 'mystage' } "
      "} "
      "layers: { "
      "  name: 'innerprod' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  const string& output_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'innerprod' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunFilterNetTest(input_proto, output_proto);
}

TEST_F(FilterNetTest, TestFilterOutByStage2) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'innerprod' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { stage: 'mystage' } "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  const string& output_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunFilterNetTest(input_proto, output_proto);
}

TEST_F(FilterNetTest, TestFilterInByStage) {
  const string& input_proto =
      "state: { stage: 'mystage' } "
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'innerprod' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { stage: 'mystage' } "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunFilterNetTest(input_proto, input_proto);
}

TEST_F(FilterNetTest, TestFilterInByStage2) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'innerprod' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  exclude: { stage: 'mystage' } "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunFilterNetTest(input_proto, input_proto);
}

TEST_F(FilterNetTest, TestFilterOutByMultipleStage) {
  const string& input_proto =
      "state: { stage: 'mystage' } "
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'innerprod' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { stage: 'mystage' stage: 'myotherstage' } "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "  include: { stage: 'mystage' } "
      "} ";
  const string& output_proto =
      "state: { stage: 'mystage' } "
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "  include: { stage: 'mystage' } "
      "} ";
  this->RunFilterNetTest(input_proto, output_proto);
}

TEST_F(FilterNetTest, TestFilterInByMultipleStage) {
  const string& input_proto =
      "state: { stage: 'mystage' } "
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'innerprod' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { stage: 'myotherstage' } "
      "  include: { stage: 'mystage' } "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "  include: { stage: 'mystage' } "
      "} ";
  this->RunFilterNetTest(input_proto, input_proto);
}

TEST_F(FilterNetTest, TestFilterInByMultipleStage2) {
  const string& input_proto =
      "state: { stage: 'mystage' stage: 'myotherstage' } "
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'innerprod' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { stage: 'mystage' stage: 'myotherstage' } "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "  include: { stage: 'mystage' } "
      "} ";
  this->RunFilterNetTest(input_proto, input_proto);
}

TEST_F(FilterNetTest, TestFilterOutByMinLevel) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'innerprod' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { min_level: 3 } "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  const string& output_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunFilterNetTest(input_proto, output_proto);
}

TEST_F(FilterNetTest, TestFilterOutByMaxLevel) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'innerprod' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { max_level: -3 } "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  const string& output_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunFilterNetTest(input_proto, output_proto);
}

TEST_F(FilterNetTest, TestFilterInByMinLevel) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'innerprod' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { min_level: 0 } "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunFilterNetTest(input_proto, input_proto);
}

TEST_F(FilterNetTest, TestFilterInByMinLevel2) {
  const string& input_proto =
      "state: { level: 7 } "
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'innerprod' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { min_level: 3 } "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunFilterNetTest(input_proto, input_proto);
}

TEST_F(FilterNetTest, TestFilterInByMaxLevel) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'innerprod' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { max_level: 0 } "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunFilterNetTest(input_proto, input_proto);
}

TEST_F(FilterNetTest, TestFilterInByMaxLevel2) {
  const string& input_proto =
      "state: { level: -7 } "
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'innerprod' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { max_level: -3 } "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunFilterNetTest(input_proto, input_proto);
}

TEST_F(FilterNetTest, TestFilterInOutByIncludeMultiRule) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'innerprod' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { min_level: 2  phase: TRAIN } "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "  include: { min_level: 2  phase: TEST } "
      "} ";
  const string& input_proto_train =
      "state: { level: 4  phase: TRAIN } " + input_proto;
  const string& input_proto_test =
      "state: { level: 4  phase: TEST } " + input_proto;
  const string& output_proto_train =
      "state: { level: 4  phase: TRAIN } "
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'innerprod' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { min_level: 2  phase: TRAIN } "
      "} ";
  const string& output_proto_test =
      "state: { level: 4  phase: TEST } "
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "  include: { min_level: 2  phase: TEST } "
      "} ";
  this->RunFilterNetTest(input_proto_train, output_proto_train);
  this->RunFilterNetTest(input_proto_test, output_proto_test);
}

TEST_F(FilterNetTest, TestFilterInByIncludeMultiRule) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'innerprod' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { min_level: 2  phase: TRAIN } "
      "  include: { phase: TEST } "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "  include: { min_level: 2  phase: TEST } "
      "  include: { phase: TRAIN } "
      "} ";
  const string& input_proto_train =
      "state: { level: 2  phase: TRAIN } " + input_proto;
  const string& input_proto_test =
      "state: { level: 2  phase: TEST } " + input_proto;
  this->RunFilterNetTest(input_proto_train, input_proto_train);
  this->RunFilterNetTest(input_proto_test, input_proto_test);
}

TEST_F(FilterNetTest, TestFilterInOutByExcludeMultiRule) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'innerprod' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  exclude: { min_level: 2  phase: TRAIN } "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "  exclude: { min_level: 2  phase: TEST } "
      "} ";
  const string& input_proto_train =
      "state: { level: 4  phase: TRAIN } " + input_proto;
  const string& input_proto_test =
      "state: { level: 4  phase: TEST } " + input_proto;
  const string& output_proto_train =
      "state: { level: 4  phase: TRAIN } "
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "  exclude: { min_level: 2  phase: TEST } "
      "} ";
  const string& output_proto_test =
      "state: { level: 4  phase: TEST } "
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'innerprod' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  exclude: { min_level: 2  phase: TRAIN } "
      "} ";
  this->RunFilterNetTest(input_proto_train, output_proto_train);
  this->RunFilterNetTest(input_proto_test, output_proto_test);
}

}  // namespace caffe
