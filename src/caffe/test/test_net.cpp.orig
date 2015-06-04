#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/text_format.h"

#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/net.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

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

  virtual void CopyNetBlobs(const bool copy_diff,
      vector<shared_ptr<Blob<Dtype> > >* blobs_copy) {
    CHECK(net_);
    const vector<shared_ptr<Blob<Dtype> > >& net_blobs = net_->blobs();
    blobs_copy->clear();
    blobs_copy->resize(net_blobs.size());
    const bool kReshape = true;
    for (int i = 0; i < net_blobs.size(); ++i) {
      (*blobs_copy)[i].reset(new Blob<Dtype>());
      (*blobs_copy)[i]->CopyFrom(*net_blobs[i], copy_diff, kReshape);
    }
  }

  virtual void CopyNetParams(const bool copy_diff,
      vector<shared_ptr<Blob<Dtype> > >* params_copy) {
    CHECK(net_);
    const vector<shared_ptr<Blob<Dtype> > >& net_params = net_->params();
    params_copy->clear();
    params_copy->resize(net_params.size());
    const bool kReshape = true;
    for (int i = 0; i < net_params.size(); ++i) {
      (*params_copy)[i].reset(new Blob<Dtype>());
      (*params_copy)[i]->CopyFrom(*net_params[i], copy_diff, kReshape);
    }
  }

  virtual void InitTinyNet(const bool force_backward = false,
                           const bool accuracy_layer = false) {
    string proto =
        "name: 'TinyTestNetwork' "
        "layer { "
        "  name: 'data' "
        "  type: 'DummyData' "
        "  dummy_data_param { "
        "    shape { "
        "      dim: 5 "
        "      dim: 2 "
        "      dim: 3 "
        "      dim: 4 "
        "    } "
        "    data_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "    shape { "
        "      dim: 5 "
        "    } "
        "    data_filler { "
        "      type: 'constant' "
        "      value: 0 "
        "    } "
        "  } "
        "  top: 'data' "
        "  top: 'label' "
        "} "
        "layer { "
        "  name: 'innerproduct' "
        "  type: 'InnerProduct' "
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
        "  param { "
        "    lr_mult: 1 "
        "    decay_mult: 1 "
        "  } "
        "  param { "
        "    lr_mult: 2 "
        "    decay_mult: 0 "
        "  } "
        "  bottom: 'data' "
        "  top: 'innerproduct' "
        "} "
        "layer { "
        "  name: 'loss' "
        "  type: 'SoftmaxWithLoss' "
        "  bottom: 'innerproduct' "
        "  bottom: 'label' "
        "  top: 'top_loss' "
        "} ";
    if (accuracy_layer) {
      proto +=
          "layer { "
          "  name: 'loss' "
          "  type: 'Accuracy' "
          "  bottom: 'innerproduct' "
          "  bottom: 'label' "
          "  top: 'accuracy' "
          "} ";
    }
    if (force_backward) {
      proto += "force_backward: true ";
    }
    InitNetFromProtoString(proto);
  }

  virtual void InitTinyNetEuclidean(const bool force_backward = false) {
    string proto =
        "name: 'TinyTestEuclidLossNetwork' "
        "layer { "
        "  name: 'data' "
        "  type: 'DummyData' "
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
        "layer { "
        "  name: 'innerproduct' "
        "  type: 'InnerProduct' "
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
        "  param { "
        "    lr_mult: 1 "
        "    decay_mult: 1 "
        "  } "
        "  param { "
        "    lr_mult: 2 "
        "    decay_mult: 0 "
        "  } "
        "  bottom: 'data' "
        "  top: 'innerproduct' "
        "} "
        "layer { "
        "  name: 'loss' "
        "  type: 'EuclideanLoss' "
        "  bottom: 'innerproduct' "
        "  bottom: 'label' "
        "} ";
    if (force_backward) {
      proto += "force_backward: true ";
    }
    InitNetFromProtoString(proto);
  }

  virtual void InitTrickyNet(Dtype* loss_weight = NULL) {
    ostringstream loss_weight_stream;
    if (loss_weight) {
      loss_weight_stream << "  loss_weight: " << *loss_weight << " ";
    }
    const string& proto =
        "name: 'TrickyTestNetwork' "
        "layer { "
        "  name: 'data' "
        "  type: 'DummyData' "
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
        "layer { "
        "  name: 'innerproduct' "
        "  type: 'InnerProduct' "
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
        "  param { "
        "    lr_mult: 1 "
        "    decay_mult: 1 "
        "  } "
        "  param { "
        "    lr_mult: 2 "
        "    decay_mult: 0 "
        "  } "
        "  bottom: 'data' "
        "  top: 'transformed_data' "
        "} "
        "layer { "
        "  name: 'innerproduct' "
        "  type: 'InnerProduct' "
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
        "  param { "
        "    lr_mult: 1 "
        "    decay_mult: 1 "
        "  } "
        "  param { "
        "    lr_mult: 2 "
        "    decay_mult: 0 "
        "  } "
        "  bottom: 'label' "
        "  top: 'transformed_label' "
        "} "
        "layer { "
        "  name: 'loss' "
        "  type: 'SoftmaxWithLoss' " +
        loss_weight_stream.str() +
        "  bottom: 'transformed_data' "
        "  bottom: 'transformed_label' "
        "} ";
    InitNetFromProtoString(proto);
  }

  // loss_weight is the loss weight for the 'EuclideanLoss' layer output.
  // midnet_loss_weight is the loss weight for the first 'InnerProduct' layer
  // output.  Should both default to 0.0 if unspecified (i.e., if NULL is
  // passed to this function).
  virtual void InitUnsharedWeightsNet(const Dtype* loss_weight = NULL,
      const Dtype* midnet_loss_weight = NULL,
      const bool force_backward = false, const bool bias_term = false,
      const Dtype blobs_lr_w1 = 1, const Dtype blobs_lr_b1 = 2,
      const Dtype blobs_lr_w2 = 1, const Dtype blobs_lr_b2 = 2) {
    string bias_str = bias_term ? "true ":"false ";
    ostringstream proto;
    proto << "name: 'UnsharedWeightsNetwork' ";
    if (force_backward) {
      proto << "force_backward: true ";
    }
    proto <<
        "layer { "
        "  name: 'data' "
        "  type: 'DummyData' "
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
        "layer { "
        "  name: 'innerproduct1' "
        "  type: 'InnerProduct' "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: " << bias_str <<
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 10 "
        "    } "
        "  } "
        "  param { "
        "    name: 'unsharedweights1' "
        "    lr_mult: " << blobs_lr_w1 <<
        "  } ";
    if (bias_term) {
      proto << "  param { lr_mult: " << blobs_lr_b1 << " } ";
    }
    proto <<
        "  bottom: 'data' "
        "  top: 'innerproduct1' ";
    if (midnet_loss_weight) {
      proto << "  loss_weight: " << *midnet_loss_weight << " ";
    }
    proto <<
        "} "
        "layer { "
        "  name: 'innerproduct2' "
        "  type: 'InnerProduct' "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: " << bias_str <<
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 10 "
        "    } "
        "  } "
        "  param { "
        "    name: 'unsharedweights2' "
        "    lr_mult: " << blobs_lr_w2 <<
        "  } ";
    if (bias_term) {
      proto << "  param { lr_mult: " << blobs_lr_b2 << " } ";
    }
    proto <<
        "  bottom: 'data' "
        "  top: 'innerproduct2' "
        "} "
        "layer { "
        "  name: 'loss' "
        "  type: 'EuclideanLoss' ";
    if (loss_weight) {
      proto << "  loss_weight: " << *loss_weight << " ";
    }
    proto <<
        "  bottom: 'innerproduct1' "
        "  bottom: 'innerproduct2' "
        "} ";
    InitNetFromProtoString(proto.str());
  }

  virtual void InitSharedWeightsNet() {
    const string& proto =
        "name: 'SharedWeightsNetwork' "
        "layer { "
        "  name: 'data' "
        "  type: 'DummyData' "
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
        "layer { "
        "  name: 'innerproduct1' "
        "  type: 'InnerProduct' "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 10 "
        "    } "
        "  } "
        "  param { name: 'sharedweights' } "
        "  bottom: 'data' "
        "  top: 'innerproduct1' "
        "} "
        "layer { "
        "  name: 'innerproduct2' "
        "  type: 'InnerProduct' "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 10 "
        "    } "
        "  } "
        "  param { name: 'sharedweights' } "
        "  bottom: 'data' "
        "  top: 'innerproduct2' "
        "} "
        "layer { "
        "  name: 'loss' "
        "  type: 'EuclideanLoss' "
        "  bottom: 'innerproduct1' "
        "  bottom: 'innerproduct2' "
        "} ";
    InitNetFromProtoString(proto);
  }

  virtual void InitDiffDataUnsharedWeightsNet() {
    const string& proto =
        "name: 'DiffDataUnsharedWeightsNetwork' "
        "layer { "
        "  name: 'data' "
        "  type: 'DummyData' "
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
        "layer { "
        "  name: 'innerproduct1' "
        "  type: 'InnerProduct' "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'constant' "
        "      value: 0.5 "
        "    } "
        "  } "
        "  param { name: 'unsharedweights1' } "
        "  bottom: 'data1' "
        "  top: 'innerproduct1' "
        "} "
        "layer { "
        "  name: 'innerproduct2' "
        "  type: 'InnerProduct' "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'constant' "
        "      value: 0.5 "
        "    } "
        "  } "
        "  param { name: 'unsharedweights2' } "
        "  bottom: 'innerproduct1' "
        "  top: 'innerproduct2' "
        "} "
        "layer { "
        "  name: 'loss' "
        "  type: 'EuclideanLoss' "
        "  bottom: 'data2' "
        "  bottom: 'innerproduct2' "
        "} ";
    InitNetFromProtoString(proto);
  }

  virtual void InitDiffDataSharedWeightsNet() {
    const string& proto =
        "name: 'DiffDataSharedWeightsNetwork' "
        "layer { "
        "  name: 'data' "
        "  type: 'DummyData' "
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
        "layer { "
        "  name: 'innerproduct1' "
        "  type: 'InnerProduct' "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'constant' "
        "      value: 0.5 "
        "    } "
        "  } "
        "  param { name: 'sharedweights' } "
        "  bottom: 'data1' "
        "  top: 'innerproduct1' "
        "} "
        "layer { "
        "  name: 'innerproduct2' "
        "  type: 'InnerProduct' "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'constant' "
        "      value: 0.5 "
        "    } "
        "  } "
        "  param { name: 'sharedweights' } "
        "  bottom: 'innerproduct1' "
        "  top: 'innerproduct2' "
        "} "
        "layer { "
        "  name: 'loss' "
        "  type: 'EuclideanLoss' "
        "  bottom: 'data2' "
        "  bottom: 'innerproduct2' "
        "} ";
    InitNetFromProtoString(proto);
  }

  virtual void InitReshapableNet() {
    const string& proto =
        "name: 'ReshapableNetwork' "
        "input: 'data' "
        "input_dim: 1 "
        "input_dim: 3 "
        "input_dim: 100 "
        "input_dim: 100 "
        "layer { "
        "  name: 'conv1' "
        "  type: 'Convolution' "
        "  bottom: 'data' "
        "  top: 'conv1' "
        "  convolution_param { "
        "    num_output: 5 "
        "    kernel_size: 3 "
        "    stride: 2 "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "    bias_filler { "
        "      type: 'constant' "
        "      value: 0.2 "
        "    } "
        "  } "
        "} "
        "layer { "
        "  name: 'relu1' "
        "  type: 'ReLU' "
        "  bottom: 'conv1' "
        "  top: 'conv1' "
        "} "
        "layer { "
        "  name: 'pool1' "
        "  type: 'Pooling' "
        "  bottom: 'conv1' "
        "  top: 'pool1' "
        "  pooling_param { "
        "    pool: MAX "
        "    kernel_size: 2 "
        "    stride: 2 "
        "  } "
        "} "
        "layer { "
        "  name: 'norm1' "
        "  type: 'LRN' "
        "  bottom: 'pool1' "
        "  top: 'norm1' "
        "  lrn_param { "
        "    local_size: 3 "
        "  } "
        "} "
        "layer { "
        "  name: 'softmax' "
        "  type: 'Softmax' "
        "  bottom: 'norm1' "
        "  top: 'softmax' "
        "} ";
    InitNetFromProtoString(proto);
  }

  virtual void InitSkipPropNet(bool test_skip_true) {
    string proto =
      "name: 'SkipPropTestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'DummyData' "
      "  dummy_data_param { "
      "    shape { "
      "      dim: 5 "
      "      dim: 2 "
      "      dim: 3 "
      "      dim: 4 "
      "    } "
      "    data_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    shape { "
      "      dim: 5 "
      "    } "
      "    data_filler { "
      "      type: 'constant' "
      "      value: 0 "
      "    } "
      "  } "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'silence' "
      "  bottom: 'label' "
      "  type: 'Silence' "
      "} "
      "layer { "
      "  name: 'innerproduct' "
      "  type: 'InnerProduct' "
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
      "  param { "
      "    lr_mult: 1 "
      "    decay_mult: 1 "
      "  } "
      "  param { "
      "    lr_mult: 2 "
      "    decay_mult: 0 "
      "  } "
      "  bottom: 'data' "
      "  top: 'innerproduct' "
      "} "
      "layer { "
      "  name: 'ip_fake_labels' "
      "  type: 'InnerProduct' "
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
      "  bottom: 'data' "
      "  top: 'fake_labels' "
      "} "
      "layer { "
      "  name: 'argmax' "
      "  bottom: 'fake_labels' "
      "  top: 'label_argmax' "
      "  type: 'ArgMax' "
      "} "
      "layer { "
      "  name: 'loss' "
      "  bottom: 'innerproduct' "
      "  bottom: 'label_argmax' ";
    if (test_skip_true)
      proto += "  propagate_down: true "
               "  propagate_down: false ";
    else
      proto += "  propagate_down: true "
               "  propagate_down: true ";
    proto +=
      "  top: 'cross_entropy_loss' "
      "  type: 'SigmoidCrossEntropyLoss' "
      "  loss_weight: 0.1 "
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

TYPED_TEST(NetTest, TestLossWeight) {
  typedef typename TypeParam::Dtype Dtype;
  // First, compute the loss and gradients with no loss_weight specified.
  // In this case, the loss weight for the 'EuclideanLoss' layer should default
  // to 1.
  vector<Blob<Dtype>*> bottom;
  Caffe::set_random_seed(this->seed_);
  const bool kForceBackward = true;
  this->InitUnsharedWeightsNet(NULL, NULL, kForceBackward);
  const Dtype loss = this->net_->ForwardBackward(bottom);
  const bool kCopyDiff = true;
  vector<shared_ptr<Blob<Dtype> > > blob_grads;
  this->CopyNetBlobs(kCopyDiff, &blob_grads);
  vector<shared_ptr<Blob<Dtype> > > param_grads;
  this->CopyNetParams(kCopyDiff, &param_grads);
  // Check that the loss is non-trivial, otherwise the test doesn't prove much.
  const Dtype kMinLossAbsValue = 1e-2;
  ASSERT_GE(fabs(loss), kMinLossAbsValue);
  const Dtype kErrorMargin = 1e-4;
  const int kNumLossWeights = 6;
  Dtype kLossWeights[kNumLossWeights] = {2, 0, 1, -1, -2.5, 3.7};
  for (int i = 0; i < kNumLossWeights; ++i) {
    Caffe::set_random_seed(this->seed_);
    this->InitUnsharedWeightsNet(&kLossWeights[i], NULL, kForceBackward);
    const Dtype weighted_loss = this->net_->ForwardBackward(bottom);
    const Dtype error_margin = kErrorMargin * fabs(kLossWeights[i]);
    EXPECT_NEAR(loss * kLossWeights[i], weighted_loss, error_margin)
        << "loss weight = " << kLossWeights[i];
    const vector<shared_ptr<Blob<Dtype> > >& weighted_blobs =
        this->net_->blobs();
    ASSERT_EQ(blob_grads.size(), weighted_blobs.size());
    for (int j = 0; j < blob_grads.size(); ++j) {
      ASSERT_EQ(blob_grads[j]->count(), weighted_blobs[j]->count());
      for (int k = 0; k < blob_grads[j]->count(); ++k) {
        EXPECT_NEAR(blob_grads[j]->cpu_diff()[k] * kLossWeights[i],
                    weighted_blobs[j]->cpu_diff()[k], error_margin);
      }
    }
    const vector<shared_ptr<Blob<Dtype> > >& weighted_params =
        this->net_->params();
    ASSERT_EQ(param_grads.size(), weighted_params.size());
    for (int j = 0; j < param_grads.size(); ++j) {
      ASSERT_EQ(param_grads[j]->count(), weighted_params[j]->count());
      for (int k = 0; k < param_grads[j]->count(); ++k) {
        EXPECT_NEAR(param_grads[j]->cpu_diff()[k] * kLossWeights[i],
                    weighted_params[j]->cpu_diff()[k], error_margin);
      }
    }
  }
}

TYPED_TEST(NetTest, TestLossWeightMidNet) {
  typedef typename TypeParam::Dtype Dtype;
  vector<Blob<Dtype>*> bottom;
  Caffe::set_random_seed(this->seed_);
  const bool kForceBackward = true;
  Dtype loss_weight = 0;
  Dtype midnet_loss_weight = 1;
  this->InitUnsharedWeightsNet(&loss_weight, &midnet_loss_weight,
                               kForceBackward);
  const Dtype loss = this->net_->ForwardBackward(bottom);
  const bool kCopyDiff = true;
  const bool kReshape = true;
  Blob<Dtype> data_grad;
  data_grad.CopyFrom(*this->net_->blob_by_name("data"), kCopyDiff, kReshape);
  // Check that the loss is non-trivial, otherwise the test doesn't prove much.
  const Dtype kMinLossAbsValue = 1e-2;
  ASSERT_GE(fabs(loss), kMinLossAbsValue);
  const Dtype kErrorMargin = 1e-4;
  const int kNumLossWeights = 6;
  Dtype kLossWeights[kNumLossWeights] = {2, 0, 1, -1, -2.5, 3.7};
  for (int i = 0; i < kNumLossWeights; ++i) {
    Caffe::set_random_seed(this->seed_);
    this->InitUnsharedWeightsNet(&loss_weight, &kLossWeights[i],
                                 kForceBackward);
    const Dtype weighted_loss = this->net_->ForwardBackward(bottom);
    const Dtype error_margin = kErrorMargin * fabs(kLossWeights[i]);
    EXPECT_NEAR(loss * kLossWeights[i], weighted_loss, error_margin)
        << "loss weight = " << kLossWeights[i];
    const shared_ptr<Blob<Dtype> >& weighted_blob =
        this->net_->blob_by_name("data");
    ASSERT_EQ(data_grad.count(), weighted_blob->count());
    for (int j = 0; j < data_grad.count(); ++j) {
      EXPECT_NEAR(data_grad.cpu_diff()[j] * kLossWeights[i],
                  weighted_blob->cpu_diff()[j], error_margin);
    }
  }
}

TYPED_TEST(NetTest, TestComboLossWeight) {
  typedef typename TypeParam::Dtype Dtype;
  vector<Blob<Dtype>*> bottom;
  Dtype loss_weight;
  Dtype midnet_loss_weight;
  const bool kForceBackward = true;
  const Dtype kErrorMargin = 1e-4;

  // Get the loss and gradients with 'EuclideanLoss' weight 1,
  // 'InnerProduct' weight 1.
  loss_weight = 1;
  midnet_loss_weight = 1;
  Caffe::set_random_seed(this->seed_);
  this->InitUnsharedWeightsNet(&loss_weight, &midnet_loss_weight,
                               kForceBackward);
  const Dtype loss = this->net_->ForwardBackward(bottom);
  const bool kCopyDiff = true;
  vector<shared_ptr<Blob<Dtype> > > blob_grads;
  this->CopyNetBlobs(kCopyDiff, &blob_grads);
  vector<shared_ptr<Blob<Dtype> > > param_grads;
  this->CopyNetParams(kCopyDiff, &param_grads);

  loss_weight = 2;
  midnet_loss_weight = 1;
  Caffe::set_random_seed(this->seed_);
  this->InitUnsharedWeightsNet(&loss_weight, &midnet_loss_weight,
                               kForceBackward);
  const Dtype loss_main_2 = this->net_->ForwardBackward(bottom);
  vector<shared_ptr<Blob<Dtype> > > blob_grads_loss_2;
  this->CopyNetBlobs(kCopyDiff, &blob_grads_loss_2);
  vector<shared_ptr<Blob<Dtype> > > param_grads_loss_2;
  this->CopyNetParams(kCopyDiff, &param_grads_loss_2);

  loss_weight = 3;
  midnet_loss_weight = 1;
  Caffe::set_random_seed(this->seed_);
  this->InitUnsharedWeightsNet(&loss_weight, &midnet_loss_weight,
                               kForceBackward);
  const Dtype loss_main_3 = this->net_->ForwardBackward(bottom);
  const vector<shared_ptr<Blob<Dtype> > >& blob_grads_loss_3 =
      this->net_->blobs();
  ASSERT_EQ(blob_grads.size(), blob_grads_loss_3.size());
  ASSERT_EQ(blob_grads_loss_2.size(), blob_grads_loss_3.size());
  for (int j = 0; j < blob_grads.size(); ++j) {
    const string& blob_name = this->net_->blob_names()[j];
    bool grad_should_change = true;
    if (blob_name == "innerproduct1_innerproduct1_0_split_0") {
      grad_should_change = false;
    }
    ASSERT_EQ(blob_grads[j]->count(), blob_grads_loss_3[j]->count());
    ASSERT_EQ(blob_grads_loss_2[j]->count(), blob_grads_loss_3[j]->count());
    for (int k = 0; k < blob_grads[j]->count(); ++k) {
      const Dtype grad_diff_2 = blob_grads_loss_2[j]->cpu_diff()[k] -
                                    blob_grads[j]->cpu_diff()[k];
      const Dtype grad_diff_3 = blob_grads_loss_3[j]->cpu_diff()[k] -
                                    blob_grads[j]->cpu_diff()[k];
      if (grad_should_change) {
        // Test non-triviality.
        const Dtype kMinGradDiffAbsValue = 1e-4;
        EXPECT_GT(fabs(grad_diff_2), kMinGradDiffAbsValue) << blob_name;
        EXPECT_NEAR(2 * grad_diff_2, grad_diff_3, kErrorMargin) << blob_name;
      } else {
        EXPECT_EQ(0, grad_diff_2) << blob_name;
        EXPECT_EQ(0, grad_diff_3) << blob_name;
      }
    }
  }

  loss_weight = 1;
  midnet_loss_weight = 2;
  Caffe::set_random_seed(this->seed_);
  this->InitUnsharedWeightsNet(&loss_weight, &midnet_loss_weight,
                               kForceBackward);
  const Dtype loss_midnet_2 = this->net_->ForwardBackward(bottom);
  this->CopyNetBlobs(kCopyDiff, &blob_grads_loss_2);
  this->CopyNetParams(kCopyDiff, &param_grads_loss_2);

  loss_weight = 1;
  midnet_loss_weight = 3;
  Caffe::set_random_seed(this->seed_);
  this->InitUnsharedWeightsNet(&loss_weight, &midnet_loss_weight,
                               kForceBackward);
  const Dtype loss_midnet_3 = this->net_->ForwardBackward(bottom);
  const vector<shared_ptr<Blob<Dtype> > >& blob_grads_midnet_loss_3 =
      this->net_->blobs();
  ASSERT_EQ(blob_grads.size(), blob_grads_midnet_loss_3.size());
  ASSERT_EQ(blob_grads_loss_2.size(), blob_grads_midnet_loss_3.size());
  const vector<string>& blob_names = this->net_->blob_names();
  for (int j = 0; j < blob_grads.size(); ++j) {
    const string& blob_name = blob_names[j];
    bool grad_should_change = false;
    if (blob_name == "innerproduct1" ||
        blob_name == "innerproduct1_innerproduct1_0_split_0" ||
        blob_name == "data_data_0_split_0" || blob_name == "data") {
      grad_should_change = true;
    }
    ASSERT_EQ(blob_grads[j]->count(), blob_grads_midnet_loss_3[j]->count());
    ASSERT_EQ(blob_grads[j]->count(), blob_grads_loss_2[j]->count());
    for (int k = 0; k < blob_grads[j]->count(); ++k) {
      const Dtype grad_diff_2 = blob_grads_loss_2[j]->cpu_diff()[k] -
                                    blob_grads[j]->cpu_diff()[k];
      const Dtype grad_diff_3 = blob_grads_midnet_loss_3[j]->cpu_diff()[k] -
                                    blob_grads[j]->cpu_diff()[k];
      if (grad_should_change) {
        // Test non-triviality.
        const Dtype kMinGradDiffAbsValue = 1e-4;
        EXPECT_GT(fabs(grad_diff_2), kMinGradDiffAbsValue) << blob_name;
        EXPECT_NEAR(2 * grad_diff_2, grad_diff_3, kErrorMargin) << blob_name;
      } else {
        EXPECT_EQ(0, grad_diff_2) << blob_name;
        EXPECT_EQ(0, grad_diff_3) << blob_name;
      }
    }
  }

  const Dtype kMinLossDiffAbsValue = 1e-4;

  Dtype loss_diff_2 = loss_main_2 - loss;
  // Test non-triviality.
  EXPECT_GT(fabs(loss_diff_2), kMinLossDiffAbsValue);
  Dtype loss_diff_3 = loss_main_3 - loss;
  EXPECT_NEAR(2 * loss_diff_2, loss_diff_3, kErrorMargin);

  loss_diff_2 = loss_midnet_2 - loss;
  // Test non-triviality.
  EXPECT_GT(fabs(loss_diff_2), kMinLossDiffAbsValue);
  loss_diff_3 = loss_midnet_3 - loss;
  EXPECT_NEAR(2 * loss_diff_2, loss_diff_3, kErrorMargin);
}

TYPED_TEST(NetTest, TestBackwardWithAccuracyLayer) {
  typedef typename TypeParam::Dtype Dtype;
  const bool kForceBackward = false;
  const bool kAccuracyLayer = true;
  this->InitTinyNet(kForceBackward, kAccuracyLayer);
  EXPECT_TRUE(this->net_->has_blob("accuracy"));
  vector<Blob<Dtype>*> bottom;
  // Test that we can do Backward even though we have an 'Accuracy' layer.
  this->net_->ForwardBackward(bottom);
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
  // memory.  (The diffs should be accumulated at update time.)
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

TYPED_TEST(NetTest, TestSharedWeightsResume) {
  typedef typename TypeParam::Dtype Dtype;

  // Create a net with weight sharing; Update it once.
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
  // memory.  (The diffs should be accumulated at update time.)
  EXPECT_NE(ip1_weights->cpu_diff(), ip2_weights->cpu_diff());
  this->net_->ForwardBackward(bottom);
  this->net_->Update();
  Blob<Dtype> shared_params;
  const bool kReshape = true;
  const bool kCopyDiff = false;
  shared_params.CopyFrom(*ip1_weights, kCopyDiff, kReshape);
  const int count = ip1_weights->count();

  // Write the net to a NetParameter, as in Solver::Snapshot.
  NetParameter net_param;
  this->net_->ToProto(&net_param);

  // Reinitialize the net and copy parameters from net_param, as in
  // Solver::Restore.
  Caffe::set_random_seed(this->seed_);
  this->InitDiffDataSharedWeightsNet();
  this->net_->CopyTrainedLayersFrom(net_param);
  ip1_weights = this->net_->layers()[1]->blobs()[0].get();
  ip2_weights = this->net_->layers()[2]->blobs()[0].get();
  ASSERT_FALSE(NULL == ip1_weights);
  ASSERT_FALSE(NULL == ip2_weights);
  EXPECT_NE(ip1_weights, ip2_weights);
  // Check that data blobs of shared weights share the same location in memory.
  EXPECT_EQ(ip1_weights->cpu_data(), ip2_weights->cpu_data());
  for (int i = 0; i < count; ++i) {
    EXPECT_FLOAT_EQ(shared_params.cpu_data()[i], ip1_weights->cpu_data()[i]);
  }
  // Check that diff blobs of shared weights are at different locations in
  // memory.  (The diffs should be accumulated at update time.)
  EXPECT_NE(ip1_weights->cpu_diff(), ip2_weights->cpu_diff());
}

TYPED_TEST(NetTest, TestParamPropagateDown) {
  typedef typename TypeParam::Dtype Dtype;
  vector<Blob<Dtype>*> bottom;
  const bool kBiasTerm = true, kForceBackward = false;
  const Dtype* kLossWeight1 = NULL;
  const Dtype* kLossWeight2 = NULL;

  // Run the net with all params learned; check that gradients are non-zero.
  Caffe::set_random_seed(this->seed_);
  Dtype blobs_lr_w1 = 1, blobs_lr_w2 = 1, blobs_lr_b1 = 2, blobs_lr_b2 = 2;
  this->InitUnsharedWeightsNet(kLossWeight1, kLossWeight2, kForceBackward,
      kBiasTerm, blobs_lr_w1, blobs_lr_w2, blobs_lr_b1, blobs_lr_b2);
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
  this->InitUnsharedWeightsNet(kLossWeight1, kLossWeight2, kForceBackward,
      kBiasTerm, blobs_lr_w1, blobs_lr_w2, blobs_lr_b1, blobs_lr_b2);
  this->net_->Forward(bottom);
  this->net_->Backward();
  const vector<shared_ptr<Blob<Dtype> > >& params2 = this->net_->params();
  ASSERT_EQ(num_params, params2.size());
  for (int i = 0; i < num_params; ++i) {
    const Dtype param_asum =
       caffe_cpu_asum(params2[i]->count(), params2[i]->cpu_diff());
    EXPECT_FLOAT_EQ(param_asum, param_asums[i]);
  }

  // Change a subset of the learning rates to zero; check that we see zero
  // gradients for those.
  Caffe::set_random_seed(this->seed_);
  blobs_lr_w1 = 1, blobs_lr_w2 = 0, blobs_lr_b1 = 0, blobs_lr_b2 = 1;
  this->InitUnsharedWeightsNet(kLossWeight1, kLossWeight2, kForceBackward,
      kBiasTerm, blobs_lr_w1, blobs_lr_w2, blobs_lr_b1, blobs_lr_b2);
  this->net_->Forward(bottom);
  this->net_->Backward();
  const vector<shared_ptr<Blob<Dtype> > >& params3 = this->net_->params();
  ASSERT_EQ(num_params, params3.size());
  for (int i = 0; i < num_params; ++i) {
    const Dtype param_asum =
       caffe_cpu_asum(params3[i]->count(), params3[i]->cpu_diff());
    if (i == 1 || i == 2) {
      EXPECT_FLOAT_EQ(0, param_asum);
    } else {
      EXPECT_FLOAT_EQ(param_asum, param_asums[i]);
    }
  }

  // Change the opposite subset of the learning rates to zero.
  Caffe::set_random_seed(this->seed_);
  blobs_lr_w1 = 0, blobs_lr_w2 = 1, blobs_lr_b1 = 1, blobs_lr_b2 = 0;
  this->InitUnsharedWeightsNet(kLossWeight1, kLossWeight2, kForceBackward,
      kBiasTerm, blobs_lr_w1, blobs_lr_w2, blobs_lr_b1, blobs_lr_b2);
  this->net_->Forward(bottom);
  this->net_->Backward();
  const vector<shared_ptr<Blob<Dtype> > >& params4 = this->net_->params();
  ASSERT_EQ(num_params, params4.size());
  for (int i = 0; i < num_params; ++i) {
    const Dtype param_asum =
       caffe_cpu_asum(params4[i]->count(), params4[i]->cpu_diff());
    if (i == 0 || i == 3) {
      EXPECT_FLOAT_EQ(0, param_asum);
    } else {
      EXPECT_FLOAT_EQ(param_asum, param_asums[i]);
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
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'innerprod' "
      "  type: 'InnerProduct' "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunFilterNetTest(input_proto, input_proto);
}

TEST_F(FilterNetTest, TestFilterLeNetTrainTest) {
  const string& input_proto =
      "name: 'LeNet' "
      "layer { "
      "  name: 'mnist' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "  data_param { "
      "    source: 'mnist-train-leveldb' "
      "    batch_size: 64 "
      "  } "
      "  transform_param { "
      "    scale: 0.00390625 "
      "  } "
      "  include: { phase: TRAIN } "
      "} "
      "layer { "
      "  name: 'mnist' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "  data_param { "
      "    source: 'mnist-test-leveldb' "
      "    batch_size: 100 "
      "  } "
      "  transform_param { "
      "    scale: 0.00390625 "
      "  } "
      "  include: { phase: TEST } "
      "} "
      "layer { "
      "  name: 'conv1' "
      "  type: 'Convolution' "
      "  bottom: 'data' "
      "  top: 'conv1' "
      "  param { "
      "    lr_mult: 1 "
      "  } "
      "  param { "
      "    lr_mult: 2 "
      "  } "
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
      "layer { "
      "  name: 'ip1' "
      "  type: 'InnerProduct' "
      "  bottom: 'conv1' "
      "  top: 'ip1' "
      "  param { "
      "    lr_mult: 1 "
      "  } "
      "  param { "
      "    lr_mult: 2 "
      "  } "
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
      "layer { "
      "  name: 'accuracy' "
      "  type: 'Accuracy' "
      "  bottom: 'ip1' "
      "  bottom: 'label' "
      "  top: 'accuracy' "
      "  include: { phase: TEST } "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
      "  bottom: 'ip2' "
      "  bottom: 'label' "
      "  top: 'loss' "
      "} ";
  const string input_proto_train = "state: { phase: TRAIN } " + input_proto;
  const string input_proto_test = "state: { phase: TEST } " + input_proto;
  const string output_proto_train =
      "name: 'LeNet' "
      "layer { "
      "  name: 'mnist' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "  data_param { "
      "    source: 'mnist-train-leveldb' "
      "    batch_size: 64 "
      "  } "
      "  transform_param { "
      "    scale: 0.00390625 "
      "  } "
      "  include: { phase: TRAIN } "
      "} "
      "layer { "
      "  name: 'conv1' "
      "  type: 'Convolution' "
      "  bottom: 'data' "
      "  top: 'conv1' "
      "  param { "
      "    lr_mult: 1 "
      "  } "
      "  param { "
      "    lr_mult: 2 "
      "  } "
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
      "layer { "
      "  name: 'ip1' "
      "  type: 'InnerProduct' "
      "  bottom: 'conv1' "
      "  top: 'ip1' "
      "  param { "
      "    lr_mult: 1 "
      "  } "
      "  param { "
      "    lr_mult: 2 "
      "  } "
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
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
      "  bottom: 'ip2' "
      "  bottom: 'label' "
      "  top: 'loss' "
      "} ";
  const string& output_proto_test =
      "name: 'LeNet' "
      "layer { "
      "  name: 'mnist' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "  data_param { "
      "    source: 'mnist-test-leveldb' "
      "    batch_size: 100 "
      "  } "
      "  transform_param { "
      "    scale: 0.00390625 "
      "  } "
      "  include: { phase: TEST } "
      "} "
      "layer { "
      "  name: 'conv1' "
      "  type: 'Convolution' "
      "  bottom: 'data' "
      "  top: 'conv1' "
      "  param { "
      "    lr_mult: 1 "
      "  } "
      "  param { "
      "    lr_mult: 2 "
      "  } "
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
      "layer { "
      "  name: 'ip1' "
      "  type: 'InnerProduct' "
      "  bottom: 'conv1' "
      "  top: 'ip1' "
      "  param { "
      "    lr_mult: 1 "
      "  } "
      "  param { "
      "    lr_mult: 2 "
      "  } "
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
      "layer { "
      "  name: 'accuracy' "
      "  type: 'Accuracy' "
      "  bottom: 'ip1' "
      "  bottom: 'label' "
      "  top: 'accuracy' "
      "  include: { phase: TEST } "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
      "  bottom: 'ip2' "
      "  bottom: 'label' "
      "  top: 'loss' "
      "} ";
  const string output_proto_train_explicit =
      output_proto_train + " state: { phase: TRAIN } ";
  const string output_proto_test_explicit =
      output_proto_test + " state: { phase: TEST } ";
  this->RunFilterNetTest(input_proto_train, output_proto_train_explicit);
  this->RunFilterNetTest(input_proto_test, output_proto_test_explicit);
}

TEST_F(FilterNetTest, TestFilterOutByStage) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "  include: { stage: 'mystage' } "
      "} "
      "layer { "
      "  name: 'innerprod' "
      "  type: 'InnerProduct' "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  const string& output_proto =
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'innerprod' "
      "  type: 'InnerProduct' "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunFilterNetTest(input_proto, output_proto);
}

TEST_F(FilterNetTest, TestFilterOutByStage2) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'innerprod' "
      "  type: 'InnerProduct' "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { stage: 'mystage' } "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  const string& output_proto =
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunFilterNetTest(input_proto, output_proto);
}

TEST_F(FilterNetTest, TestFilterInByStage) {
  const string& input_proto =
      "state: { stage: 'mystage' } "
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'innerprod' "
      "  type: 'InnerProduct' "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { stage: 'mystage' } "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunFilterNetTest(input_proto, input_proto);
}

TEST_F(FilterNetTest, TestFilterInByStage2) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'innerprod' "
      "  type: 'InnerProduct' "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  exclude: { stage: 'mystage' } "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunFilterNetTest(input_proto, input_proto);
}

TEST_F(FilterNetTest, TestFilterOutByMultipleStage) {
  const string& input_proto =
      "state: { stage: 'mystage' } "
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'innerprod' "
      "  type: 'InnerProduct' "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { stage: 'mystage' stage: 'myotherstage' } "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "  include: { stage: 'mystage' } "
      "} ";
  const string& output_proto =
      "state: { stage: 'mystage' } "
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
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
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'innerprod' "
      "  type: 'InnerProduct' "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { stage: 'myotherstage' } "
      "  include: { stage: 'mystage' } "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
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
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'innerprod' "
      "  type: 'InnerProduct' "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { stage: 'mystage' stage: 'myotherstage' } "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "  include: { stage: 'mystage' } "
      "} ";
  this->RunFilterNetTest(input_proto, input_proto);
}

TEST_F(FilterNetTest, TestFilterInByNotStage) {
  const string& input_proto =
      "state: { stage: 'mystage' } "
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'innerprod' "
      "  type: 'InnerProduct' "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { not_stage: 'myotherstage' } "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "  include: { not_stage: 'myotherstage' } "
      "} ";
  this->RunFilterNetTest(input_proto, input_proto);
}

TEST_F(FilterNetTest, TestFilterOutByNotStage) {
  const string& input_proto =
      "state: { stage: 'mystage' } "
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'innerprod' "
      "  type: 'InnerProduct' "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { not_stage: 'mystage' } "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "  include: { not_stage: 'mystage' } "
      "} ";
  const string& output_proto =
      "state: { stage: 'mystage' } "
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} ";
  this->RunFilterNetTest(input_proto, output_proto);
}

TEST_F(FilterNetTest, TestFilterOutByMinLevel) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'innerprod' "
      "  type: 'InnerProduct' "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { min_level: 3 } "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  const string& output_proto =
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunFilterNetTest(input_proto, output_proto);
}

TEST_F(FilterNetTest, TestFilterOutByMaxLevel) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'innerprod' "
      "  type: 'InnerProduct' "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { max_level: -3 } "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  const string& output_proto =
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunFilterNetTest(input_proto, output_proto);
}

TEST_F(FilterNetTest, TestFilterInByMinLevel) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'innerprod' "
      "  type: 'InnerProduct' "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { min_level: 0 } "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunFilterNetTest(input_proto, input_proto);
}

TEST_F(FilterNetTest, TestFilterInByMinLevel2) {
  const string& input_proto =
      "state: { level: 7 } "
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'innerprod' "
      "  type: 'InnerProduct' "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { min_level: 3 } "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunFilterNetTest(input_proto, input_proto);
}

TEST_F(FilterNetTest, TestFilterInByMaxLevel) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'innerprod' "
      "  type: 'InnerProduct' "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { max_level: 0 } "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunFilterNetTest(input_proto, input_proto);
}

TEST_F(FilterNetTest, TestFilterInByMaxLevel2) {
  const string& input_proto =
      "state: { level: -7 } "
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'innerprod' "
      "  type: 'InnerProduct' "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { max_level: -3 } "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunFilterNetTest(input_proto, input_proto);
}

TEST_F(FilterNetTest, TestFilterInOutByIncludeMultiRule) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'innerprod' "
      "  type: 'InnerProduct' "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { min_level: 2  phase: TRAIN } "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
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
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'innerprod' "
      "  type: 'InnerProduct' "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { min_level: 2  phase: TRAIN } "
      "} ";
  const string& output_proto_test =
      "state: { level: 4  phase: TEST } "
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
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
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'innerprod' "
      "  type: 'InnerProduct' "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  include: { min_level: 2  phase: TRAIN } "
      "  include: { phase: TEST } "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
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
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'innerprod' "
      "  type: 'InnerProduct' "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  exclude: { min_level: 2  phase: TRAIN } "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
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
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "  exclude: { min_level: 2  phase: TEST } "
      "} ";
  const string& output_proto_test =
      "state: { level: 4  phase: TEST } "
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'innerprod' "
      "  type: 'InnerProduct' "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "  exclude: { min_level: 2  phase: TRAIN } "
      "} ";
  this->RunFilterNetTest(input_proto_train, output_proto_train);
  this->RunFilterNetTest(input_proto_test, output_proto_test);
}

TYPED_TEST(NetTest, TestReshape) {
  typedef typename TypeParam::Dtype Dtype;
  // We set up bottom blobs of two different sizes, switch between
  // them, and check that forward and backward both run and the results
  // are the same.
  Caffe::set_random_seed(this->seed_);
  Caffe::set_mode(Caffe::CPU);
  FillerParameter filler_param;
  filler_param.set_std(1);
  GaussianFiller<Dtype> filler(filler_param);
  Blob<Dtype> blob1(4, 3, 9, 11);
  Blob<Dtype> blob2(2, 3, 12, 10);
  filler.Fill(&blob1);
  filler.Fill(&blob2);

  this->InitReshapableNet();
  Blob<Dtype>* input_blob = this->net_->input_blobs()[0];
  Blob<Dtype>* output_blob = this->net_->output_blobs()[0];
  input_blob->Reshape(blob1.num(), blob1.channels(), blob1.height(),
      blob1.width());
  caffe_copy(blob1.count(), blob1.cpu_data(), input_blob->mutable_cpu_data());
  this->net_->ForwardPrefilled();
  // call backward just to make sure it runs
  this->net_->Backward();
  Blob<Dtype> output1(output_blob->num(), output_blob->channels(),
      output_blob->height(), output_blob->width());
  caffe_copy(output1.count(), output_blob->cpu_data(),
      output1.mutable_cpu_data());

  input_blob->Reshape(blob2.num(), blob2.channels(), blob2.height(),
      blob2.width());
  caffe_copy(blob2.count(), blob2.cpu_data(), input_blob->mutable_cpu_data());
  this->net_->ForwardPrefilled();
  this->net_->Backward();
  Blob<Dtype> output2(output_blob->num(), output_blob->channels(),
      output_blob->height(), output_blob->width());
  caffe_copy(output2.count(), output_blob->cpu_data(),
      output2.mutable_cpu_data());

  input_blob->Reshape(blob1.num(), blob1.channels(), blob1.height(),
      blob1.width());
  caffe_copy(blob1.count(), blob1.cpu_data(), input_blob->mutable_cpu_data());
  this->net_->ForwardPrefilled();
  this->net_->Backward();
  for (int i = 0; i < output1.count(); ++i) {
    CHECK_EQ(*(output1.cpu_data() + i), *(output_blob->cpu_data() + i));
  }

  input_blob->Reshape(blob2.num(), blob2.channels(), blob2.height(),
      blob2.width());
  caffe_copy(blob2.count(), blob2.cpu_data(), input_blob->mutable_cpu_data());
  this->net_->ForwardPrefilled();
  this->net_->Backward();
  for (int i = 0; i < output2.count(); ++i) {
    CHECK_EQ(*(output2.cpu_data() + i), *(output_blob->cpu_data() + i));
  }
}

TYPED_TEST(NetTest, TestSkipPropagateDown) {
  // check bottom_need_backward if propagate_down is true
  this->InitSkipPropNet(false);
  vector<bool> vec_layer_need_backward = this->net_->layer_need_backward();
  for (int layer_id = 0; layer_id < this->net_->layers().size(); ++layer_id) {
    string layer_name = this->net_->layer_names()[layer_id];
    if (layer_name == "loss") {
      // access to bottom_need_backward coresponding to label's blob
      bool need_back = this->net_->bottom_need_backward()[layer_id][1];
      // if propagate_down is true, the loss layer will try to
      // backpropagate on labels
      EXPECT_TRUE(need_back) << "bottom_need_backward should be True";
    }
    // layer_need_backward should be True except for data and silence layers
    if (layer_name.find("data") != std::string::npos ||
          layer_name == "silence") {
      EXPECT_FALSE(vec_layer_need_backward[layer_id])
          << "layer_need_backward for " << layer_name << " should be False";
    } else {
      EXPECT_TRUE(vec_layer_need_backward[layer_id])
          << "layer_need_backward for " << layer_name << " should be True";
    }
  }
  // check bottom_need_backward if propagat_down is false
  this->InitSkipPropNet(true);
  vec_layer_need_backward.clear();
  vec_layer_need_backward = this->net_->layer_need_backward();
  for (int layer_id = 0; layer_id < this->net_->layers().size(); ++layer_id) {
    string layer_name = this->net_->layer_names()[layer_id];
    if (layer_name == "loss") {
      // access to bottom_need_backward coresponding to label's blob
      bool need_back = this->net_->bottom_need_backward()[layer_id][1];
      // if propagate_down is false, the loss layer will not try to
      // backpropagate on labels
      EXPECT_FALSE(need_back) << "bottom_need_backward should be False";
    }
    // layer_need_backward should be False except for innerproduct and
    // loss layers
    if (layer_name == "innerproduct" || layer_name == "loss") {
      EXPECT_TRUE(vec_layer_need_backward[layer_id])
          << "layer_need_backward for " << layer_name << " should be True";
    } else {
      EXPECT_FALSE(vec_layer_need_backward[layer_id])
          << "layer_need_backward for " << layer_name << " should be False";
    }
  }
}

}  // namespace caffe
