/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <string>
#include <vector>

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/engine_parser.hpp"
#include "caffe/filler.hpp"

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/tanh_layer.hpp"
#ifdef MKL2017_SUPPORTED
#include "caffe/layers/mkl_layers.hpp"
#endif
#ifdef MKLDNN_SUPPORTED
#include "caffe/layers/mkldnn_layers.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class TestEngineSelection : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  TestEngineSelection() {}
  virtual ~TestEngineSelection() {}

  virtual void InitNetFromProtoString(const string& proto) {
    NetParameter param;
    CHECK(google::protobuf::TextFormat::ParseFromString(proto, &param));
    net_.reset(new Net<Dtype>(param));
  }

  virtual void InitNet(const string& net_engine) {
    string proto =
        "engine: '" + net_engine + "' "
        "layer { "
        "  name: 'data' "
        "  type: 'Input' "
        "  top: 'data' "
        "  input_param { "
        "  shape: { dim: 1 dim: 3 dim: 100 dim: 100 } "
        "  } "
        "} "
        "layer { "
        "  name: 'conv1' "
        "  type: 'Convolution' "
        "  bottom: 'data' "
        "  top: 'conv1' "
        "  convolution_param { "
        "    num_output: 5 "
        "    kernel_size: 3 "
        "    stride: 2 "
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
        "  name: 'ip1' "
        "  type: 'InnerProduct' "
        "  inner_product_param { "
        "    num_output: 1 "
        "  } "
        "  bottom: 'norm1' "
        "  top: 'ip1' "
        "} "
        "layer { "
        "  name: 'relu2' "
        "  type: 'ReLU' "
        "  bottom: 'ip1' "
        "  top: 'ip1' "
        "} "
        " layer {"
        "   bottom: 'ip1'"
        "   top: 'bn1'"
        "   name: 'bn1'"
        "   type: 'BatchNorm'"
        " }"
        " layer {"
        "   bottom: 'pool1'"
        "   bottom: 'norm1'"
        "   top: 'concat1'"
        "   name: 'concat1'"
        "   type: 'Concat'"
        " } "
        " layer { "
        "   bottom: 'concat1' "
        "   bottom: 'concat1' "
        "   top: 'eltw1' "
        "   name: 'eltw1' "
        "   type: 'Eltwise' "
        "}"
        " layer { "
        "   bottom: 'eltw1' "
        "   top: 'split1' "
        "   name: 'split1' "
        "   type: 'Split' "
        "}";

    InitNetFromProtoString(proto);
  }

    shared_ptr<Net<Dtype> > net_;
};

TYPED_TEST_CASE(TestEngineSelection, TestDtypesAndDevices);

TYPED_TEST(TestEngineSelection, TestEngineParser) {
  EngineParser ep1("CAFFE");
  EXPECT_TRUE(ep1.isEngine("CAFFE"));
  EXPECT_FALSE(ep1.isEngine("MKLDNN"));
  EXPECT_FALSE(ep1.isEngine("MKL2017"));
  EXPECT_FALSE(ep1.isEngine("CUDNN"));

#ifdef MKL2017_SUPPORTED
  EngineParser ep2("MKL2017");
  EXPECT_FALSE(ep2.isEngine("CAFFE"));
  EXPECT_FALSE(ep2.isEngine("MKLDNN"));
  EXPECT_TRUE(ep2.isEngine("MKL2017"));
  EXPECT_FALSE(ep2.isEngine("CUDNN"));
#endif

#ifdef MKLDNN_SUPPORTED
  EngineParser ep3("MKLDNN:CPU,FPGA");
  EXPECT_FALSE(ep3.isEngine("CAFFE"));
  EXPECT_TRUE(ep3.isEngine("MKLDNN"));
  EXPECT_FALSE(ep3.isEngine("MKL2017"));
  EXPECT_FALSE(ep3.isEngine("CUDNN"));

  EXPECT_EQ(2, ep3.getNumberOfSubEngines());

  EXPECT_EQ(&ep3.getMKLDNNSubEngine(0), &CpuEngine::Instance().get_engine());
#ifdef FPGA_ENABLED
  EXPECT_EQ(&ep3.getMKLDNNSubEngine(1), &FPGAEngine::Instance().get_engine());
#endif
  EngineParser ep4("MKLDNN:FPGA,CPU,FPGA");
  EXPECT_FALSE(ep4.isEngine("CAFFE"));
  EXPECT_TRUE(ep4.isEngine("MKLDNN"));
  EXPECT_FALSE(ep4.isEngine("MKL2017"));
  EXPECT_FALSE(ep4.isEngine("CUDNN"));

  EXPECT_EQ(3, ep4.getNumberOfSubEngines());

  EXPECT_EQ(&ep4.getMKLDNNSubEngine(1), &CpuEngine::Instance().get_engine());

#ifdef FPGA_ENABLED
  EXPECT_EQ(&ep4.getMKLDNNSubEngine(0), &FPGAEngine::Instance().get_engine());
  EXPECT_EQ(&ep4.getMKLDNNSubEngine(2), &FPGAEngine::Instance().get_engine());
#endif

#endif  // #ifdef MKLDNN_SUPPORTED

#ifdef USE_CUDNN
  EngineParser ep5("CUDNN");
  EXPECT_FALSE(ep5.isEngine("CAFFE"));
  EXPECT_FALSE(ep5.isEngine("MKLDNN"));
  EXPECT_FALSE(ep5.isEngine("MKL2017"));
  EXPECT_TRUE(ep5.isEngine("CUDNN"));
#endif
}

TYPED_TEST(TestEngineSelection, TestEngineParserNetCAFFE) {
  typedef typename TypeParam::Dtype Dtype;

  void* null_ptr = NULL;
  this->InitNet("CAFFE");
  Net<Dtype>* net = this->net_.get();

  // conv1 verification
  Layer<Dtype>* conv1_layer = net->layer_by_name("conv1").get();
  ConvolutionLayer<Dtype>* conv1_caffe =
          dynamic_cast<ConvolutionLayer<Dtype>* >(conv1_layer);
  EXPECT_NE(null_ptr, conv1_caffe);

#ifdef MKL2017_SUPPORTED
  MKLConvolutionLayer<Dtype>* conv1_mkl =
          dynamic_cast<MKLConvolutionLayer<Dtype>* >(conv1_layer);
  EXPECT_EQ(null_ptr, conv1_mkl);
#endif
#ifdef MKLDNN_SUPPORTED
  MKLDNNConvolutionLayer<Dtype>* conv1_mkldnn =
          dynamic_cast<MKLDNNConvolutionLayer<Dtype>* >(conv1_layer);
  EXPECT_EQ(null_ptr, conv1_mkldnn);
#endif
  // relu1 verification
  Layer<Dtype>* relu1_layer = net->layer_by_name("relu1").get();
  ReLULayer<Dtype>* relu1_caffe =
          dynamic_cast<ReLULayer<Dtype>* >(relu1_layer);
  EXPECT_NE(null_ptr, relu1_caffe);

  // relu2 verification
  Layer<Dtype>* relu2_layer = net->layer_by_name("relu2").get();
  ReLULayer<Dtype>* relu2_caffe =
          dynamic_cast<ReLULayer<Dtype>* >(relu2_layer);
  EXPECT_NE(null_ptr, relu2_caffe);

  // pool1 verification
  Layer<Dtype>* pool1_layer = net->layer_by_name("pool1").get();
  PoolingLayer<Dtype>* pool1_caffe =
          dynamic_cast<PoolingLayer<Dtype>* >(pool1_layer);
  EXPECT_NE(null_ptr, pool1_caffe);

  // norm1 verification
  Layer<Dtype>* norm1_layer = net->layer_by_name("norm1").get();
  LRNLayer<Dtype>* norm1_caffe =
          dynamic_cast<LRNLayer<Dtype>* >(norm1_layer);
  EXPECT_NE(null_ptr, norm1_caffe);

  // ip1 verification
  Layer<Dtype>* ip1_layer = net->layer_by_name("ip1").get();
  InnerProductLayer<Dtype>* ip1_caffe =
          dynamic_cast<InnerProductLayer<Dtype>* >(ip1_layer);
  EXPECT_NE(null_ptr, ip1_caffe);

  // bn1 verification
  Layer<Dtype>* bn1_layer = net->layer_by_name("bn1").get();
  BatchNormLayer<Dtype>* bn1_caffe =
          dynamic_cast<BatchNormLayer<Dtype>* >(bn1_layer);
  EXPECT_NE(null_ptr, bn1_caffe);

  // concat1 verification
  Layer<Dtype>* concat1_layer = net->layer_by_name("concat1").get();
  ConcatLayer<Dtype>* concat1_caffe =
          dynamic_cast<ConcatLayer<Dtype>* >(concat1_layer);
  EXPECT_NE(null_ptr, concat1_caffe);

  // eltw1 verification
  Layer<Dtype>* eltw1_layer = net->layer_by_name("eltw1").get();
  EltwiseLayer<Dtype>* eltw1_caffe =
          dynamic_cast<EltwiseLayer<Dtype>* >(eltw1_layer);
  EXPECT_NE(null_ptr, eltw1_caffe);

  // Do all the automatically inserted splits have correct engine?
  const vector<shared_ptr<Layer<Dtype> > >& layers = net->layers();
  for (int i = 0; i < layers.size(); i++) {
    if (layers[i]->layer_param().type() == "Split") {
      string name = layers[i]->layer_param().name();
      Layer<Dtype>* split_layer = net->layer_by_name(name).get();
      SplitLayer<Dtype>* split_caffe =
          dynamic_cast<SplitLayer<Dtype>* >(split_layer);
      EXPECT_NE(null_ptr, split_caffe);
    }
  }
}

#ifdef MKL2017_SUPPORTED
TYPED_TEST(TestEngineSelection, TestEngineParserNetMKL2017) {
  typedef typename TypeParam::Dtype Dtype;

  void* null_ptr = NULL;
  this->InitNet("MKL2017");
  Net<Dtype>* net = this->net_.get();

  // conv1 verification
  Layer<Dtype>* conv1_layer = net->layer_by_name("conv1").get();
  MKLConvolutionLayer<Dtype>* conv1_mkl =
          dynamic_cast<MKLConvolutionLayer<Dtype>* >(conv1_layer);
  EXPECT_NE(null_ptr, conv1_mkl);

  // ConvolutionLayer is a base for MKLConvolutionLayer, so this is not nullptr
  ConvolutionLayer<Dtype>* conv1_caffe =
          dynamic_cast<ConvolutionLayer<Dtype>* >(conv1_layer);
  EXPECT_NE(null_ptr, conv1_caffe);

  // relu1 verification
  Layer<Dtype>* relu1_layer = net->layer_by_name("relu1").get();
  MKLReLULayer<Dtype>* relu1_mkl =
          dynamic_cast<MKLReLULayer<Dtype>* >(relu1_layer);
  EXPECT_NE(null_ptr, relu1_mkl);

  // relu2 verification
  Layer<Dtype>* relu2_layer = net->layer_by_name("relu2").get();
  MKLReLULayer<Dtype>* relu2_mkl =
          dynamic_cast<MKLReLULayer<Dtype>* >(relu2_layer);
  EXPECT_NE(null_ptr, relu2_mkl);

  // pool1 verification
  Layer<Dtype>* pool1_layer = net->layer_by_name("pool1").get();
  MKLPoolingLayer<Dtype>* pool1_mkl =
          dynamic_cast<MKLPoolingLayer<Dtype>* >(pool1_layer);
  EXPECT_NE(null_ptr, pool1_mkl);

  // norm1 verification
  Layer<Dtype>* norm1_layer = net->layer_by_name("norm1").get();
  MKLLRNLayer<Dtype>* norm1_mkl =
          dynamic_cast<MKLLRNLayer<Dtype>* >(norm1_layer);
  EXPECT_NE(null_ptr, norm1_mkl);

  // ip1 verification
  Layer<Dtype>* ip1_layer = net->layer_by_name("ip1").get();
  InnerProductLayer<Dtype>* ip1_caffe =
          dynamic_cast<InnerProductLayer<Dtype>* >(ip1_layer);
  EXPECT_NE(null_ptr, ip1_caffe);

  // bn1 verification
  Layer<Dtype>* bn1_layer = net->layer_by_name("bn1").get();
  MKLBatchNormLayer<Dtype>* bn1_mkl =
          dynamic_cast<MKLBatchNormLayer<Dtype>* >(bn1_layer);
  EXPECT_NE(null_ptr, bn1_mkl);

  // concat1 verification
  Layer<Dtype>* concat1_layer = net->layer_by_name("concat1").get();
  MKLConcatLayer<Dtype>* concat1_mkl =
          dynamic_cast<MKLConcatLayer<Dtype>* >(concat1_layer);
  EXPECT_NE(null_ptr, concat1_mkl);

  // eltw1 verification
  Layer<Dtype>* eltw1_layer = net->layer_by_name("eltw1").get();
  MKLEltwiseLayer<Dtype>* eltw1_mkl =
          dynamic_cast<MKLEltwiseLayer<Dtype>* >(eltw1_layer);
  EXPECT_NE(null_ptr, eltw1_mkl);

  // Do all the automatically inserted splits have correct engine?
  const vector<shared_ptr<Layer<Dtype> > >& layers = net->layers();
  for (int i = 0; i < layers.size(); i++) {
    if (layers[i]->layer_param().type() == "Split") {
      string name = layers[i]->layer_param().name();
      Layer<Dtype>* split_layer = net->layer_by_name(name).get();
      MKLSplitLayer<Dtype>* split_mkl =
          dynamic_cast<MKLSplitLayer<Dtype>* >(split_layer);
      EXPECT_NE(null_ptr, split_mkl);
    }
  }
}
#endif

#ifdef MKLDNN_SUPPORTED
TYPED_TEST(TestEngineSelection, TestEngineParserNetMKLDNN) {
  typedef typename TypeParam::Dtype Dtype;

  void* null_ptr = NULL;
  this->InitNet("MKLDNN:CPU");
  Net<Dtype>* net = this->net_.get();

  // conv1 verification
  Layer<Dtype>* conv1_layer = net->layer_by_name("conv1").get();
  MKLDNNConvolutionLayer<Dtype>* conv1_mkldnn =
          dynamic_cast<MKLDNNConvolutionLayer<Dtype>* >(conv1_layer);
  EXPECT_NE(null_ptr, conv1_mkldnn);

  // MKLDNNConvolutionLayer is derived from ConvolutionLayer, so this is OK
  ConvolutionLayer<Dtype>* conv1_caffe =
          dynamic_cast<ConvolutionLayer<Dtype>* >(conv1_layer);
  EXPECT_NE(null_ptr, conv1_caffe);

  // relu1 verification
  Layer<Dtype>* relu1_layer = net->layer_by_name("relu1").get();
  MKLDNNReLULayer<Dtype>* relu1_mkldnn =
          dynamic_cast<MKLDNNReLULayer<Dtype>* >(relu1_layer);
  EXPECT_EQ(null_ptr, relu1_mkldnn);

  // relu2 verification
  Layer<Dtype>* relu2_layer = net->layer_by_name("relu2").get();
  MKLDNNReLULayer<Dtype>* relu2_mkldnn =
          dynamic_cast<MKLDNNReLULayer<Dtype>* >(relu2_layer);
  EXPECT_NE(null_ptr, relu2_mkldnn);

  // pool1 verification
  Layer<Dtype>* pool1_layer = net->layer_by_name("pool1").get();
  MKLDNNPoolingLayer<Dtype>* pool1_mkldnn =
          dynamic_cast<MKLDNNPoolingLayer<Dtype>* >(pool1_layer);
  EXPECT_NE(null_ptr, pool1_mkldnn);

  // norm1 verification
  Layer<Dtype>* norm1_layer = net->layer_by_name("norm1").get();
  MKLDNNLRNLayer<Dtype>* norm1_mkldnn =
          dynamic_cast<MKLDNNLRNLayer<Dtype>* >(norm1_layer);
  EXPECT_NE(null_ptr, norm1_mkldnn);

    // ip1 verification
  Layer<Dtype>* ip1_layer = net->layer_by_name("ip1").get();
  MKLDNNInnerProductLayer<Dtype>* ip1_mkldnn =
          dynamic_cast<MKLDNNInnerProductLayer<Dtype>* >(ip1_layer);
  EXPECT_NE(null_ptr, ip1_mkldnn);

  // bn1 verification
  Layer<Dtype>* bn1_layer = net->layer_by_name("bn1").get();
  MKLDNNBatchNormLayer<Dtype>* bn1_mkldnn =
          dynamic_cast<MKLDNNBatchNormLayer<Dtype>* >(bn1_layer);
  EXPECT_NE(null_ptr, bn1_mkldnn);

  // concat1 verification
  Layer<Dtype>* concat1_layer = net->layer_by_name("concat1").get();
  MKLDNNConcatLayer<Dtype>* concat1_mkldnn =
          dynamic_cast<MKLDNNConcatLayer<Dtype>* >(concat1_layer);
  EXPECT_NE(null_ptr, concat1_mkldnn);

  // eltw1 verification
  Layer<Dtype>* eltw1_layer = net->layer_by_name("eltw1").get();
  EltwiseLayer<Dtype>* eltw1_caffe =
          dynamic_cast<EltwiseLayer<Dtype>* >(eltw1_layer);
  EXPECT_NE(null_ptr, eltw1_caffe);

  // Do all the automatically inserted splits have correct engine?
  const vector<shared_ptr<Layer<Dtype> > >& layers = net->layers();
  for (int i = 0; i < layers.size(); i++) {
    if (layers[i]->layer_param().type() == "Split") {
      string name = layers[i]->layer_param().name();
      Layer<Dtype>* split_layer = net->layer_by_name(name).get();
      SplitLayer<Dtype>* split_caffe =
          dynamic_cast<SplitLayer<Dtype>* >(split_layer);
      EXPECT_NE(null_ptr, split_caffe);
    }
  }
}

#endif
}  // namespace caffe
