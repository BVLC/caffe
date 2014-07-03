// Copyright 2014 BVLC and contributors.

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class TripletRankingHingeLossLayerTest : public ::testing::Test {
 protected:
  TripletRankingHingeLossLayerTest()
      : blob_bottom_query_(new Blob<Dtype>(10, 11, 1, 1)),
        blob_bottom_similar_sample_(new Blob<Dtype>(10, 11, 1, 1)),
        blob_bottom_dissimilar_sample_(new Blob<Dtype>(10, 11, 1, 1)) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_query_);
    filler.Fill(this->blob_bottom_similar_sample_);
    filler.Fill(this->blob_bottom_dissimilar_sample_);
    blob_bottom_vec_.push_back(blob_bottom_query_);
    blob_bottom_vec_.push_back(blob_bottom_similar_sample_);
    blob_bottom_vec_.push_back(blob_bottom_dissimilar_sample_);
  }
  virtual ~TripletRankingHingeLossLayerTest() {
    delete blob_bottom_query_;
    delete blob_bottom_similar_sample_;
    delete blob_bottom_dissimilar_sample_;
  }
  Blob<Dtype>* const blob_bottom_query_;
  Blob<Dtype>* const blob_bottom_similar_sample_;
  Blob<Dtype>* const blob_bottom_dissimilar_sample_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(TripletRankingHingeLossLayerTest, Dtypes);

TYPED_TEST(TripletRankingHingeLossLayerTest, TestGradientL1CPU) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  TripletRankingHingeLossParameter* triplet_ranking_hinge_loss_param =
      layer_param.mutable_triplet_ranking_hinge_loss_param();
  triplet_ranking_hinge_loss_param->set_norm(
      TripletRankingHingeLossParameter_Norm_L1);
  TripletRankingHingeLossLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  GradientChecker<TypeParam> checker(1e-2, 1e-3, 1701, 1, 0.01);
  checker.CheckGradientSingle(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_), 0, -1, -1);
}

TYPED_TEST(TripletRankingHingeLossLayerTest, TestGradientL1GPU) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  TripletRankingHingeLossParameter* triplet_ranking_hinge_loss_param =
      layer_param.mutable_triplet_ranking_hinge_loss_param();
  triplet_ranking_hinge_loss_param->set_norm(
      TripletRankingHingeLossParameter_Norm_L1);
  TripletRankingHingeLossLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  GradientChecker<TypeParam> checker(1e-2, 1e-3, 1701, 1, 0.01);
  checker.CheckGradientSingle(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_), 0, -1, -1);
}

TYPED_TEST(TripletRankingHingeLossLayerTest, TestGradientL2CPU) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  TripletRankingHingeLossParameter* triplet_ranking_hinge_loss_param =
      layer_param.mutable_triplet_ranking_hinge_loss_param();
  triplet_ranking_hinge_loss_param->set_norm(
      TripletRankingHingeLossParameter_Norm_L2);
  TripletRankingHingeLossLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  GradientChecker<TypeParam> checker(1e-2, 1e-3, 1701, 1, 0.01);
  checker.CheckGradientSingle(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_), 0, -1, -1);
}

TYPED_TEST(TripletRankingHingeLossLayerTest, TestGradientL2GPU) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  TripletRankingHingeLossParameter* triplet_ranking_hinge_loss_param =
      layer_param.mutable_triplet_ranking_hinge_loss_param();
  triplet_ranking_hinge_loss_param->set_norm(
      TripletRankingHingeLossParameter_Norm_L2);
  TripletRankingHingeLossLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  GradientChecker<TypeParam> checker(1e-2, 1e-3, 1701, 1, 0.01);
  checker.CheckGradientSingle(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_), 0, -1, -1);
}

}  // namespace caffe
