#include <cmath>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/annotated_data_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/multibox_loss_layer.hpp"
#include "caffe/layers/prior_box_layer.hpp"
#include "caffe/layers/reshape_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_conv_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

static bool kBoolChoices[] = {false, true};
static MultiBoxLossParameter_MatchType kMatchTypes[] =
  {MultiBoxLossParameter_MatchType_BIPARTITE,
   MultiBoxLossParameter_MatchType_PER_PREDICTION};

template <typename TypeParam>
class MultiBoxLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MultiBoxLossLayerTest()
      : num_(10),
        num_classes_(3),
        width_(2),
        height_(2),
        num_priors_per_location_(6),
        num_priors_(width_ * height_ * num_priors_per_location_),
        blob_bottom_prior_(new Blob<Dtype>(num_, 2, num_priors_ * 4, 1)),
        blob_bottom_loc_(new Blob<Dtype>(num_, 1, num_priors_ * 4, 1)),
        blob_bottom_conf_(new Blob<Dtype>(
                num_, 1, num_priors_ * num_classes_, 1)),
        blob_bottom_gt_(new Blob<Dtype>(1, 1, 4, 7)),
        blob_top_loss_(new Blob<Dtype>()) {
    blob_bottom_vec_.push_back(blob_bottom_prior_);
    blob_bottom_vec_.push_back(blob_bottom_loc_);
    blob_bottom_vec_.push_back(blob_bottom_conf_);
    blob_bottom_vec_.push_back(blob_bottom_gt_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~MultiBoxLossLayerTest() {
    delete blob_bottom_prior_;
    delete blob_bottom_loc_;
    delete blob_bottom_conf_;
    delete blob_bottom_gt_;
    delete blob_top_loss_;
  }

  // Fill the bottom blobs.
  void Fill(bool share_location) {
    int loc_classes = share_location ? 1 : num_classes_;
    // Create fake network which simulates a simple multi box network.
    vector<Blob<Dtype>*> fake_bottom_vec;
    vector<Blob<Dtype>*> fake_top_vec;
    LayerParameter layer_param;
    // Fake input (image) of size 20 x 20
    Blob<Dtype>* fake_input = new Blob<Dtype>(num_, 3, 20, 20);

    // 1) Fill ground truth.
    string filename;
    GetTempDirname(&filename);
    DataParameter_DB backend = DataParameter_DB_LMDB;
    scoped_ptr<db::DB> db(db::GetDB(backend));
    db->Open(filename, db::NEW);
    scoped_ptr<db::Transaction> txn(db->NewTransaction());
    for (int i = 0; i < num_; ++i) {
      AnnotatedDatum anno_datum;
      // Fill data.
      Datum* datum = anno_datum.mutable_datum();
      datum->set_channels(3);
      datum->set_height(20);
      datum->set_width(20);
      std::string* data = datum->mutable_data();
      for (int j = 0; j < 20*20; ++j) {
        data->push_back(static_cast<uint8_t>(j));
      }
      anno_datum.set_type(AnnotatedDatum_AnnotationType_BBOX);
      if (i == 0 || i == 4) {
        AnnotationGroup* anno_group = anno_datum.add_annotation_group();
        anno_group->set_group_label(1);
        Annotation* anno = anno_group->add_annotation();
        anno->set_instance_id(0);
        NormalizedBBox* bbox = anno->mutable_bbox();
        bbox->set_xmin(0.1);
        bbox->set_ymin(0.1);
        bbox->set_xmax(0.3);
        bbox->set_ymax(0.3);
      }
      if (i == 4) {
        AnnotationGroup* anno_group = anno_datum.add_annotation_group();
        anno_group->set_group_label(2);
        Annotation* anno = anno_group->add_annotation();
        anno->set_instance_id(0);
        NormalizedBBox* bbox = anno->mutable_bbox();
        bbox->set_xmin(0.2);
        bbox->set_ymin(0.2);
        bbox->set_xmax(0.4);
        bbox->set_ymax(0.4);
        anno = anno_group->add_annotation();
        anno->set_instance_id(1);
        bbox = anno->mutable_bbox();
        bbox->set_xmin(0.6);
        bbox->set_ymin(0.6);
        bbox->set_xmax(0.8);
        bbox->set_ymax(0.9);
      }
      stringstream ss;
      ss << i;
      string out;
      CHECK(anno_datum.SerializeToString(&out));
      txn->Put(ss.str(), out);
    }
    txn->Commit();
    db->Close();
    DataParameter* data_param = layer_param.mutable_data_param();
    data_param->set_batch_size(num_);
    data_param->set_source(filename.c_str());
    data_param->set_backend(backend);
    AnnotatedDataLayer<Dtype> anno_data_layer(layer_param);
    fake_top_vec.clear();
    fake_top_vec.push_back(fake_input);
    fake_top_vec.push_back(blob_bottom_gt_);
    anno_data_layer.SetUp(fake_bottom_vec, fake_top_vec);
    anno_data_layer.Forward(fake_bottom_vec, fake_top_vec);

    // 2) Fill prior bboxes.
    // Fake layer
    Blob<Dtype>* fake_blob = new Blob<Dtype>(num_, 10, height_, width_);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(fake_blob);
    filler.Fill(fake_input);
    fake_bottom_vec.clear();
    fake_bottom_vec.push_back(fake_blob);
    fake_bottom_vec.push_back(fake_input);

    PriorBoxParameter* prior_box_param = layer_param.mutable_prior_box_param();
    prior_box_param->set_min_size(5);
    prior_box_param->set_max_size(10);
    prior_box_param->add_aspect_ratio(2.);
    prior_box_param->add_aspect_ratio(3.);
    prior_box_param->set_flip(true);

    PriorBoxLayer<Dtype> prior_layer(layer_param);
    fake_top_vec.clear();
    fake_top_vec.push_back(blob_bottom_prior_);
    prior_layer.SetUp(fake_bottom_vec, fake_top_vec);
    prior_layer.Forward(fake_bottom_vec, fake_top_vec);

    // 3) Fill bbox location predictions.
    ConvolutionParameter* convolution_param =
        layer_param.mutable_convolution_param();
    convolution_param->add_kernel_size(1);
    convolution_param->add_stride(1);
    convolution_param->set_num_output(
        num_priors_per_location_ * loc_classes * 4);
    convolution_param->mutable_weight_filler()->set_type("gaussian");
    convolution_param->mutable_bias_filler()->set_type("constant");
    convolution_param->mutable_bias_filler()->set_value(0.1);
    ConvolutionLayer<Dtype> conv_layer_loc(layer_param);
    fake_bottom_vec.clear();
    fake_bottom_vec.push_back(fake_blob);
    fake_input->Reshape(num_, height_, width_,
                        num_priors_per_location_ * loc_classes * 4);
    fake_top_vec.clear();
    fake_top_vec.push_back(fake_input);
    conv_layer_loc.SetUp(fake_bottom_vec, fake_top_vec);
    conv_layer_loc.Forward(fake_bottom_vec, fake_top_vec);

    BlobShape* blob_shape_loc =
        layer_param.mutable_reshape_param()->mutable_shape();
    blob_shape_loc->add_dim(num_);
    blob_shape_loc->add_dim(1);
    blob_shape_loc->add_dim(num_priors_ * loc_classes * 4);
    blob_shape_loc->add_dim(1);
    ReshapeLayer<Dtype> reshape_layer_loc(layer_param);
    fake_bottom_vec.clear();
    fake_bottom_vec.push_back(fake_input);
    fake_top_vec.clear();
    fake_top_vec.push_back(blob_bottom_loc_);
    reshape_layer_loc.SetUp(fake_bottom_vec, fake_top_vec);
    reshape_layer_loc.Forward(fake_bottom_vec, fake_top_vec);

    // 4) Fill bbox confidence predictions.
    convolution_param->set_num_output(num_priors_per_location_ * num_classes_);
    ConvolutionLayer<Dtype> conv_layer_conf(layer_param);
    fake_bottom_vec.clear();
    fake_bottom_vec.push_back(fake_blob);
    fake_input->Reshape(num_, height_, width_,
                        num_priors_per_location_ * num_classes_);
    fake_top_vec.clear();
    fake_top_vec.push_back(fake_input);
    conv_layer_conf.SetUp(fake_bottom_vec, fake_top_vec);
    conv_layer_conf.Forward(fake_bottom_vec, fake_top_vec);

    layer_param.mutable_reshape_param()->clear_shape();
    BlobShape* blob_shape_conf =
        layer_param.mutable_reshape_param()->mutable_shape();
    blob_shape_conf->add_dim(num_);
    blob_shape_conf->add_dim(1);
    blob_shape_conf->add_dim(num_priors_ * num_classes_);
    blob_shape_conf->add_dim(1);
    ReshapeLayer<Dtype> reshape_layer_conf(layer_param);
    fake_bottom_vec.clear();
    fake_bottom_vec.push_back(fake_input);
    fake_top_vec.clear();
    fake_top_vec.push_back(blob_bottom_conf_);
    reshape_layer_conf.SetUp(fake_bottom_vec, fake_top_vec);
    reshape_layer_conf.Forward(fake_bottom_vec, fake_top_vec);

    delete fake_blob;
    delete fake_input;
  }
  int num_;
  int num_classes_;
  int width_;
  int height_;
  int num_priors_per_location_;
  int num_priors_;
  Blob<Dtype>* const blob_bottom_prior_;
  Blob<Dtype>* const blob_bottom_loc_;
  Blob<Dtype>* const blob_bottom_conf_;
  Blob<Dtype>* const blob_bottom_gt_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MultiBoxLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(MultiBoxLossLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MultiBoxLossParameter* multibox_loss_param =
      layer_param.mutable_multibox_loss_param();
  multibox_loss_param->set_num_classes(3);
  for (int i = 0; i < 2; ++i) {
    bool share_location = kBoolChoices[i];
    this->Fill(share_location);
    for (int j = 0; j < 2; ++j) {
      MultiBoxLossParameter_MatchType match_type = kMatchTypes[j];
      for (int k = 0; k < 2; ++k) {
        bool use_prior = kBoolChoices[k];
        multibox_loss_param->set_share_location(share_location);
        multibox_loss_param->set_match_type(match_type);
        multibox_loss_param->set_use_prior_for_matching(use_prior);
        MultiBoxLossLayer<Dtype> layer(layer_param);
        layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      }
    }
  }
}

TYPED_TEST(MultiBoxLossLayerTest, TestLocGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_propagate_down(false);
  layer_param.add_propagate_down(true);
  MultiBoxLossParameter* multibox_loss_param =
      layer_param.mutable_multibox_loss_param();
  multibox_loss_param->set_num_classes(3);
  for (int i = 0; i < 2; ++i) {
    bool share_location = kBoolChoices[i];
    this->Fill(share_location);
    for (int j = 0; j < 2; ++j) {
      MultiBoxLossParameter_MatchType match_type = kMatchTypes[j];
      for (int k = 0; k < 2; ++k) {
        bool use_prior = kBoolChoices[k];
        multibox_loss_param->set_share_location(share_location);
        multibox_loss_param->set_match_type(match_type);
        multibox_loss_param->set_use_prior_for_matching(use_prior);
        MultiBoxLossLayer<Dtype> layer(layer_param);
        GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
        checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                        this->blob_top_vec_, 1);
      }
    }
  }
}

TYPED_TEST(MultiBoxLossLayerTest, TestConfGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_propagate_down(false);
  layer_param.add_propagate_down(false);
  layer_param.add_propagate_down(true);
  MultiBoxLossParameter* multibox_loss_param =
      layer_param.mutable_multibox_loss_param();
  multibox_loss_param->set_num_classes(3);
  for (int i = 0; i < 2; ++i) {
    bool share_location = kBoolChoices[i];
    this->Fill(share_location);
    for (int j = 0; j < 2; ++j) {
      MultiBoxLossParameter_MatchType match_type = kMatchTypes[j];
      for (int k = 0; k < 2; ++k) {
        bool use_prior = kBoolChoices[k];
        multibox_loss_param->set_share_location(share_location);
        multibox_loss_param->set_match_type(match_type);
        multibox_loss_param->set_use_prior_for_matching(use_prior);
        MultiBoxLossLayer<Dtype> layer(layer_param);
        GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
        checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                        this->blob_top_vec_, 2);
      }
    }
  }
}

}  // namespace caffe
