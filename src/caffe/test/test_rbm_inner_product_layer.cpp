#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "boost/lexical_cast.hpp"

#include "google/protobuf/text_format.h"

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/rbm_inner_product_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

// get all possible binary vectors of a given length
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > > get_combi(int blob_length) {
  const int num_combinations = pow(2, blob_length);
  vector<shared_ptr<Blob<Dtype> > > answer;
  vector<int> blob_size(4, 1);
  blob_size[2] = blob_length;
  for (int i = 0; i < num_combinations; ++i) {
    answer.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(blob_size)));
    for (int j = 0; j < blob_length; ++j) {
      answer[i]->mutable_cpu_data()[j] =
        (i / static_cast<int>(pow(2, j))) % 2 == 0;
    }
  }
  return answer;
}

// calculate the energy of a given hidden / visible pair for a layer's values
template <typename Dtype>
Dtype calculate_energy(const shared_ptr<Blob<Dtype> >& visable_vector,
                       const shared_ptr<Blob<Dtype> >& hidden_vector,
                       RBMInnerProductLayer<Dtype>* layer) {
  const shared_ptr<Blob<Dtype> > weight_matrix  = layer->blobs()[0];
  const shared_ptr<Blob<Dtype> > hidden_bias  = layer->blobs()[1];
  const shared_ptr<Blob<Dtype> > visable_bias = layer->blobs()[2];
  Dtype matrix_sum = caffe_cpu_dot(visable_bias->count(),
                       visable_bias->cpu_data(), visable_vector->cpu_data());

  matrix_sum += caffe_cpu_dot(hidden_bias->count(),
                       hidden_bias->cpu_data(), hidden_vector->cpu_data());

  Blob<Dtype> tmp_blob(vector<int>(1, hidden_vector->count()));

  caffe_cpu_gemv(CblasNoTrans, hidden_vector->count(), visable_vector->count(),
      (Dtype)1., weight_matrix->cpu_data(), visable_vector->cpu_data(),
      (Dtype)0., tmp_blob.mutable_cpu_data());

  matrix_sum += caffe_cpu_dot(hidden_vector->count(),
                       hidden_vector->cpu_data(), tmp_blob.cpu_data());

  // no negative due to double negation
  return exp(matrix_sum);
}

// calculate the overlap of two discrete probability measures
template <typename Dtype>
Dtype calculate_overlap(const vector<shared_ptr<Blob<Dtype> > >& all_visable,
                        const vector<shared_ptr<Blob<Dtype> > >& all_hidden,
                        const vector<Dtype>& actual_probability,
                        RBMInnerProductLayer<Dtype>* layer) {
  // now calculate the estimated probability of everything
  vector<Dtype> est_probability(all_visable.size());

  Dtype total = 0;
  for (int i = 0; i < all_visable.size(); i++) {
    est_probability[i] = 0;
    for (int j = 0; j < all_hidden.size(); j++) {
      est_probability[i] += calculate_energy(all_visable[i], all_hidden[j],
                                             layer);
    }
    total += est_probability[i];
  }
  Dtype overlap = 0;
  for (int i = 0; i < all_visable.size(); i++) {
    est_probability[i] /= total;
    overlap += std::min(actual_probability[i], est_probability[i]);
  }
  return overlap;
}

template <typename TypeParam>
class RBMInnerProductLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  RBMInnerProductLayerTest()
      : blob_bottom_input_(new Blob<Dtype>(2, 3, 4, 5)),
        pre_activation_h1_(new Blob<Dtype>()),
        post_activation_h1_(new Blob<Dtype>()),
        sample_h1_(new Blob<Dtype>()),
        blob_top_error_1_(new Blob<Dtype>()),
        blob_top_error_2_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_type("gaussian");
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_input_);
    blob_bottom_vec_.push_back(blob_bottom_input_);
  }

  virtual ~RBMInnerProductLayerTest() {
    delete blob_bottom_input_;
    delete pre_activation_h1_;
    delete post_activation_h1_;
    delete sample_h1_;
    delete blob_top_error_1_;
    delete blob_top_error_2_;
  }

  virtual void InitLayerFromProtoString(const string& proto) {
    LayerParameter layer_param;
    CHECK(google::protobuf::TextFormat::ParseFromString(proto, &layer_param));
    layer_.reset(new RBMInnerProductLayer<Dtype>(layer_param));
    target_layer_.reset(new RBMInnerProductLayer<Dtype>(layer_param));
  }

  virtual void Fill() {
    // fill both the layer and the target_layer with (different) random values
    FillerParameter filler_param;
    filler_param.set_type("gaussian");
    GaussianFiller<Dtype> filler(filler_param);
    for (int i = 0; i < layer_->blobs().size(); ++i) {
      filler.Fill(layer_->blobs()[i].get());
    }
    for (int i = 0; i < target_layer_->blobs().size(); ++i) {
      filler.Fill(target_layer_->blobs()[i].get());
    }
    filler.Fill(blob_bottom_input_);
  }

  virtual string getLayerText(const string& extra_text = "",
                              bool forward_is_update = true,
                              bool visable_bias_term = true,
                              int num_output = 10,
                              int sample_steps_in_update = 2) {
    string proto =
      "name: 'rbm_inner_product_layer' "
      "type: 'RBMInnerProduct' "
      "rbm_inner_product_param { "
      "  connection_layer_param { "
      "    name: 'connection_inner_product' "
      "    type: 'InnerProduct' "
      "    inner_product_param { "
      "      num_output: ";
    proto += boost::lexical_cast<string>(num_output);
    proto +=
      "      bias_term: true "
      "      weight_filler: { "
      "        type: 'gaussian' "
      "        mean: 0.0 "
      "        std:  0.1 "
      "      } "
      "      bias_filler: { "
      "        type: 'gaussian' "
      "        mean: 0.0 "
      "        std:  0.1 "
      "      } "
      "    } "
      "  } "
      "  hidden_activation_layer_param { "
      "    name: 'hidden_activation' "
      "    type: 'Sigmoid' "
      "  } "
      "  hidden_sampling_layer_param { "
      "    name: 'hidden_sample' "
      "    type: 'BernoulliSample' "
      "  } ";
    proto += "  sample_steps_in_update: ";
    proto += boost::lexical_cast<string>(sample_steps_in_update);
    if (forward_is_update) {
      proto += "  forward_is_update: true ";
    } else {
      proto += "  forward_is_update: false ";
    }
    if (visable_bias_term) {
      proto +=
        "  visible_bias_term: true "
        "  visible_bias_filler { "
        "    type: 'gaussian' "
        "    mean: 0.0 "
        "    std:  0.1 "
        "  } ";
    } else {
      proto += "  visible_bias_term: false ";
    }
    proto += extra_text;
    proto += "} ";
    return proto;
  }

  Blob<Dtype>* const blob_bottom_input_;
  Blob<Dtype>* const pre_activation_h1_;
  Blob<Dtype>* const post_activation_h1_;
  Blob<Dtype>* const sample_h1_;
  Blob<Dtype>* const blob_top_error_1_;
  Blob<Dtype>* const blob_top_error_2_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  shared_ptr<RBMInnerProductLayer<Dtype> > layer_;
  shared_ptr<RBMInnerProductLayer<Dtype> > target_layer_;
};

TYPED_TEST_CASE(RBMInnerProductLayerTest, TestDtypesAndDevices);

TYPED_TEST(RBMInnerProductLayerTest, TestSetUpNoVisibleActivation) {
  string proto = this->getLayerText();
  this->InitLayerFromProtoString(proto);

  this->blob_top_vec_.clear();
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_bottom_input_->num(), 2);
  EXPECT_EQ(this->blob_bottom_input_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_input_->height(), 4);
  EXPECT_EQ(this->blob_bottom_input_->width(), 5);

  // add something to the top
  this->blob_top_vec_.push_back(this->pre_activation_h1_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->pre_activation_h1_->num(), 2);
  EXPECT_EQ(this->pre_activation_h1_->channels(), 10);
  EXPECT_EQ(this->pre_activation_h1_->height(), 1);
  EXPECT_EQ(this->pre_activation_h1_->width(), 1);

  // add more to the top
  this->blob_top_vec_.push_back(this->post_activation_h1_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->post_activation_h1_->num(), 2);
  EXPECT_EQ(this->post_activation_h1_->channels(), 10);
  EXPECT_EQ(this->post_activation_h1_->height(), 1);
  EXPECT_EQ(this->post_activation_h1_->width(), 1);

  // add even more to the top
  this->blob_top_vec_.push_back(this->sample_h1_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->sample_h1_->num(), 2);
  EXPECT_EQ(this->sample_h1_->channels(), 10);
  EXPECT_EQ(this->sample_h1_->height(), 1);
  EXPECT_EQ(this->sample_h1_->width(), 1);

  // test that error tops are resized correctly
  string extra_text =
      "  loss_measure: RECONSTRUCTION "
      "  loss_measure: FREE_ENERGY ";
  proto = this->getLayerText(extra_text);
  this->InitLayerFromProtoString(proto);
  this->blob_top_vec_.push_back(this->blob_top_error_1_);
  this->blob_top_vec_.push_back(this->blob_top_error_2_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_error_1_->num(), 2);
  EXPECT_EQ(this->blob_top_error_1_->channels(), 3);
  EXPECT_EQ(this->blob_top_error_1_->height(), 4);
  EXPECT_EQ(this->blob_top_error_1_->width(), 5);

  EXPECT_EQ(this->blob_top_error_2_->num(), 2);
  EXPECT_EQ(this->blob_top_error_2_->channels(), 1);
  EXPECT_EQ(this->blob_top_error_2_->height(), 1);
  EXPECT_EQ(this->blob_top_error_2_->width(), 1);
}

TYPED_TEST(RBMInnerProductLayerTest, TestSetUpWithVisibleActivation) {
    string extra_text =
      "  visible_activation_layer_param { "
      "    name: 'hidden_activation' "
      "    type: 'Sigmoid' "
      "  } "
      "  visible_sampling_layer_param { "
      "    name: 'hidden_sample' "
      "    type: 'BernoulliSample' "
      "  } ";

  string proto = this->getLayerText(extra_text);
  this->InitLayerFromProtoString(proto);

  this->blob_top_vec_.clear();
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_bottom_input_->num(), 2);
  EXPECT_EQ(this->blob_bottom_input_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_input_->height(), 4);
  EXPECT_EQ(this->blob_bottom_input_->width(), 5);

  // add something to the top
  this->blob_top_vec_.push_back(this->pre_activation_h1_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->pre_activation_h1_->num(), 2);
  EXPECT_EQ(this->pre_activation_h1_->channels(), 10);
  EXPECT_EQ(this->pre_activation_h1_->height(), 1);
  EXPECT_EQ(this->pre_activation_h1_->width(), 1);

  // add more to the top
  this->blob_top_vec_.push_back(this->post_activation_h1_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->post_activation_h1_->num(), 2);
  EXPECT_EQ(this->post_activation_h1_->channels(), 10);
  EXPECT_EQ(this->post_activation_h1_->height(), 1);
  EXPECT_EQ(this->post_activation_h1_->width(), 1);

  // add even more to the top
  this->blob_top_vec_.push_back(this->sample_h1_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->sample_h1_->num(), 2);
  EXPECT_EQ(this->sample_h1_->channels(), 10);
  EXPECT_EQ(this->sample_h1_->height(), 1);
  EXPECT_EQ(this->sample_h1_->width(), 1);

  // test that error tops are resized correctly
  extra_text +=
      "  loss_measure: RECONSTRUCTION "
      "  loss_measure: FREE_ENERGY ";
  proto = this->getLayerText(extra_text);
  this->InitLayerFromProtoString(proto);
  this->blob_top_vec_.push_back(this->blob_top_error_1_);
  this->blob_top_vec_.push_back(this->blob_top_error_2_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_error_1_->num(), 2);
  EXPECT_EQ(this->blob_top_error_1_->channels(), 3);
  EXPECT_EQ(this->blob_top_error_1_->height(), 4);
  EXPECT_EQ(this->blob_top_error_1_->width(), 5);

  EXPECT_EQ(this->blob_top_error_2_->num(), 2);
  EXPECT_EQ(this->blob_top_error_2_->channels(), 1);
  EXPECT_EQ(this->blob_top_error_2_->height(), 1);
  EXPECT_EQ(this->blob_top_error_2_->width(), 1);
}

TYPED_TEST(RBMInnerProductLayerTest, TestForwardNoUpdate) {
  typedef typename TypeParam::Dtype Dtype;
  string extra_text = "  loss_measure: FREE_ENERGY ";
  string proto = this->getLayerText(extra_text, false);

  // run forward with no non error output
  this->InitLayerFromProtoString(proto);
  this->blob_top_vec_.clear();
  this->blob_top_vec_.push_back(this->blob_top_error_1_);

  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  this->Fill();
  this->layer_->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // create an inner product layer that is a copy of this one
  this->blob_top_vec_.clear();
  this->blob_top_vec_.push_back(this->pre_activation_h1_);

  string ip_proto =
      "name: 'inner_product_layer' "
      "type: 'InnerProduct' "
      "inner_product_param { "
      "  num_output: 10 "
      "  bias_term: true "
      "} ";
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(ip_proto, &layer_param));
  InnerProductLayer<Dtype> ip_layer(layer_param);
  Blob<Dtype> ip_top_blob;
  vector<Blob<Dtype>*> ip_top_vec;
  ip_top_vec.push_back(&ip_top_blob);
  ip_layer.SetUp(this->blob_bottom_vec_, ip_top_vec);
  this->Fill();

  // copy the weights so they are the same
  caffe_copy(ip_layer.blobs()[0]->count(), this->layer_->blobs()[0]->cpu_data(),
             ip_layer.blobs()[0]->mutable_cpu_data());
  caffe_copy(ip_layer.blobs()[1]->count(), this->layer_->blobs()[1]->cpu_data(),
             ip_layer.blobs()[1]->mutable_cpu_data());

  // do a forward with both layers
  ip_layer.Forward(this->blob_bottom_vec_, ip_top_vec);

  this->blob_top_vec_.push_back(this->blob_top_error_1_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  this->layer_->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  ASSERT_EQ(ip_top_blob.count(), this->pre_activation_h1_->count());
  // make sure the data is the same
  for (int i = 0; i < ip_top_blob.count(); ++i) {
    EXPECT_FLOAT_EQ(ip_top_blob.cpu_data()[i],
                    this->pre_activation_h1_->cpu_data()[i]);
  }

  // now do a forward and a squash
  SigmoidLayer<Dtype> sigmoid_layer(layer_param);
  this->blob_top_vec_.clear();
  this->blob_top_vec_.push_back(this->pre_activation_h1_);
  this->blob_top_vec_.push_back(this->post_activation_h1_);
  this->blob_top_vec_.push_back(this->blob_top_error_1_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  this->layer_->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(ip_top_blob.count(), this->pre_activation_h1_->count());
  for (int i = 0; i < ip_top_blob.count(); ++i) {
    EXPECT_FLOAT_EQ(ip_top_blob.cpu_data()[i],
                    this->pre_activation_h1_->cpu_data()[i]);
  }
  sigmoid_layer.SetUp(ip_top_vec, ip_top_vec);
  sigmoid_layer.Forward(ip_top_vec, ip_top_vec);
  ASSERT_EQ(ip_top_blob.count(), this->post_activation_h1_->count());
  for (int i = 0; i < ip_top_blob.count(); ++i) {
    EXPECT_FLOAT_EQ(ip_top_blob.cpu_data()[i],
                    this->post_activation_h1_->cpu_data()[i]);
  }

  // check that the sampling really gives us just zeros and ones
  this->blob_top_vec_.clear();
  this->blob_top_vec_.push_back(this->pre_activation_h1_);
  this->blob_top_vec_.push_back(this->post_activation_h1_);
  this->blob_top_vec_.push_back(this->sample_h1_);
  this->blob_top_vec_.push_back(this->blob_top_error_1_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  this->layer_->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < ip_top_blob.count(); ++i) {
    EXPECT_TRUE((this->sample_h1_->cpu_data()[i] == 0 ||
                 this->sample_h1_->cpu_data()[i] == 1));
  }
}

TYPED_TEST(RBMInnerProductLayerTest, TestBackward) {
  typedef typename TypeParam::Dtype Dtype;
  string extra_text =
      "  visible_activation_layer_param { "
      "    name: 'hidden_activation' "
      "    type: 'Sigmoid' "
      "  } "
      "  visible_sampling_layer_param { "
      "    name: 'hidden_sample' "
      "    type: 'BernoulliSample' "
      "  } "
      "  loss_measure: FREE_ENERGY ";
  string proto = this->getLayerText(extra_text, false, false);

  // run forward with no non error output
  this->InitLayerFromProtoString(proto);
  this->blob_top_vec_.clear();
  this->blob_top_vec_.push_back(this->pre_activation_h1_);
  this->blob_top_vec_.push_back(this->blob_top_error_1_);

  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  this->Fill();

  string ip_proto =
      "name: 'inner_product_layer' "
      "type: 'InnerProduct' "
      "inner_product_param { "
      "  num_output: 10 "
      "  bias_term: true "
      "} ";
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(ip_proto, &layer_param));
  InnerProductLayer<Dtype> ip_layer(layer_param);
  Blob<Dtype> ip_top_blob;
  vector<Blob<Dtype>*> ip_top_vec;
  ip_top_vec.push_back(&ip_top_blob);
  ip_layer.SetUp(this->blob_bottom_vec_, ip_top_vec);
  Blob<Dtype> ip_bottom_blob;
  ip_bottom_blob.ReshapeLike(*this->blob_bottom_input_);
  vector<Blob<Dtype>*> ip_bottom_vec;
  ip_bottom_vec.push_back(&ip_bottom_blob);

  // fill the top with random values
  FillerParameter filler_param;
  filler_param.set_type("gaussian");
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(&ip_top_blob);
  caffe_copy(ip_top_blob.count(), ip_top_blob.cpu_data(),
             ip_top_blob.mutable_cpu_diff());
  caffe_copy(ip_top_blob.count(), ip_top_blob.cpu_data(),
             this->pre_activation_h1_->mutable_cpu_diff());

  // copy the weights so they are the same
  caffe_copy(ip_layer.blobs()[0]->count(), this->layer_->blobs()[0]->cpu_data(),
             ip_layer.blobs()[0]->mutable_cpu_data());
  caffe_copy(ip_layer.blobs()[1]->count(), this->layer_->blobs()[1]->cpu_data(),
             ip_layer.blobs()[1]->mutable_cpu_data());

  // do a backward with both layers
  vector<bool> prop_down(1, true);
  ip_layer.Backward(ip_top_vec, prop_down, ip_bottom_vec);
  this->layer_->Backward(this->blob_top_vec_, prop_down,
                         this->blob_bottom_vec_);

  ASSERT_EQ(ip_bottom_blob.count(), this->blob_bottom_input_->count());

  // make sure the diffs are the same
  for (int i = 0; i < ip_bottom_blob.count(); ++i) {
    EXPECT_FLOAT_EQ(ip_bottom_blob.cpu_diff()[i],
                    this->blob_bottom_input_->cpu_diff()[i]);
  }

  // make sure that the data is squashed and sampled
  for (int i = 0; i < ip_top_blob.count(); ++i) {
    EXPECT_TRUE((this->blob_bottom_input_->cpu_data()[i] == 0 ||
                 this->blob_bottom_input_->cpu_data()[i] == 1));
  }
}

// test if samples generated by rbm has correct distribution
TYPED_TEST(RBMInnerProductLayerTest, TestSample) {
  typedef typename TypeParam::Dtype Dtype;
  Caffe::set_random_seed(1702);
  const int num_input(4), num_output(3), batch_size(10);
  const int num_samples(500), sample_length(10);

  string extra_text =
      "  visible_activation_layer_param { "
      "    name: 'hidden_activation' "
      "    type: 'Sigmoid' "
      "  } "
      "  visible_sampling_layer_param { "
      "    name: 'hidden_sample' "
      "    type: 'BernoulliSample' "
      "  } ";
  string proto = this->getLayerText(extra_text, false, true, num_output);
  this->InitLayerFromProtoString(proto);

  vector<int> input_shape(4, 1);
  input_shape[0] = batch_size;
  input_shape[2] = num_input;
  this->blob_bottom_input_->Reshape(input_shape);
  // now calculate the probability of each possible visable vector
  vector<shared_ptr<Blob<Dtype> > > all_visable = get_combi<Dtype>(num_input);
  vector<shared_ptr<Blob<Dtype> > > all_hidden  = get_combi<Dtype>(num_output);
  vector<Dtype> probability(all_visable.size());
  map<string, int> vector_to_index;

  // see how probabable different samples are
  for (int i = 0; i < all_visable.size(); i++) {
    probability[i] = 0;
    ostringstream stuff;
    for (int j = 0; j < all_visable[i]->count(); ++j) {
      stuff << "_" << all_visable[i]->cpu_data()[j];
    }
    vector_to_index[stuff.str()] = i;
  }

  this->blob_top_vec_.clear();
  this->blob_top_vec_.push_back(this->pre_activation_h1_);
  this->blob_top_vec_.push_back(this->post_activation_h1_);
  this->blob_top_vec_.push_back(this->sample_h1_);
  this->target_layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  this->Fill();
  // generate samples from the layer
  for (int i = 0; i < num_samples / batch_size; ++i) {
    caffe_set(this->blob_bottom_vec_[0]->count(), Dtype(0.),
              this->blob_bottom_vec_[0]->mutable_cpu_data());
    for (int j = 0; j < sample_length; ++j) {
      this->target_layer_->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      // copy the samples
      caffe_copy(this->sample_h1_->count(), this->sample_h1_->cpu_data(),
                 this->pre_activation_h1_->mutable_cpu_diff());
      this->target_layer_->Backward(this->blob_top_vec_, vector<bool>(1, true),
                                    this->blob_bottom_vec_);
    }

    // see how often we get different samples
    for (int k = 0; k < batch_size; ++k) {
      ostringstream stuff;
      for (int j = 0; j < this->blob_bottom_input_->count(1); ++j) {
        int idx = k*this->blob_bottom_input_->count(1) + j;
        stuff << "_" << this->blob_bottom_input_->cpu_data()[idx];
      }
      probability[vector_to_index[stuff.str()]] += Dtype(1.0) / num_samples;
    }
  }
  // check if overlap between actual and sampled probability measures is high
  Dtype overlap = calculate_overlap(all_visable, all_hidden, probability,
                                    this->target_layer_.get());
  EXPECT_GE(overlap, .9);
}

}  // namespace caffe
