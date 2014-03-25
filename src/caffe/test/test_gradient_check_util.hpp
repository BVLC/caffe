// Copyright 2013 Yangqing Jia

#ifndef CAFFE_TEST_GRADIENT_CHECK_UTIL_H_
#define CAFFE_TEST_GRADIENT_CHECK_UTIL_H_

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/net.hpp"

using std::max;

namespace caffe {

// The gradient checker adds a L2 normalization loss function on top of the
// top blobs, and checks the gradient.
template <typename Dtype>
class GradientChecker {
 public:
  GradientChecker(const Dtype stepsize, const Dtype threshold,
      const unsigned int seed = 1701, const Dtype kink = 0.,
      const Dtype kink_range = -1)
      : stepsize_(stepsize), threshold_(threshold), seed_(seed),
        kink_(kink), kink_range_(kink_range) {}
  // Checks the gradient of a layer, with provided bottom layers and top
  // layers.
  // Note that after the gradient check, we do not guarantee that the data
  // stored in the layer parameters and the blobs are unchanged.
  void CheckGradient(Layer<Dtype>* layer, vector<Blob<Dtype>*>* bottom,
      vector<Blob<Dtype>*>* top, int check_bottom = -1) {
      layer->SetUp(*bottom, top);
      CheckGradientSingle(layer, bottom, top, check_bottom, -1, -1);
  }
  void CheckGradientExhaustive(Layer<Dtype>* layer,
      vector<Blob<Dtype>*>* bottom, vector<Blob<Dtype>*>* top,
      int check_bottom = -1);

  void CheckGradientSingle(Layer<Dtype>* layer, vector<Blob<Dtype>*>* bottom,
      vector<Blob<Dtype>*>* top, int check_bottom, int top_id,
      int top_data_id);

  // Checks the gradient of a network. This network should not have any data
  // layers or loss layers, since the function does not explicitly deal with
  // such cases yet. All input blobs and parameter blobs are going to be
  // checked, layer-by-layer to avoid numerical problems to accumulate.
  void CheckGradientNet(const Net<Dtype>& net,
      const vector<Blob<Dtype>*>& input);

 protected:
  Dtype GetObjAndGradient(vector<Blob<Dtype>*>* top, int top_id = -1,
      int top_data_id = -1);
  Dtype stepsize_;
  Dtype threshold_;
  unsigned int seed_;
  Dtype kink_;
  Dtype kink_range_;
};


// Detailed implementations are as follows.


template <typename Dtype>
void GradientChecker<Dtype>::CheckGradientSingle(Layer<Dtype>* layer,
    vector<Blob<Dtype>*>* bottom, vector<Blob<Dtype>*>* top,
    int check_bottom, int top_id, int top_data_id) {
  // First, figure out what blobs we need to check against.
  vector<Blob<Dtype>*> blobs_to_check;
  for (int i = 0; i < layer->blobs().size(); ++i) {
    blobs_to_check.push_back(layer->blobs()[i].get());
  }
  if (check_bottom < 0) {
    for (int i = 0; i < bottom->size(); ++i) {
      blobs_to_check.push_back((*bottom)[i]);
    }
  } else {
    CHECK(check_bottom < bottom->size());
    blobs_to_check.push_back((*bottom)[check_bottom]);
  }
  // go through the bottom and parameter blobs
  // LOG(ERROR) << "Checking " << blobs_to_check.size() << " blobs.";
  for (int blob_id = 0; blob_id < blobs_to_check.size(); ++blob_id) {
    Blob<Dtype>* current_blob = blobs_to_check[blob_id];
    // LOG(ERROR) << "Blob " << blob_id << ": checking "
    //     << current_blob->count() << " parameters.";
    // go through the values
    for (int feat_id = 0; feat_id < current_blob->count(); ++feat_id) {
      // First, obtain the original data
      Caffe::set_random_seed(seed_);
      // Get any loss from the layer
      Dtype computed_objective = layer->Forward(*bottom, top);
      // Get additional loss from the objective
      computed_objective += GetObjAndGradient(top, top_id, top_data_id);
      layer->Backward(*top, true, bottom);
      Dtype computed_gradient = current_blob->cpu_diff()[feat_id];
      // compute score by adding stepsize
      current_blob->mutable_cpu_data()[feat_id] += stepsize_;
      Caffe::set_random_seed(seed_);
      Dtype positive_objective = layer->Forward(*bottom, top);
      positive_objective += GetObjAndGradient(top, top_id, top_data_id);
      // compute score by subtracting stepsize
      current_blob->mutable_cpu_data()[feat_id] -= stepsize_ * 2;
      Caffe::set_random_seed(seed_);
      Dtype negative_objective = layer->Forward(*bottom, top);
      negative_objective += GetObjAndGradient(top, top_id, top_data_id);
      // Recover stepsize
      current_blob->mutable_cpu_data()[feat_id] += stepsize_;
      Dtype estimated_gradient = (positive_objective - negative_objective) /
          stepsize_ / 2.;
      Dtype feature = current_blob->cpu_data()[feat_id];
      // LOG(ERROR) << "debug: " << current_blob->cpu_data()[feat_id] << " "
      //     << current_blob->cpu_diff()[feat_id];
      if (kink_ - kink_range_ > feature || feature > kink_ + kink_range_) {
        // We check relative accuracy, but for too small values, we threshold
        // the scale factor by 1.
        Dtype scale = max(
            max(fabs(computed_gradient), fabs(estimated_gradient)), 1.);
        EXPECT_NEAR(computed_gradient, estimated_gradient, threshold_ * scale)
          << "debug: (top_id, top_data_id, blob_id, feat_id)="
          << top_id << "," << top_data_id << "," << blob_id << "," << feat_id;
      }
      // LOG(ERROR) << "Feature: " << current_blob->cpu_data()[feat_id];
      // LOG(ERROR) << "computed gradient: " << computed_gradient
      //    << " estimated_gradient: " << estimated_gradient;
    }
  }
}

template <typename Dtype>
void GradientChecker<Dtype>::CheckGradientExhaustive(Layer<Dtype>* layer,
    vector<Blob<Dtype>*>* bottom, vector<Blob<Dtype>*>* top, int check_bottom) {
  layer->SetUp(*bottom, top);
  // LOG(ERROR) << "Exhaustive Mode.";
  for (int i = 0; i < top->size(); ++i) {
    // LOG(ERROR) << "Exhaustive: blob " << i << " size " << top[i]->count();
    for (int j = 0; j < (*top)[i]->count(); ++j) {
      // LOG(ERROR) << "Exhaustive: blob " << i << " data " << j;
      CheckGradientSingle(layer, bottom, top, check_bottom, i, j);
    }
  }
}

template <typename Dtype>
void GradientChecker<Dtype>::CheckGradientNet(
    const Net<Dtype>& net, const vector<Blob<Dtype>*>& input) {
  const vector<shared_ptr<Layer<Dtype> > >& layers = net.layers();
  vector<vector<Blob<Dtype>*> >& bottom_vecs = net.bottom_vecs();
  vector<vector<Blob<Dtype>*> >& top_vecs = net.top_vecs();
  for (int i = 0; i < layers.size(); ++i) {
    net.Forward(input);
    LOG(ERROR) << "Checking gradient for " << layers[i]->layer_param().name();
    CheckGradientExhaustive(*(layers[i].get()), bottom_vecs[i], top_vecs[i]);
  }
}

template <typename Dtype>
Dtype GradientChecker<Dtype>::GetObjAndGradient(vector<Blob<Dtype>*>* top,
    int top_id, int top_data_id) {
  Dtype loss = 0;
  if (top_id < 0) {
    // the loss will be half of the sum of squares of all outputs
    for (int i = 0; i < top->size(); ++i) {
      Blob<Dtype>* top_blob = (*top)[i];
      const Dtype* top_blob_data = top_blob->cpu_data();
      Dtype* top_blob_diff = top_blob->mutable_cpu_diff();
      int count = top_blob->count();
      for (int j = 0; j < count; ++j) {
        loss += top_blob_data[j] * top_blob_data[j];
      }
      // set the diff: simply the data.
      memcpy(top_blob_diff, top_blob_data, sizeof(Dtype) * top_blob->count());
    }
    loss /= 2.;
  } else {
    // the loss will be the top_data_id-th element in the top_id-th blob.
    for (int i = 0; i < top->size(); ++i) {
      Blob<Dtype>* top_blob = (*top)[i];
      Dtype* top_blob_diff = top_blob->mutable_cpu_diff();
      memset(top_blob_diff, 0, sizeof(Dtype) * top_blob->count());
    }
    loss = (*top)[top_id]->cpu_data()[top_data_id];
    (*top)[top_id]->mutable_cpu_diff()[top_data_id] = 1.;
  }
  return loss;
}

}  // namespace caffe

#endif  // CAFFE_TEST_GRADIENT_CHECK_UTIL_H_
