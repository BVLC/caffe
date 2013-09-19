#include <algorithm>
#include <cmath>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "caffeine/test/test_gradient_check_util.hpp"

using std::max;

namespace caffeine {

template <typename Dtype>
void GradientChecker<Dtype>::CheckGradient(Layer<Dtype>& layer,
    vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>& top,
    int check_bottom) {
  layer.SetUp(bottom, &top);
  // First, figure out what blobs we need to check against.
  vector<Blob<Dtype>*> blobs_to_check;
  for (int i = 0; i < layer.params().size(); ++i) {
    blobs_to_check.push_back(&layer.params()[i]);
  }
  if (check_bottom < 0) {
    for (int i = 0; i < bottom.size(); ++i) {
      blobs_to_check.push_back(bottom[i]);
    }
  } else {
    CHECK(check_bottom < bottom.size());
    blobs_to_check.push_back(bottom[check_bottom]);
  }
  // go through the blobs
  //LOG(ERROR) << "Checking " << blobs_to_check.size() << " blobs.";
  for (int blobid = 0; blobid < blobs_to_check.size(); ++blobid) {
    Blob<Dtype>* current_blob = blobs_to_check[blobid];
    //LOG(ERROR) << "Blob " << blobid << ": checking " << current_blob->count()
        << " parameters.";
    // go through the values
    for (int feat_id = 0; feat_id < current_blob->count(); ++feat_id) {
      // First, obtain the original data
      Caffeine::set_random_seed(seed_);
      layer.Forward(bottom, &top);
      Dtype computed_objective = GetObjAndGradient(top);
      // Get any additional loss from the layer
      computed_objective += layer.Backward(top, true, &bottom);
      Dtype computed_gradient = current_blob->cpu_diff()[feat_id];
      // compute score by adding stepsize
      current_blob->mutable_cpu_data()[feat_id] += stepsize_;
      Caffeine::set_random_seed(seed_);
      layer.Forward(bottom, &top);
      Dtype positive_objective = GetObjAndGradient(top);
      positive_objective += layer.Backward(top, true, &bottom);
      // compute score by subtracting stepsize
      current_blob->mutable_cpu_data()[feat_id] -= stepsize_ * 2;
      Caffeine::set_random_seed(seed_);
      layer.Forward(bottom, &top);
      Dtype negative_objective = GetObjAndGradient(top);
      negative_objective += layer.Backward(top, true, &bottom);
      // Recover stepsize
      current_blob->mutable_cpu_data()[feat_id] += stepsize_;
      Dtype estimated_gradient = (positive_objective - negative_objective) /
          stepsize_ / 2.;
      Dtype feature = current_blob->cpu_data()[feat_id];
      //LOG(ERROR) << "debug: " << current_blob->cpu_data()[feat_id] << " "
      //    << current_blob->cpu_diff()[feat_id];
      if (kink_ - kink_range_ > feature || feature > kink_ + kink_range_) {
        // We check relative accuracy, but for too small values, we threshold
        // the scale factor by 1.
        Dtype scale = max(max(fabs(computed_gradient), fabs(estimated_gradient)),
            1.);
        EXPECT_GT(computed_gradient, estimated_gradient - threshold_ * scale);
        EXPECT_LT(computed_gradient, estimated_gradient + threshold_ * scale);
      }
      //LOG(ERROR) << "Feature: " << current_blob->cpu_data()[feat_id];
      //LOG(ERROR) << "computed gradient: " << computed_gradient
      //    << " estimated_gradient: " << estimated_gradient;
    }
  }
}

template <typename Dtype>
Dtype GradientChecker<Dtype>::GetObjAndGradient(vector<Blob<Dtype>*>& top) {
  Dtype loss = 0;
  for (int i = 0; i < top.size(); ++i) {
    Blob<Dtype>* top_blob = top[i];
    const Dtype* top_blob_data = top_blob->cpu_data();
    Dtype* top_blob_diff = top_blob->mutable_cpu_diff();
    int count = top_blob->count();
    for (int j = 0; j < count; ++j) {
      loss += top_blob_data[j] * top_blob_data[j];
    }
    // set the diff: simply the data.
    memcpy(top_blob_diff, top_blob_data, sizeof(Dtype) * count);
  }
  loss /= 2.;
  return loss;
}

INSTANTIATE_CLASS(GradientChecker);

}  // namespace caffeine
