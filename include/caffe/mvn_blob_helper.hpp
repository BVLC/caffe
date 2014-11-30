#pragma once

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/blob_finder.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template<typename Dtype>
struct MvnBlobHelper {
  typedef std::vector<Blob<Dtype>* > BlobVec;

  MvnBlobHelper():
    layer_param_(),
    mean_index_(-1),
    variance_index_(-1),
    data_index_(-1),
    blob_finder_(),
    initialized_(false) {
  }

  MvnBlobHelper(const LayerParameter& layer_param,
                  const BlobFinder<Dtype>& blob_finder):
    layer_param_(layer_param),
    mean_index_(-1),
    variance_index_(-1),
    data_index_(-1),
    blob_finder_(blob_finder),
    initialized_(false) {
  }

  void LazyInit(const vector<Blob<Dtype>*>& blobs) {
    if (!initialized_) {
      SetUp(blobs);
      initialized_ = true;
    }
  }

  void SetUp(const vector<Blob<Dtype>*>& blobs) {
    CHECK(NumBlobs() == blobs.size()) << "There are " <<
            blobs.size() << " blobs, but " << NumBlobs()
            << " implied by the MVNParameter in layer " << layer_param_.name();
    for (int i = 0; i < blobs.size(); ++i) {
      Blob<Dtype>* blob = blobs[i];
      std::string name = blob_finder_.NameFromPointer(blob);
      if (name == MeanName()) {
        mean_index_ = i;
      } else if (name == VarianceName()) {
        variance_index_ = i;
      } else {
        // The blob not named as the mean or variance blob, by elimination,
        // must be the data blob.
        data_index_ = i;
      }
    }

    if (HasMeanTop()) {
      CHECK_NE(mean_index_, -1) << "Mean blob " << MeanName() <<
              " was specified in layer " << layer_param_.name() << " but " <<
              "was not found in blobs.";
    }
    if (HasVarianceTop()) {
      CHECK_NE(variance_index_, -1) << "Mean blob " << VarianceName() <<
                                  " was specified in layer "
                                << layer_param_.name() << " but "
                                << "was not found in blobs.";
    }

    CHECK_NE(data_index_, -1) << "No data blob found for layer " <<
                                layer_param_.name();
  }

  // Get the number of top blobs this layer will have. It is the number of
  // output blobs if it is an MVNLayer. It is the number of input blobs if
  // it is an InverseMVNLayer.
  int NumBlobs() const {
    int num = 1;
    if ( HasMeanTop() )
      num++;
    if ( HasVarianceTop() )
      num++;
    return num;
  }

  // Get the blob that has the mean of each of the inputs to the MVNLayer.
  Blob<Dtype>* MeanBlob(const BlobVec& blobs) {
    LazyInit(blobs);
    CHECK(HasMeanTop()) << "Layer " << layer_param_.name() << " has no "
                        << "mean blob.";
    return blobs[mean_index_];
  }

  // Get the blob that has the scales by which each input to the MVNLayer is
  // scaled by.
  Blob<Dtype>* VarianceBlob(const BlobVec& blobs) {
    LazyInit(blobs);
    CHECK(HasVarianceTop()) << "Layer " << layer_param_.name() << " has no "
                      << "variance blob.";
    return blobs[variance_index_];
  }

  // Get the output top blob of this layer.
  Blob<Dtype>* DataBlob(const BlobVec& blobs) {
    LazyInit(blobs);
    CHECK(data_index_ < static_cast<int>(blobs.size()))
                  << "Invalid data blob index in MVNLayer "
                                    << this->layer_param_.name();
    return blobs[data_index_];
  }

  // Indicates if the layer exports the mean in the top blobs.
  bool HasMeanTop() const  {
    return layer_param_.mvn_param().has_mean_blob();
  }

  // Name of the mean blob.
  std::string MeanName() const {
    return layer_param_.mvn_param().mean_blob();
  }

  // Name of the variance blob.
  std::string VarianceName() const {
    return layer_param_.mvn_param().variance_blob();
  }

  // Return indication if the layer scales to a variance of one.
  bool HasVarianceTop() const {
    return layer_param_.mvn_param().has_variance_blob();
  }

  // The MVNLayer's parameter.
  LayerParameter layer_param_;
  int mean_index_;
  int variance_index_;
  int data_index_;
  BlobFinder<Dtype> blob_finder_;
  bool initialized_;
};

}  // namespace caffe
