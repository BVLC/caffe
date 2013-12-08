// Copyright 2013 Yangqing Jia

// Fillers are random number generators that fills a blob using the specified
// algorithm. The expectation is that they are only going to be used during
// initialization time and will not involve any GPUs.

#ifndef CAFFE_FILLER_HPP
#define CAFFE_FILLER_HPP

//#include <mkl.h>
#include <string>

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Filler {
 public:
  explicit Filler(const FillerParameter& param) : filler_param_(param) {}
  virtual ~Filler() {}
  virtual void Fill(Blob<Dtype>* blob) = 0;
 protected:
  FillerParameter filler_param_;
};  // class Filler


template <typename Dtype>
class ConstantFiller : public Filler<Dtype> {
 public:
  explicit ConstantFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    const int count = blob->count();
    const Dtype value = this->filler_param_.value();
    CHECK(count);
    for (int i = 0; i < count; ++i) {
      data[i] = value;
    }
  };
};

template <typename Dtype>
class UniformFiller : public Filler<Dtype> {
 public:
  explicit UniformFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    caffe_vRngUniform<Dtype>(blob->count(), blob->mutable_cpu_data(),
        Dtype(this->filler_param_.min()),
        Dtype(this->filler_param_.max()));
  }
};

template <typename Dtype>
class GaussianFiller : public Filler<Dtype> {
 public:
  explicit GaussianFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    CHECK(blob->count());
    caffe_vRngGaussian<Dtype>(blob->count(), blob->mutable_cpu_data(),
        Dtype(this->filler_param_.mean()),
        Dtype(this->filler_param_.std()));
  }
};

template <typename Dtype>
class PositiveUnitballFiller : public Filler<Dtype> {
 public:
  explicit PositiveUnitballFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    DCHECK(blob->count());
    caffe_vRngUniform<Dtype>(blob->count(), blob->mutable_cpu_data(), 0, 1);
    // We expect the filler to not be called very frequently, so we will
    // just use a simple implementation
    int dim = blob->count() / blob->num();
    CHECK(dim);
    for (int i = 0; i < blob->num(); ++i) {
      Dtype sum = 0;
      for (int j = 0; j < dim; ++j) {
        sum += data[i * dim + j];
      }
      for (int j = 0; j < dim; ++j) {
        data[i * dim + j] /= sum;
      }
    }
  }
};

// A filler based on the paper [Bengio and Glorot 2010]: Understanding
// the difficulty of training deep feedforward neuralnetworks, but does not
// use the fan_out value.
//
// It fills the incoming matrix by randomly sampling uniform data from
// [-scale, scale] where scale = sqrt(3 / fan_in) where fan_in is the number
// of input nodes. You should make sure the input blob has shape (num, a, b, c)
// where a * b * c = fan_in.
template <typename Dtype>
class XavierFiller : public Filler<Dtype> {
 public:
  explicit XavierFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    int fan_in = blob->count() / blob->num();
    Dtype scale = sqrt(Dtype(3) / fan_in);
    caffe_vRngUniform<Dtype>(blob->count(), blob->mutable_cpu_data(),
        -scale, scale);
  }
};


// A function to get a specific filler from the specification given in
// FillerParameter. Ideally this would be replaced by a factory pattern,
// but we will leave it this way for now.
template <typename Dtype>
Filler<Dtype>* GetFiller(const FillerParameter& param) {
  const std::string& type = param.type();
  if (type == "constant") {
    return new ConstantFiller<Dtype>(param);
  } else if (type == "gaussian") {
    return new GaussianFiller<Dtype>(param);
  } else if (type == "positive_unitball") {
    return new PositiveUnitballFiller<Dtype>(param);
  } else if (type == "uniform") {
    return new UniformFiller<Dtype>(param);
  } else if (type == "xavier") {
    return new XavierFiller<Dtype>(param);
  } else {
    CHECK(false) << "Unknown filler name: " << param.type();
  }
  return (Filler<Dtype>*)(NULL);
}

}  // namespace caffe

#endif  // CAFFE_FILLER_HPP_
