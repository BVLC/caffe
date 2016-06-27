// Fillers are random number generators that fills a blob using the specified
// algorithm. The expectation is that they are only going to be used during
// initialization time and will not involve any GPUs.

#ifndef CAFFE_FILLER_HPP
#define CAFFE_FILLER_HPP

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

/// @brief Fills a Blob with constant or randomly-generated data.
template <typename Dtype>
class Filler {
 public:
  explicit Filler(const FillerParameter& param) : filler_param_(param) {}
  virtual ~Filler() {}
  virtual void Fill(Blob<Dtype>* blob) = 0;
 protected:
  FillerParameter filler_param_;
};  // class Filler


/// @brief Fills a Blob with constant values @f$ x = 0 @f$.
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
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/// @brief Fills a Blob with uniformly distributed values @f$ x\sim U(a, b) @f$.
template <typename Dtype>
class UniformFiller : public Filler<Dtype> {
 public:
  explicit UniformFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    caffe_rng_uniform<Dtype>(blob->count(), Dtype(this->filler_param_.min()),
        Dtype(this->filler_param_.max()), blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/// @brief Fills a Blob with uniformly distributed values @f$ x\sim U(a, b) @f$.
//         RNG is only called once, and its output is stored in
//         UniformStaticFillerdata. Followup calls to the Filler function
//         will bypass RNG.
template <typename Dtype>
class UniformStaticFiller : public Filler<Dtype> {
 public:
  explicit UniformStaticFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {
        uniform_static_filler_flag_ = false;
      }
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    const int blob_count = blob->count();
    CHECK(blob_count);

    if (uniform_static_filler_flag_ == false) {
      uniform_static_filler_data_.resize(blob_count);
      caffe_rng_uniform(blob_count, Dtype(this->filler_param_.min()),
          Dtype(this->filler_param_.max()),
          &uniform_static_filler_data_.front());
      uniform_static_filler_flag_ = true;
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
        << "Sparsity not supported by this Filler.";
    caffe_copy(blob_count, &uniform_static_filler_data_.front(), data);
  }

 private:
  std::vector<Dtype> uniform_static_filler_data_;
  bool uniform_static_filler_flag_;
};


/// @brief Fills a Blob with Gaussian-distributed values @f$ x = a @f$.
template <typename Dtype>
class GaussianFiller : public Filler<Dtype> {
 public:
  explicit GaussianFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    CHECK(blob->count());
    caffe_rng_gaussian<Dtype>(blob->count(), Dtype(this->filler_param_.mean()),
        Dtype(this->filler_param_.std()), blob->mutable_cpu_data());
    int sparse = this->filler_param_.sparse();
    CHECK_GE(sparse, -1);
    if (sparse >= 0) {
      // Sparse initialization is implemented for "weight" blobs; i.e. matrices.
      // These have num == channels == 1; width is number of inputs; height is
      // number of outputs.  The 'sparse' variable specifies the mean number
      // of non-zero input weights for a given output.
      CHECK_GE(blob->num_axes(), 1);
      const int num_outputs = blob->shape(0);
      Dtype non_zero_probability = Dtype(sparse) / Dtype(num_outputs);
      rand_vec_.reset(new SyncedMemory(blob->count() * sizeof(int)));
      int* mask = reinterpret_cast<int*>(rand_vec_->mutable_cpu_data());
      caffe_rng_bernoulli(blob->count(), non_zero_probability, mask);
      for (int i = 0; i < blob->count(); ++i) {
        data[i] *= mask[i];
      }
    }
  }

 protected:
  shared_ptr<SyncedMemory> rand_vec_;
};


/// @brief Fills a Blob with Gaussian-distributed values @f$ x = a @f$.
//         RNG is only called once, and its output is stored in
//         GaussianStaticFillerdata. Followup calls to the Filler function
//         will bypass RNG.
template <typename Dtype>
class GaussianStaticFiller : public Filler<Dtype> {
 public:
  explicit GaussianStaticFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {
    gaussian_static_filler_flag_ = false;
  }
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    const int blob_count = blob->count();
    CHECK(blob_count);

    if (gaussian_static_filler_flag_ == false) {
      gaussian_static_filler_data_.resize(blob_count);
      caffe_rng_gaussian(blob_count, Dtype(this->filler_param_.mean()),
        Dtype(this->filler_param_.std()),
        &gaussian_static_filler_data_.front());
      gaussian_static_filler_flag_ = true;
    }
    caffe_copy(blob_count, &gaussian_static_filler_data_.front(), data);
    int sparse = this->filler_param_.sparse();
    CHECK_GE(sparse, -1);
    if (sparse >= 0) {
      // Sparse initialization is implemented for "weight" blobs; i.e. matrices.
      // These have num == channels == 1; width is number of inputs; height is
      // number of outputs.  The 'sparse' variable specifies the mean number
      // of non-zero input weights for a given output.
      CHECK_GE(blob->num_axes(), 1);
      const int num_outputs = blob->shape(0);
      Dtype non_zero_probability = Dtype(sparse) / Dtype(num_outputs);
      rand_vec_.resize(blob_count * sizeof(int));
      int* mask = reinterpret_cast<int*>(&rand_vec_.front());
      caffe_rng_bernoulli(blob_count, non_zero_probability, mask);
      for (int i = 0; i < blob_count; ++i) {
        data[i] *= mask[i];
      }
    }
  }

 protected:
  std::vector<int> rand_vec_;

 private:
  std::vector<Dtype> gaussian_static_filler_data_;
  bool gaussian_static_filler_flag_;
};

/** @brief Fills a Blob with values @f$ x \in [0, 1] @f$
 *         such that @f$ \forall i \sum_j x_{ij} = 1 @f$.
 */
template <typename Dtype>
class PositiveUnitballFiller : public Filler<Dtype> {
 public:
  explicit PositiveUnitballFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    DCHECK(blob->count());
    caffe_rng_uniform<Dtype>(blob->count(), 0, 1, blob->mutable_cpu_data());
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
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/** @brief Fills a Blob with values @f$ x \in [0, 1] @f$
 *         such that @f$ \forall i \sum_j x_{ij} = 1 @f$.
 *         RNG is only called once, and its output is stored in
 *         PositiveUnitballStaticFillerdata. Followup calls to the Filler function
 *         will bypass RNG.
 */
template <typename Dtype>
class PositiveUnitballStaticFiller : public Filler<Dtype> {
 public:
  explicit PositiveUnitballStaticFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {
    positive_unitball_static_filler_flag_ = false;
  }
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    const int blob_count = blob->count();
    CHECK(blob_count);
    if (positive_unitball_static_filler_flag_ == false) {
      positive_unitball_static_filler_data_.resize(blob_count);
      caffe_rng_uniform(blob_count, Dtype(0.), Dtype(1.),
          &positive_unitball_static_filler_data_.front());

      int dim = blob_count / blob->num();
      CHECK(dim);
      for (int i = 0; i < blob->num(); ++i) {
        Dtype sum = 0;
        for (int j = 0; j < dim; ++j) {
          sum += positive_unitball_static_filler_data_[i * dim + j];
        }
        for (int j = 0; j < dim; ++j) {
          positive_unitball_static_filler_data_[i * dim + j] /= sum;
        }
      }
      positive_unitball_static_filler_flag_ = true;
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
    caffe_copy(blob_count,
        &positive_unitball_static_filler_data_.front(), data);
  }


 private:
  std::vector<Dtype> positive_unitball_static_filler_data_;
  bool positive_unitball_static_filler_flag_;
};


/**
 * @brief Fills a Blob with values @f$ x \sim U(-a, +a) @f$ where @f$ a @f$ is
 *        set inversely proportional to number of incoming nodes, outgoing
 *        nodes, or their average.
 *
 * A Filler based on the paper [Bengio and Glorot 2010]: Understanding
 * the difficulty of training deep feedforward neuralnetworks.
 *
 * It fills the incoming matrix by randomly sampling uniform data from [-scale,
 * scale] where scale = sqrt(3 / n) where n is the fan_in, fan_out, or their
 * average, depending on the variance_norm option. You should make sure the
 * input blob has shape (num, a, b, c) where a * b * c = fan_in and num * b * c
 * = fan_out. Note that this is currently not the case for inner product layers.
 *
 * TODO(dox): make notation in above comment consistent with rest & use LaTeX.
 */
template <typename Dtype>
class XavierFiller : public Filler<Dtype> {
 public:
  explicit XavierFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    int fan_in = blob->count() / blob->num();
    int fan_out = blob->count() / blob->channels();
    Dtype n = fan_in;  // default to fan_in
    if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_AVERAGE) {
      n = (fan_in + fan_out) / Dtype(2);
    } else if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_FAN_OUT) {
      n = fan_out;
    }
    Dtype scale = sqrt(Dtype(3) / n);
    caffe_rng_uniform<Dtype>(blob->count(), -scale, scale,
        blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};


/**
 * @brief Fills a Blob with values @f$ x \sim U(-a, +a) @f$ where @f$ a @f$ is
 *        set inversely proportional to number of incoming nodes, outgoing
 *        nodes, or their average.
 *
 * A Filler based on the paper [Bengio and Glorot 2010]: Understanding
 * the difficulty of training deep feedforward neuralnetworks.
 *
 * It fills the incoming matrix by randomly sampling uniform data from [-scale,
 * scale] where scale = sqrt(3 / n) where n is the fan_in, fan_out, or their
 * average, depending on the variance_norm option. You should make sure the
 * input blob has shape (num, a, b, c) where a * b * c = fan_in and num * b * c
 * = fan_out. Note that this is currently not the case for inner product layers.
 *
 * RNG is only called once, and its output is stored in
 * XavierStaticFillerdata. Followup calls to the Filler function
 * will bypass RNG.
 *
 * TODO(dox): make notation in above comment consistent with rest & use LaTeX.
 */
template <typename Dtype>
class XavierStaticFiller : public Filler<Dtype> {
 public:
  explicit XavierStaticFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {
    xavier_static_filler_flag_ = false;
  }
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    const int blob_count = blob->count();
    CHECK(blob_count);

    if (xavier_static_filler_flag_ == false) {
      xavier_static_filler_data_.resize(blob_count);
      int fan_in = blob_count / blob->num();
      int fan_out = blob_count / blob->channels();
      Dtype n = fan_in;  // default to fan_in
      if (this->filler_param_.variance_norm() ==
         FillerParameter_VarianceNorm_AVERAGE) {
        n = (fan_in + fan_out) / Dtype(2);
      } else if (this->filler_param_.variance_norm() ==
          FillerParameter_VarianceNorm_FAN_OUT) {
        n = fan_out;
      }
      Dtype scale = sqrt(Dtype(3) / n);
      caffe_rng_uniform(blob_count, Dtype(-scale), Dtype(scale),
          &xavier_static_filler_data_.front());
      xavier_static_filler_flag_ = true;
    }
    caffe_copy(blob_count, &xavier_static_filler_data_.front(), data);
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }

 private:
  std::vector<Dtype> xavier_static_filler_data_;
  bool xavier_static_filler_flag_;
};


/**
 * @brief Fills a Blob with values @f$ x \sim N(0, \sigma^2) @f$ where
 *        @f$ \sigma^2 @f$ is set inversely proportional to number of incoming
 *        nodes, outgoing nodes, or their average.
 *
 * A Filler based on the paper [He, Zhang, Ren and Sun 2015]: Specifically
 * accounts for ReLU nonlinearities.
 *
 * Aside: for another perspective on the scaling factor, see the derivation of
 * [Saxe, McClelland, and Ganguli 2013 (v3)].
 *
 * It fills the incoming matrix by randomly sampling Gaussian data with std =
 * sqrt(2 / n) where n is the fan_in, fan_out, or their average, depending on
 * the variance_norm option. You should make sure the input blob has shape (num,
 * a, b, c) where a * b * c = fan_in and num * b * c = fan_out. Note that this
 * is currently not the case for inner product layers.
 */
template <typename Dtype>
class MSRAFiller : public Filler<Dtype> {
 public:
  explicit MSRAFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    int fan_in = blob->count() / blob->num();
    int fan_out = blob->count() / blob->channels();
    Dtype n = fan_in;  // default to fan_in
    if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_AVERAGE) {
      n = (fan_in + fan_out) / Dtype(2);
    } else if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_FAN_OUT) {
      n = fan_out;
    }
    Dtype std = sqrt(Dtype(2) / n);
    caffe_rng_gaussian<Dtype>(blob->count(), Dtype(0), std,
        blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};


/**
 * @brief Fills a Blob with values @f$ x \sim N(0, \sigma^2) @f$ where
 *        @f$ \sigma^2 @f$ is set inversely proportional to number of incoming
 *        nodes, outgoing nodes, or their average.
 *
 * A Filler based on the paper [He, Zhang, Ren and Sun 2015]: Specifically
 * accounts for ReLU nonlinearities.
 *
 * Aside: for another perspective on the scaling factor, see the derivation of
 * [Saxe, McClelland, and Ganguli 2013 (v3)].
 *
 * It fills the incoming matrix by randomly sampling Gaussian data with std =
 * sqrt(2 / n) where n is the fan_in, fan_out, or their average, depending on
 * the variance_norm option. You should make sure the input blob has shape (num,
 * a, b, c) where a * b * c = fan_in and num * b * c = fan_out. Note that this
 * is currently not the case for inner product layers.
 *
 * RNG is only called once, and its output is stored in
 * MSRAStaticFillerdata. Followup calls to the Filler function
 * will bypass RNG.
 */
template <typename Dtype>
class MSRAStaticFiller : public Filler<Dtype> {
 public:
  explicit MSRAStaticFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {
    msra_static_filler_flag_ = false;
  }
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    const int blob_count = blob->count();
    CHECK(blob_count);

    if (msra_static_filler_flag_ == false) {
      msra_static_filler_data_.resize(blob_count);
      int fan_in = blob_count / blob->num();
      int fan_out = blob_count / blob->channels();
      Dtype n = fan_in;  // default to fan_in
      if (this->filler_param_.variance_norm() ==
          FillerParameter_VarianceNorm_AVERAGE) {
        n = (fan_in + fan_out) / Dtype(2);
      } else if (this->filler_param_.variance_norm() ==
          FillerParameter_VarianceNorm_FAN_OUT) {
        n = fan_out;
      }
      Dtype std = sqrt(Dtype(2) / n);
      caffe_rng_gaussian(blob_count, Dtype(0), std,
          &msra_static_filler_data_.front());
      msra_static_filler_flag_ = true;
    }
    caffe_copy(blob_count, &msra_static_filler_data_.front(), data);
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }

 private:
  std::vector<Dtype> msra_static_filler_data_;
  bool msra_static_filler_flag_;
};

/*!
@brief Fills a Blob with coefficients for bilinear interpolation.

A common use case is with the DeconvolutionLayer acting as upsampling.
You can upsample a feature map with shape of (B, C, H, W) by any integer factor
using the following proto.
\code
layer {
  name: "upsample", type: "Deconvolution"
  bottom: "{{bottom_name}}" top: "{{top_name}}"
  convolution_param {
    kernel_size: {{2 * factor - factor % 2}} stride: {{factor}}
    num_output: {{C}} group: {{C}}
    pad: {{ceil((factor - 1) / 2.)}}
    weight_filler: { type: "bilinear" } bias_term: false
  }
  param { lr_mult: 0 decay_mult: 0 }
}
\endcode
Please use this by replacing `{{}}` with your values. By specifying
`num_output: {{C}} group: {{C}}`, it behaves as
channel-wise convolution. The filter shape of this deconvolution layer will be
(C, 1, K, K) where K is `kernel_size`, and this filler will set a (K, K)
interpolation kernel for every channel of the filter identically. The resulting
shape of the top feature map will be (B, C, factor * H, factor * W).
Note that the learning rate and the
weight decay are set to 0 in order to keep coefficient values of bilinear
interpolation unchanged during training. If you apply this to an image, this
operation is equivalent to the following call in Python with Scikit.Image.
\code{.py}
out = skimage.transform.rescale(img, factor, mode='constant', cval=0)
\endcode
 */
template <typename Dtype>
class BilinearFiller : public Filler<Dtype> {
 public:
  explicit BilinearFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK_EQ(blob->num_axes(), 4) << "Blob must be 4 dim.";
    CHECK_EQ(blob->width(), blob->height()) << "Filter must be square";
    Dtype* data = blob->mutable_cpu_data();
    int f = ceil(blob->width() / 2.);
    float c = (2 * f - 1 - f % 2) / (2. * f);
    for (int i = 0; i < blob->count(); ++i) {
      float x = i % blob->width();
      float y = (i / blob->width()) % blob->height();
      data[i] = (1 - fabs(x / f - c)) * (1 - fabs(y / f - c));
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/*!
@brief Fills a Blob with coefficients for bilinear interpolation.

A common use case is with the DeconvolutionLayer acting as upsampling.
You can upsample a feature map with shape of (B, C, H, W) by any integer factor
using the following proto.
\code
layer {
  name: "upsample", type: "Deconvolution"
  bottom: "{{bottom_name}}" top: "{{top_name}}"
  convolution_param {
    kernel_size: {{2 * factor - factor % 2}} stride: {{factor}}
    num_output: {{C}} group: {{C}}
    pad: {{ceil((factor - 1) / 2.)}}
    weight_filler: { type: "bilinear" } bias_term: false
  }
  param { lr_mult: 0 decay_mult: 0 }
}
\endcode
Please use this by replacing `{{}}` with your values. By specifying
`num_output: {{C}} group: {{C}}`, it behaves as
channel-wise convolution. The filter shape of this deconvolution layer will be
(C, 1, K, K) where K is `kernel_size`, and this filler will set a (K, K)
interpolation kernel for every channel of the filter identically. The resulting
shape of the top feature map will be (B, C, factor * H, factor * W).
Note that the learning rate and the
weight decay are set to 0 in order to keep coefficient values of bilinear
interpolation unchanged during training. If you apply this to an image, this
operation is equivalent to the following call in Python with Scikit.Image.
\code{.py}
out = skimage.transform.rescale(img, factor, mode='constant', cval=0)
\endcode

 * RNG is only called once, and its output is stored in
 * BilinearStaticFillerdata. Followup calls to the Filler function
 * will bypass RNG.
 */
template <typename Dtype>
class BilinearStaticFiller : public Filler<Dtype> {
 public:
  explicit BilinearStaticFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {
    bilinear_static_filler_flag_ = false;
  }
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    const int blob_count = blob->count();
    const int blob_width = blob->width();
    const int blob_height = blob->height();

    CHECK_EQ(blob->num_axes(), 4) << "Blob must be 4 dim.";
    CHECK_EQ(blob_width, blob_height) << "Filter must be square";
    if (bilinear_static_filler_flag_ == false) {
      bilinear_static_filler_data_.resize(blob_count);
      int f = ceil(blob_width / 2.);
      float c = (2 * f - 1 - f % 2) / (2. * f);
      for (int i = 0; i < blob_count; ++i) {
        float x = i % blob_width;
        float y = (i / blob_width) % blob_height;
        bilinear_static_filler_data_[i] =
          (1 - fabs(x / f - c)) * (1 - fabs(y / f - c));
      }
      bilinear_static_filler_flag_ = true;
    }
    caffe_copy(blob_count, &bilinear_static_filler_data_.front(), data);
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }


 private:
  std::vector<Dtype> bilinear_static_filler_data_;
  bool bilinear_static_filler_flag_;
};

/**
 * @brief Get a specific filler from the specification given in FillerParameter.
 *
 * Ideally this would be replaced by a factory pattern, but we will leave it
 * this way for now.
 */
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
  } else if (type == "msra") {
    return new MSRAFiller<Dtype>(param);
  } else if (type == "bilinear") {
    return new BilinearFiller<Dtype>(param);
  } else if (type == "gaussianstatic") {
    return new GaussianStaticFiller<Dtype>(param);
  } else if (type == "positive_unitballstatic") {
    return new PositiveUnitballStaticFiller<Dtype>(param);
  } else if (type == "uniformstatic") {
    return new UniformStaticFiller<Dtype>(param);
  } else if (type == "xavierstatic") {
    return new XavierStaticFiller<Dtype>(param);
  } else if (type == "msrastatic") {
    return new MSRAStaticFiller<Dtype>(param);
  } else if (type == "bilinearstatic") {
    return new BilinearStaticFiller<Dtype>(param);
  } else {
    CHECK(false) << "Unknown filler name: " << param.type();
  }
  return (Filler<Dtype>*)(NULL);
}

}  // namespace caffe

#endif  // CAFFE_FILLER_HPP_
