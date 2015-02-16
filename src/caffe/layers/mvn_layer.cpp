                                                       #include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

bool please_dump = false;
int train_iter = -1;

template <typename Dtype>
void MVNLayer<Dtype>::SetBlobFinder(const BlobFinder<Dtype> &blob_finder) {
  this->blob_helper_ = MvnBlobHelper<Dtype>(this->layer_param_, blob_finder);
}

template <typename Dtype>
void MVNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                            const vector<Blob<Dtype> *> &top) {
  const MVNParameter& param = this->layer_param_.mvn_param();
  // If the parameter specifies that the variance blob should be added to the
  // vector of top blobs, then the parameter must also specificy that
  // variance is to be normalized.
  bool it_has_var_blob = param.has_variance_blob();
  bool it_says_norm_var = param.normalize_variance();
  if (it_has_var_blob) {
    CHECK(it_has_var_blob && it_says_norm_var)
      << "MVNLayer " << this->layer_param_.name()
         << " specifies a top blob name for the variance blob, but does not "
         << "compute variance.";
  }
}

template <typename Dtype>
void MVNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Blob<Dtype>* input_blob = bottom[0];

  // Reshape the data output blob, which is always in the top vector.
  blob_helper_.DataBlob(top)->Reshape(input_blob->num(), input_blob->channels(),
      input_blob->height(), input_blob->width());

  mean_.Reshape(input_blob->num(), input_blob->channels(), 1, 1);
  if (blob_helper_.HasMeanTop()) {
    // If the mean_ is exported as a top blob, have to shape it too.
    blob_helper_.MeanBlob(top)->ReshapeLike(mean_);
  }

  variance_.Reshape(input_blob->num(), input_blob->channels(), 1, 1);
  if (blob_helper_.HasVarianceTop()) {
    // If variance_ is exported as a top blob, have to shape it too.
    blob_helper_.VarianceBlob(top)->ReshapeLike(variance_);
  }

  temp_.Reshape(input_blob->num(), input_blob->channels(),
      input_blob->height(), input_blob->width());
  sum_multiplier_.Reshape(1, 1, input_blob->height(), input_blob->width());
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
}

template <typename Dtype>
void MVNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();

  // Get the blob that has our output
  Blob<Dtype>* top_blob = blob_helper_.DataBlob(top);
  int num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();

  int dim = bottom[0]->count() / num;
  Dtype eps = 1e-10;

  if (this->layer_param_.mvn_param().normalize_variance()) {
    // put the squares of bottom into temp_
    caffe_powx(bottom[0]->count(), bottom_data, Dtype(2),
        temp_.mutable_cpu_data());

    std::ofstream stream;

    // computes variance using var(X) = E(X^2) - (EX)^2
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
        sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, temp_.cpu_data(),
        sum_multiplier_.cpu_data(), 0.,
        variance_.mutable_cpu_data());  // E(X^2)
    caffe_powx(mean_.count(), mean_.cpu_data(), Dtype(2),
        temp_.mutable_cpu_data());  // (EX)^2
    caffe_sub(mean_.count(), variance_.cpu_data(), temp_.cpu_data(),
        variance_.mutable_cpu_data());  // variance

    // Check for slightly negative values of the variance, which would cause
    // NaN result in the subsequent square root.
    Dtype* write_var = variance_.mutable_cpu_data();
    const Dtype* read_var = variance_.cpu_data();
    for (int i = 0; i < variance_.count(); ++i) {
      write_var[i] = read_var[i] < eps ? eps : read_var[i];
    }

    // do mean and variance normalization
    // subtract mean
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
            mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
            temp_.mutable_cpu_data());
    caffe_add(temp_.count(), bottom_data, temp_.cpu_data(),
              top_blob->mutable_cpu_data());
    // normalize variance
    caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5),
          variance_.mutable_cpu_data());
    caffe_add_scalar(variance_.count(), eps, variance_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
          temp_.mutable_cpu_data());
    caffe_div(temp_.count(), top_blob->cpu_data(), temp_.cpu_data(),
              top_blob->mutable_cpu_data());
    if (blob_helper_.HasVarianceTop()) {
      // If the variance is exported as a top blob, it should just mirror the
      // data in the member mean_ blob.
      blob_helper_.VarianceBlob(top)->ShareData(variance_);
    }
  } else {
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
            sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX

    // Outer product of the means vector and the vector of -1's to
    // create a matrix the same size as the bottom data.
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
            mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
            temp_.mutable_cpu_data());

    // subtract mean
    caffe_add(temp_.count(), bottom_data, temp_.cpu_data(),
              top_blob->mutable_cpu_data());
  }

  if (blob_helper_.HasMeanTop()) {
    // If the mean is exported as a top blob, it should just mirror the
    // data in the member mean_ blob.
    blob_helper_.MeanBlob(top)->ShareData(mean_);
  }
}

template <typename Dtype>
void MVNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  int num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();

  int dim = bottom[0]->count() / num;
  Dtype eps = 1e-10;

  if (this->layer_param_.mvn_param().normalize_variance()) {
    caffe_mul(temp_.count(), top_data, top_diff, bottom_diff);
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., bottom_diff,
          sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
          bottom_diff);
    caffe_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., top_diff,
            sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
            mean_.cpu_data(), sum_multiplier_.cpu_data(), 1.,
            bottom_diff);

    caffe_cpu_axpby(temp_.count(), Dtype(1), top_diff, Dtype(-1. / dim),
        bottom_diff);

    // put the squares of bottom into temp_
    caffe_powx(temp_.count(), bottom_data, Dtype(2),
        temp_.mutable_cpu_data());

    // computes variance using var(X) = E(X^2) - (EX)^2
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
        sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, temp_.cpu_data(),
        sum_multiplier_.cpu_data(), 0.,
        variance_.mutable_cpu_data());  // E(X^2)
    caffe_powx(mean_.count(), mean_.cpu_data(), Dtype(2),
        temp_.mutable_cpu_data());  // (EX)^2
    caffe_sub(mean_.count(), variance_.cpu_data(), temp_.cpu_data(),
        variance_.mutable_cpu_data());  // variance

    // normalize variance
    caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5),
          variance_.mutable_cpu_data());

    caffe_add_scalar(variance_.count(), eps, variance_.mutable_cpu_data());

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
        variance_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
        temp_.mutable_cpu_data());

    caffe_div(temp_.count(), bottom_diff, temp_.cpu_data(), bottom_diff);
  } else {
    caffe_copy(temp_.count(), top_diff, bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(MVNLayer);
#endif

INSTANTIATE_CLASS(MVNLayer);
REGISTER_LAYER_CLASS(MVN);

}  // namespace caffe
