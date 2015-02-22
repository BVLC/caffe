#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InverseMVNLayer<Dtype>::SetBlobFinder(
    const BlobFinder<Dtype> &blob_finder) {
  this->blob_helper_ = MvnBlobHelper<Dtype>(this->layer_param_, blob_finder);
}

template <typename Dtype>
void InverseMVNLayer<Dtype>::LayerSetUp(
        const std::vector<Blob<Dtype> *> &bottom,
        const std::vector<Blob<Dtype> *> &top) {
  const MVNParameter& param = this->layer_param_.mvn_param();

  // If the parameter specifies a name for the variance and or mean blob,
  // then that blob must appear in the bottom vector of blobs.
  bool specifies_var_blob = param.has_variance_blob();
  if (specifies_var_blob) {
    CHECK(blob_helper_.VarianceBlob(bottom) != NULL) << "InverseMVNLayer " <<
      this->layer_param_.name() << " specifies variance blob " <<
      param.variance_blob() << " but it was not found in bottom blobs.";
  }

  bool it_has_mean_blob = param.has_mean_blob();
  CHECK(it_has_mean_blob) << "InverseMVNLayer requires a mean in the "
                             << "bottom blobs.";

  CHECK(blob_helper_.MeanBlob(bottom) != NULL) << "Mean blob "
         << param.mean_blob() << " not found bottom blobs of layer "
         << this->layer_param_.name();

  CHECK(blob_helper_.DataBlob(bottom) != NULL) << "No bottom data blob found "
    << "for InverseMVNLayer " << this->layer_param_.name();
}

template <typename Dtype>
void InverseMVNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Blob<Dtype>* output_blob = top[0];
  Blob<Dtype>* input_blob = blob_helper_.DataBlob(bottom);

  // Reshape the data output blob, which is always in the top vector.
  output_blob->Reshape(input_blob->num(), input_blob->channels(),
                       input_blob->height(), input_blob->width());

  temp_.Reshape(input_blob->num(), input_blob->channels(),
      input_blob->height(), input_blob->width());
  if ( this->layer_param_.mvn_param().across_channels() ) {
    sum_multiplier_.Reshape(1, input_blob->channels(), input_blob->height(),
                            input_blob->width());
  } else {
    sum_multiplier_.Reshape(1, 1, input_blob->height(), input_blob->width());
  }
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
}

template <typename Dtype>
void InverseMVNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>* bottom_blob = blob_helper_.DataBlob(bottom);
  const Dtype* bottom_data = bottom_blob->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom_blob->num();
  else
    num = bottom_blob->num() * bottom_blob->channels();

  int dim = bottom_blob->count() / num;

  Blob<Dtype>* mean_blob = blob_helper_.MeanBlob(bottom);
  if (this->layer_param_.mvn_param().normalize_variance()) {
    // Get the variance blob.
    Blob<Dtype>* variance_blob = blob_helper_.VarianceBlob(bottom);

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_blob->cpu_data(), sum_multiplier_.cpu_data(), 0.,
          temp_.mutable_cpu_data());

    // element-wise un-normalization for variance.
    caffe_mul(temp_.count(), bottom_data, temp_.cpu_data(), top_data);

    // Create the matrix of means of the same dimension as the bottom and top
    // data,blob_helper_.DataBlob(bottom) so we can do element-wise addition to
    // add the mean back.
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
            mean_blob->cpu_data(), sum_multiplier_.cpu_data(), 0.,
            temp_.mutable_cpu_data());

    // Element-wise addition of the means.
    caffe_add(temp_.count(), top_data, temp_.cpu_data(), top_data);
  } else {
    // Create the matrix of means of the same dimension as the bottom and top
    // data,blob_helper_.DataBlob(bottom) so we can do element-wise addition to
    // add the mean back.
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
            mean_blob->cpu_data(), sum_multiplier_.cpu_data(), 0.,
            temp_.mutable_cpu_data());

    // Element-wise addition of the means.
    caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);
  }
}

template <typename Dtype>
void InverseMVNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();

  Blob<Dtype>* bottom_blob = blob_helper_.DataBlob(bottom);
  Dtype* bottom_diff = bottom_blob->mutable_cpu_diff();

  int num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom_blob->num();
  else
    num = bottom_blob->num() * bottom[0]->channels();

  int dim = bottom_blob->count() / num;

  if (this->layer_param_.mvn_param().normalize_variance()) {
    Blob<Dtype>* variance_blob = blob_helper_.VarianceBlob(bottom);

    // Create a blob of same shape as the top diffs, and put the
    // corresponding variance in each element.
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_blob->cpu_data(), sum_multiplier_.cpu_data(), 0.,
          temp_.mutable_cpu_data());

    // Multiply each top_diff by the variance (elementwise).
    caffe_mul(temp_.count(), temp_.cpu_data(), top_diff, bottom_diff);
  } else {
    caffe_copy(temp_.count(), top_diff, bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(InverseMVNLayer);
#endif

INSTANTIATE_CLASS(InverseMVNLayer);
REGISTER_LAYER_CLASS(InverseMVN);
}  // namespace caffe
