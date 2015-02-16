#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InverseMVNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>* bottom_blob = blob_helper_.DataBlob(bottom);
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

    // Fill a matrix of the same dimension as the input data blob that has
    // the variance at every location.
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_blob->gpu_data(), sum_multiplier_.gpu_data(), 0.,
          temp_.mutable_gpu_data());

    caffe_gpu_mul(temp_.count(), bottom_blob->gpu_data(), temp_.gpu_data(),
                  top[0]->mutable_gpu_data());

    // Create the matrix of means of the same dimension as the bottom and top
    // data,blob_helper_.DataBlob(bottom) so we can do element-wise addition to
    // add the mean back.
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
            mean_blob->gpu_data(), sum_multiplier_.gpu_data(), 0.,
            temp_.mutable_gpu_data());

    // Element-wise addition of the means.
    caffe_gpu_add(temp_.count(), top[0]->gpu_data(), temp_.gpu_data(),
        top[0]->mutable_gpu_data());
  } else {
    // Create the matrix of means of the same dimension as the bottom and top
    // data,blob_helper_.DataBlob(bottom) so we can do element-wise addition to
    // add the mean back.
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
            mean_blob->gpu_data(), sum_multiplier_.gpu_data(), 0.,
            temp_.mutable_gpu_data());

    // Element-wise addition of the means.
    caffe_gpu_add(temp_.count(), bottom_blob->gpu_data(), temp_.gpu_data(),
                  top[0]->mutable_gpu_data());
  }
}

template <typename Dtype>
void InverseMVNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();

  Blob<Dtype>* bottom_blob = blob_helper_.DataBlob(bottom);
  Dtype* bottom_diff = bottom_blob->mutable_gpu_diff();

  int num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom_blob->num();
  else
    num = bottom_blob->num() * bottom[0]->channels();

  int dim = bottom_blob->count() / num;

  if (this->layer_param_.mvn_param().normalize_variance()) {
    Blob<Dtype>* variance_blob = blob_helper_.VarianceBlob(bottom);

    // Create a blob of same shape as the top diffs, and put the
    // corresponding variance (actually standard deviation) in each
    // element.
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_blob->gpu_data(), sum_multiplier_.gpu_data(), 0.,
          temp_.mutable_gpu_data());

    // Multiply each top_diff by the variance (elementwise).
    caffe_gpu_mul(temp_.count(), temp_.gpu_data(), top_diff, bottom_diff);
  } else {
    caffe_copy(temp_.count(), top_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InverseMVNLayer);

}  // namespace caffe
