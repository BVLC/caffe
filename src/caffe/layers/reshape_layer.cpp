#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void ReshapeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1) << "Reshape Layer takes a single blob as input.";
  CHECK_EQ(top.size(), 1) << "Reshape Layer takes a single blob as output.";

  num_out = this->layer_param_.reshape_param().num();
  // Dimensions set to 0 (either by default or explicitly) will be copied from
  // the bottom layer.
  if (num_out == 0) {
    num_out = bottom[0]->num();
  }

  channels_out = this->layer_param_.reshape_param().channels();
  if (channels_out == 0) {
    channels_out = bottom[0]->channels();
  }

  width_out = this->layer_param_.reshape_param().width();
  if (width_out == 0) {
    width_out = bottom[0]->width();
  }

  height_out = this->layer_param_.reshape_param().height();
  if (height_out == 0) {
    height_out = bottom[0]->height();
  }

  FillInSingleUnspecifiedDimension(bottom[0]->count());
}

template <typename Dtype>
void ReshapeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(num_out, channels_out, height_out, width_out);

  const size_t out_count = num_out * channels_out * height_out * width_out;
  CHECK_EQ(out_count, bottom[0]->count()) <<
      "Bottom layer count isn't equal to predicted; output layer size is " <<
      num_out << "x" << channels_out << "x" << height_out << "x" << width_out;
}

template <typename Dtype>
void ReshapeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ShareData(*bottom[0]);
}

template <typename Dtype>
void ReshapeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  bottom[0]->ShareDiff(*top[0]);
}

/**
 * @brief Fill in a single dimension left unspecified.
 *
 * If a dimension is set to -1, it will be filled in with a value inferred from
 * the count of the bottom layer (if the product of the nonzero dimensions is a
 * divisor of the count).
 *
 * @param bottom_count Count of the bottom layer.
 */
template <typename Dtype>
void ReshapeLayer<Dtype>::FillInSingleUnspecifiedDimension(int bottom_count) {
  int* const dimensions[] = {&num_out, &channels_out, &width_out, &height_out};
  const size_t N_DIMENSIONS = 4;

  // How many -1 dimensions do we have.
  int n_unspecified = 0;
  // Product of the remaining dimensions.
  int product_without_unspecified_dim = 1;

  for (size_t i = 0; i < N_DIMENSIONS; i++) {
    if (*(dimensions[i]) == -1) {
      n_unspecified++;
    } else {
      product_without_unspecified_dim *= *(dimensions[i]);
    }
  }

  if (n_unspecified == 0) {
    // Everything is filled out, nothing to do.
    return;
  }

  CHECK_EQ(n_unspecified, 1) << "Only one dimension can be set -1.";
  CHECK_EQ(bottom_count % product_without_unspecified_dim, 0) <<
    "Bottom layer count " << bottom_count << " not divisible by product " <<
    product_without_unspecified_dim;

  // Fill up the one remaining dimension.
  for (size_t i = 0; i < N_DIMENSIONS; i++) {
    if (*(dimensions[i]) == -1) {
      *(dimensions[i]) = bottom_count / product_without_unspecified_dim;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ReshapeLayer);
#endif

INSTANTIATE_CLASS(ReshapeLayer);
REGISTER_LAYER_CLASS(RESHAPE, ReshapeLayer);
}  // namespace caffe
