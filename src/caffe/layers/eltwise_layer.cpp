#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EltwiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  op_ = this->layer_param_.eltwise_param().operation();
  coeff_blob_ = this->layer_param().eltwise_param().coeff_blob();
  if (coeff_blob_) {
    CHECK_EQ(op_, EltwiseParameter_EltwiseOp_SUM)
        << "coeff_blob option only implemented for the SUM operation";
  }
  const int coeff_size = this->layer_param().eltwise_param().coeff_size();
  CHECK(coeff_size == 0 || (!coeff_blob_ && coeff_size == bottom.size())
                        || (coeff_blob_ && coeff_size == bottom.size() - 1)) <<
      "Eltwise Layer takes one coefficient per bottom blob.";
  CHECK(op_ == EltwiseParameter_EltwiseOp_SUM
      || this->layer_param().eltwise_param().coeff_size() == 0) <<
      "Eltwise layer only takes coefficients for summation.";
  // Blob-wise coefficients for the elementwise operation.
  coeffs_.resize(bottom.size(), 1);
  if (coeff_size) {
    for (int i = 0; i < bottom.size() - coeff_blob_; ++i) {
      coeffs_[i] = this->layer_param().eltwise_param().coeff(i);
    }
  }
  stable_prod_grad_ = this->layer_param_.eltwise_param().stable_prod_grad();
}

template <typename Dtype>
void EltwiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    if (coeff_blob_ && i == bottom.size() - 1) {
      CHECK_EQ(i, bottom[i]->shape(0))
          << "Dimension of coeff blob axis 0 must equal the number of bottom "
          << "blobs (not including the coeff blob itself).";
      for (int input_axis = 0, coeff_axis = 1;
           coeff_axis < bottom[i]->num_axes(); ++input_axis, ++coeff_axis) {
        CHECK_EQ(bottom[0]->shape(input_axis), bottom[i]->shape(coeff_axis))
            << "Each axis i >= 1 of the coeff blob must match the (i-1)th "
            << "axis of the input.";
      }
    } else {
      CHECK(bottom[i]->shape() == bottom[0]->shape());
    }
  }
  top[0]->ReshapeLike(*bottom[0]);
  // If max operation, we will initialize the vector index part.
  if (this->layer_param_.eltwise_param().operation() ==
      EltwiseParameter_EltwiseOp_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->shape());
  }
}

template <typename Dtype>
void EltwiseLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int* mask = NULL;
  const Dtype* bottom_data_a = NULL;
  const Dtype* bottom_data_b = NULL;
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
    caffe_mul(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), top_data);
    for (int i = 2; i < bottom.size(); ++i) {
      caffe_mul(count, top_data, bottom[i]->cpu_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    caffe_set(count, Dtype(0), top_data);
    // TODO(shelhamer) does BLAS optimize to sum for coeff = 1?
    for (int i = 0; i < bottom.size() - coeff_blob_; ++i) {
      if (coeff_blob_) {
        const int num = bottom[bottom.size() - 1]->count() /
                        (bottom.size() - 1);
        const int dim = bottom[i]->count() / num;
        const Dtype* bottom_data = bottom[i]->cpu_data();
        const Dtype* coeff_data = bottom[bottom.size() - 1]->cpu_data();
        for (int j = 0; j < num; ++j, bottom_data += dim, top_data += dim) {
          const Dtype coeff = coeffs_[i] * coeff_data[i * num + j];
          caffe_axpy(dim, coeff, bottom_data, top_data);
        }
        top_data = top[0]->mutable_cpu_data();
      } else {
        caffe_axpy(count, coeffs_[i], bottom[i]->cpu_data(), top_data);
      }
    }
    break;
  case EltwiseParameter_EltwiseOp_MAX:
    // Initialize
    mask = max_idx_.mutable_cpu_data();
    caffe_set(count, -1, mask);
    caffe_set(count, Dtype(-FLT_MAX), top_data);
    // bottom 0 & 1
    bottom_data_a = bottom[0]->cpu_data();
    bottom_data_b = bottom[1]->cpu_data();
    for (int idx = 0; idx < count; ++idx) {
      if (bottom_data_a[idx] > bottom_data_b[idx]) {
        top_data[idx] = bottom_data_a[idx];  // maxval
        mask[idx] = 0;  // maxid
      } else {
        top_data[idx] = bottom_data_b[idx];  // maxval
        mask[idx] = 1;  // maxid
      }
    }
    // bottom 2++
    for (int blob_idx = 2; blob_idx < bottom.size(); ++blob_idx) {
      bottom_data_b = bottom[blob_idx]->cpu_data();
      for (int idx = 0; idx < count; ++idx) {
        if (bottom_data_b[idx] > top_data[idx]) {
          top_data[idx] = bottom_data_b[idx];  // maxval
          mask[idx] = blob_idx;  // maxid
        }
      }
    }
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
}

template <typename Dtype>
void EltwiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int* mask = NULL;
  const int count = top[0]->count();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  for (int i = 0; i < bottom.size() - coeff_blob_; ++i) {
    if (propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
      switch (op_) {
      case EltwiseParameter_EltwiseOp_PROD:
        if (stable_prod_grad_) {
          bool initialized = false;
          for (int j = 0; j < bottom.size(); ++j) {
            if (i == j) { continue; }
            if (!initialized) {
              caffe_copy(count, bottom[j]->cpu_data(), bottom_diff);
              initialized = true;
            } else {
              caffe_mul(count, bottom[j]->cpu_data(), bottom_diff,
                        bottom_diff);
            }
          }
        } else {
          caffe_div(count, top_data, bottom_data, bottom_diff);
        }
        caffe_mul(count, bottom_diff, top_diff, bottom_diff);
        break;
      case EltwiseParameter_EltwiseOp_SUM:
        if (coeff_blob_) {
          const int num = bottom[bottom.size() - 1]->count() /
                          (bottom.size() - 1);
          const int dim = bottom[i]->count() / num;
          const Dtype* coeff_data = bottom[bottom.size() - 1]->cpu_data();
          for (int j = 0; j < num; ++j, bottom_diff += dim, top_diff += dim) {
            const Dtype coeff = coeffs_[i] * coeff_data[i * num + j];
            caffe_cpu_scale(dim, coeff, top_diff, bottom_diff);
          }
        } else if (coeffs_[i] == Dtype(1.)) {
          caffe_copy(count, top_diff, bottom_diff);
        } else {
          caffe_cpu_scale(count, coeffs_[i], top_diff, bottom_diff);
        }
        break;
      case EltwiseParameter_EltwiseOp_MAX:
        mask = max_idx_.cpu_data();
        for (int index = 0; index < count; ++index) {
          Dtype gradient = 0;
          if (mask[index] == i) {
            gradient += top_diff[index];
          }
          bottom_diff[index] = gradient;
        }
        break;
      default:
        LOG(FATAL) << "Unknown elementwise operation.";
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EltwiseLayer);
#endif

INSTANTIATE_CLASS(EltwiseLayer);
REGISTER_LAYER_CLASS(Eltwise);

}  // namespace caffe
