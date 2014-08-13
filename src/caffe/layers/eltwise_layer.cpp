#include <vector>
#include <cfloat>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EltwiseLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  CHECK(this->layer_param().eltwise_param().coeff_size() == 0
      || this->layer_param().eltwise_param().coeff_size() == bottom.size()) <<
      "Eltwise Layer takes one coefficient per bottom blob.";
  CHECK(!(this->layer_param().eltwise_param().operation()
      == EltwiseParameter_EltwiseOp_PROD
      && this->layer_param().eltwise_param().coeff_size())) <<
      "Eltwise layer only takes coefficients for summation.";
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK_EQ(num, bottom[i]->num());
    CHECK_EQ(channels, bottom[i]->channels());
    CHECK_EQ(height, bottom[i]->height());
    CHECK_EQ(width, bottom[i]->width());
  }
  (*top)[0]->Reshape(num, channels, height, width);
  op_ = this->layer_param_.eltwise_param().operation();
  // Blob-wise coefficients for the elementwise operation.
  coeffs_ = vector<Dtype>(bottom.size(), 1);
  if (this->layer_param().eltwise_param().coeff_size()) {
    for (int i = 0; i < bottom.size(); ++i) {
      coeffs_[i] = this->layer_param().eltwise_param().coeff(i);
    }
  }
  // If max operation, we will initialize the vector index part.
  if (this->layer_param_.eltwise_param().operation() ==
		  EltwiseParameter_EltwiseOp_MAX && top->size() == 1) {
    max_idx_.reset(new Blob<int>(bottom[0]->num(), channels,
                                 height, width));
  }
}

template <typename Dtype>
Dtype EltwiseLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  int* mask = NULL;
  const Dtype* bottom_data_a = NULL;
  const Dtype* bottom_data_b = NULL;
  const int count = (*top)[0]->count();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
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
    for (int i = 0; i < bottom.size(); ++i) {
      caffe_axpy(count, coeffs_[i], bottom[i]->cpu_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_MAX:
	// Initialize
	mask = max_idx_->mutable_cpu_data();
	caffe_set(count, -1, mask);
	caffe_set(count, Dtype(-FLT_MAX), top_data);
	// bottom 0 & 1
	bottom_data_a = bottom[0]->cpu_data();
	bottom_data_b = bottom[1]->cpu_data();
	for (int idx = 0; idx < count; ++idx) {
		if (bottom_data_a[idx] > bottom_data_b[idx]) {
			top_data[idx] = bottom_data_a[idx]; // maxval
			mask[idx] = 0; // maxid
		} else {
			top_data[idx] = bottom_data_b[idx]; // maxval
			mask[idx] = 1; // maxid
		}
	}
	// bottom 2++
	bottom_data_a = top_data;
	for (int blob_idx = 2; blob_idx < bottom.size(); ++blob_idx) {
		bottom_data_b = bottom[blob_idx]->cpu_data();
		for (int idx = 0; idx < count; ++idx) {
			if (bottom_data_a[idx] < bottom_data_b[idx]) {
				top_data[idx] = bottom_data_b[idx];	// maxval
				mask[idx] = blob_idx;	// maxid
			}
		}
	}
	/*
	NOT_IMPLEMENTED;
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	// The main loop
	for (int n = 0; n < bottom[0]->num(); ++n) {
		for (int c = 0; c < channels; ++c) {
			for (int h = 0; h < height; ++h) {
				for (int w = 0; w < width; ++w) {
				}
			}
		}
	}
	*/
	break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
  return Dtype(0.);
}

template <typename Dtype>
void EltwiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  const int* mask = NULL;
  const int count = top[0]->count();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  for (int i = 0; i < bottom->size(); ++i) {
    if (propagate_down[i]) {
      const Dtype* bottom_data = (*bottom)[i]->cpu_data();
      Dtype* bottom_diff = (*bottom)[i]->mutable_cpu_diff();
      switch (op_) {
      case EltwiseParameter_EltwiseOp_PROD:
        caffe_div(count, top_data, bottom_data, bottom_diff);
        caffe_mul(count, bottom_diff, top_diff, bottom_diff);
        break;
      case EltwiseParameter_EltwiseOp_SUM:
        if (coeffs_[i] == Dtype(1)) {
          caffe_copy(count, top_diff, bottom_diff);
        } else {
          caffe_cpu_scale(count, coeffs_[i], top_diff, bottom_diff);
        }
        break;
      case EltwiseParameter_EltwiseOp_MAX:
    	mask = max_idx_->cpu_data();
    	caffe_set(count, Dtype(0), bottom_diff);
    	for (int idx = 0; idx < count; ++idx) {
    		Dtype gradient = 0;
    		if (mask[idx] == i) {
    			gradient += top_diff[idx];
    		}
    		bottom_diff[idx] = gradient;
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


}  // namespace caffe
