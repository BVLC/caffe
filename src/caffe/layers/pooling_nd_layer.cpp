#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template<typename Dtype>
void PoolingNDLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  // Set the max number of top blobs before calling base Layer::SetUp.
  // If doing MAX pooling, we can optionally output an extra top Blob
  // for the mask.  Otherwise, we only have one top Blob.
  if (this->layer_param_.pooling_param().pool()
      == PoolingParameter_PoolMethod_MAX) {
    max_top_blobs_ = 2;
  } else {
    max_top_blobs_ = 1;
  }
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(pool_param.axis());
  channels_ = bottom[0]->shape(channel_axis_);

  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 1);
  vector<int> size_shape(1, num_spatial_axes_);

  kernel_shape_.Reshape(size_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();

  CHECK(!(pool_param.kernel_size_size() > 0) !=
      !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK((pool_param.kernel_size_size() > 0) ||
      (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";

  if (pool_param.has_kernel_h() && pool_param.has_kernel_w()) {
    kernel_shape_data[0] = pool_param.kernel_h();
    kernel_shape_data[1] = pool_param.kernel_w();
  } else {
    const int num_kernel_dims = pool_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_);
    for (int i = 0; i < num_spatial_axes_; ++i) {
      kernel_shape_data[i] =
          pool_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
    }
  }

  // Setup stride dimensions (stride_).
  stride_.Reshape(size_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (pool_param.has_stride_h() || pool_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, pool_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = pool_param.stride_h();
    stride_data[1] = pool_param.stride_w();
  } else {
    const int num_stride_dims = pool_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims);";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          pool_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(size_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (pool_param.has_pad_h() || pool_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, pool_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = pool_param.pad_h();
    pad_data[1] = pool_param.pad_w();
  } else {
    const int num_pad_dims = pool_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims);";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          pool_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup kernel stride dimensions
  kstride_.Reshape(size_shape);
  int* kstride_data = kstride_.mutable_cpu_data();
  if (pool_param.has_kstride_h() || pool_param.has_kstride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kstride_h & kstride_w can only be used for 2D convolution.";
    CHECK_EQ(0, pool_param.kstride_size())
        << "Etiher kstride or kstirde_h/w should be specified; not both.";
    kstride_data[0] = pool_param.pad_h();
    kstride_data[1] = pool_param.pad_w();
  } else {
    const int num_kstride_dims = pool_param.kstride_size();
    CHECK(num_kstride_dims == 0 || num_kstride_dims == 1 ||
          num_kstride_dims == num_spatial_axes_)
      << "kstride must be specified once, or once per spatial dimension "
      << "(kstride specified " << num_kstride_dims << " times; "
      << num_spatial_axes_ << " spatial dims);";
    const int kDefaultKstride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      kstride_data[i] = (num_kstride_dims == 0) ? kDefaultKstride :
          pool_param.kstride((num_kstride_dims == 1) ? 0 : i);
    }
  }

  size_.Reshape(size_shape);
  pooled_size_.Reshape(size_shape);
  ext_kernel_shape_.Reshape(size_shape);
  int* size_data = size_.mutable_cpu_data();
  int* pooled_size_data = pooled_size_.mutable_cpu_data();
  int* ext_kernel_shape_data = ext_kernel_shape_.mutable_cpu_data();

  vector<int> top_shape = bottom[0]->shape();
  for (int i = 0; i < num_spatial_axes_; ++i) {
    size_data[i] = bottom[0]->shape(channel_axis_ + 1 + i);
    ext_kernel_shape_data[i] = (kernel_shape_data[i] - 1) * kstride_data[i] + 1;
    pooled_size_data[i] = static_cast<int>(ceil(
        static_cast<float>(size_data[i] + 2 * pad_data[i]
            - ext_kernel_shape_data[i]) / stride_data[i])) + 1;
    top_shape[channel_axis_ + 1 + i] = pooled_size_data[i];
  }
  top[0]->Reshape(top_shape);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool()
      == PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(top_shape);
  }
}

template<typename Dtype>
void PoolingNDLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  LayerSetUp(bottom, top);
}

template<typename Dtype>
void PoolingNDLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  LOG(FATAL)<< "Forward_cpu() not implemented in PoolingNDLayer.";
}

template<typename Dtype>
void PoolingNDLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL)<< "Backward_cpu() not implemented in PoolingNDLayer.";
  return;
}

#ifdef CPU_ONLY
STUB_GPU(PoolingNDLayer);
#endif

INSTANTIATE_CLASS(PoolingNDLayer);
REGISTER_LAYER_CLASS(PoolingND);

}    // namespace caffe
