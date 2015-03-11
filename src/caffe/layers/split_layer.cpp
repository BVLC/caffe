#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SplitLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (this->layer_param_.split_param().top_device_size() != 0) {
    CHECK_EQ(this->layer_param_.split_param().top_device_size(), top.size());
  }
  CUDA_CHECK(cudaStreamCreate(&stream_));
}

template <typename Dtype>
SplitLayer<Dtype>::~SplitLayer() {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaStreamDestroy(stream_));
#endif
}

template <typename Dtype>
void SplitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  count_ = bottom[0]->count();
  for (int i = 0; i < top.size(); ++i) {
    // Do not allow in-place computation in the SplitLayer.  Instead, share data
    // by reference in the forward pass, and keep separate diff allocations in
    // the backward pass.  (Technically, it should be possible to share the diff
    // blob of the first split output with the input, but this seems to cause
    // some strange effects in practice...)
    CHECK_NE(top[i], bottom[0]) << this->type() << " Layer does not "
        "allow in-place computation.";
    top[i]->ReshapeLike(*bottom[0]);
    CHECK_EQ(count_, top[i]->count());
  }
  copy_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void SplitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (this->layer_param_.split_param().top_device_size() == 0) {
    for (int i = 0; i < top.size(); ++i) {
      top[i]->ShareData(*bottom[0]);
    }
    return;
  }
#ifndef CPU_ONLY
  const DeviceParameter device = this->layer_param_.device();
  const Dtype* bottom_data = bottom[0]->data(device.type());
  for (int i = 0; i < top.size(); ++i) {
    const DeviceParameter top_device =
      this->layer_param_.split_param().top_device(i);
    if (top_device.type() == device.type()
        && top_device.device_id() == device.device_id()) {
      top[i]->ShareData(*bottom[0]);
    } else {
      int current_device;
      CUDA_CHECK(cudaGetDevice(&current_device));
      if (top_device.type() == DeviceParameter_DeviceType_GPU) {
        CUDA_CHECK(cudaSetDevice(top_device.device_id()));
      }
      Dtype* top_data = top[i]->mutable_data(top_device.type());
      CUDA_CHECK(cudaSetDevice(current_device));

      CUDA_CHECK(cudaMemcpyAsync(top_data, bottom_data, count_ * sizeof(Dtype),
          cudaMemcpyDefault, stream_));
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream_));
#endif
}

template <typename Dtype>
void SplitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  if (this->layer_param_.split_param().top_device_size() == 0) {
    if (top.size() == 1) {
      caffe_copy(count_, top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
      return;
    }
    caffe_add(count_, top[0]->cpu_diff(), top[1]->cpu_diff(),
              bottom[0]->mutable_cpu_diff());
    // Add remaining top blob diffs.
    for (int i = 2; i < top.size(); ++i) {
      const Dtype* top_diff = top[i]->cpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      caffe_axpy(count_, Dtype(1.), top_diff, bottom_diff);
    }
    return;
  }
#ifndef CPU_ONLY
  const DeviceParameter device = this->layer_param_.device();
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), stream_));
  Dtype *bottom_diff = bottom[0]->mutable_diff(device.type());
  CUDA_CHECK(cudaMemsetAsync(bottom_diff, 0, count_ * sizeof(Dtype), stream_));
  // Add remaining top blob diffs.
  for (int i = 0; i < top.size(); ++i) {
    const DeviceParameter top_device =
      this->layer_param_.split_param().top_device(i);
    const Dtype* top_diff = top[i]->diff(top_device.type());
    if (top_device.type() == device.type()
        && top_device.device_id() == device.device_id()) {
      if (device.type() == DeviceParameter_DeviceType_CPU) {
        caffe_axpy(count_, Dtype(1.), top_diff, bottom_diff);
      } else {
        caffe_gpu_axpy(count_, Dtype(1.), top_diff, bottom_diff);
      }
    } else {
      Dtype* copied_diff = copy_.mutable_data(device.type());
      CUDA_CHECK(cudaMemcpyAsync(copied_diff, top_diff, count_ * sizeof(Dtype),
          cudaMemcpyDefault, stream_));
      if (device.type() == DeviceParameter_DeviceType_CPU) {
        caffe_axpy(count_, Dtype(1.), copied_diff, bottom_diff);
      } else {
        caffe_gpu_axpy(count_, Dtype(1.), copied_diff, bottom_diff);
      }
    }
  }
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), NULL));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
#endif
}

#ifdef CPU_ONLY
STUB_GPU(SplitLayer);
#endif

INSTANTIATE_CLASS(SplitLayer);
REGISTER_LAYER_CLASS(Split);

}  // namespace caffe
