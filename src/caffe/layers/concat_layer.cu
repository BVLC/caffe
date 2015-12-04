#include <vector>

<<<<<<< HEAD
#include "caffe/layers/concat_layer.hpp"
=======
#include "caffe/common_layers.hpp"
>>>>>>> pod-caffe-pod.hpp-merge
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
__global__ void Concat(const int nthreads, const Dtype* in_data,
    const bool forward, const int num_concats, const int concat_size,
    const int top_concat_axis, const int bottom_concat_axis,
    const int offset_concat_axis, Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int total_concat_size = concat_size * bottom_concat_axis;
    const int concat_num = index / total_concat_size;
    const int concat_index = index % total_concat_size;
    const int top_index = concat_index +
        (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
    if (forward) {
      out_data[top_index] = in_data[index];
    } else {
      out_data[index] = in_data[top_index];
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    }
  }
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
    }
  }
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
    }
  }
>>>>>>> pod-caffe-pod.hpp-merge
}

template <typename Dtype>
void ConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom.size() == 1) { return; }
  Dtype* top_data = top[0]->mutable_gpu_data();
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  const bool kForward = true;
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    const int bottom_concat_size = bottom_concat_axis * concat_input_size_;
    const int nthreads = bottom_concat_size * num_concats_;
    Concat<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, bottom_data, kForward, num_concats_, concat_input_size_,
        top_concat_axis, bottom_concat_axis, offset_concat_axis, top_data);
    offset_concat_axis += bottom_concat_axis;
  }
=======
void ConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  if (concat_dim_ == 0) {
    int offset_num = 0;
    for (int i = 0; i < bottom.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      caffe_copy(bottom[i]->count(), bottom_data,
        top_data + top[0]->offset(offset_num));
      offset_num += bottom[i]->num();
    }
  } else if (concat_dim_ == 1) {
    int offset_channel = 0;
    for (int i = 0; i < bottom.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      int num_elem =
        bottom[i]->channels() * bottom[i]->height() * bottom[i]->width();
      for (int n = 0; n < num_; ++n) {
        caffe_copy(num_elem, bottom_data+bottom[i]->offset(n),
          top_data + top[0]->offset(n, offset_channel));
      }
      offset_channel += bottom[i]->channels();
    }
  } else {
    LOG(FATAL) << "concat_dim along dim" << concat_dim_ <<
      " not implemented yet";
  }
>>>>>>> origin/BVLC/parallel
=======
    }
  }
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
    }
  }
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
}

template <typename Dtype>
void ConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom.size() == 1) { return; }
  Dtype* top_data = top[0]->mutable_gpu_data();
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  const bool kForward = true;
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    const int bottom_concat_size = bottom_concat_axis * concat_input_size_;
    const int nthreads = bottom_concat_size * num_concats_;
    Concat<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, bottom_data, kForward, num_concats_, concat_input_size_,
        top_concat_axis, bottom_concat_axis, offset_concat_axis, top_data);
    offset_concat_axis += bottom_concat_axis;
  }
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
}

template <typename Dtype>
void ConcatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  if (bottom.size() == 1) { return; }
  const Dtype* top_diff = top[0]->gpu_diff();
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
  if (bottom.size() == 1) { return; }
  const Dtype* top_diff = top[0]->gpu_diff();
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
  if (bottom.size() == 1) { return; }
  const Dtype* top_diff = top[0]->gpu_diff();
=======
<<<<<<< HEAD
<<<<<<< HEAD
  if (bottom.size() == 1) { return; }
  const Dtype* top_diff = top[0]->gpu_diff();
>>>>>>> pod/device/blob.hpp
=======
  if (bottom.size() == 1) { return; }
  const Dtype* top_diff = top[0]->gpu_diff();
>>>>>>> pod-caffe-pod.hpp-merge
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  const bool kForward = false;
  for (int i = 0; i < bottom.size(); ++i) {
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    if (propagate_down[i]) {
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      const int bottom_concat_size = bottom_concat_axis * concat_input_size_;
      const int nthreads = bottom_concat_size * num_concats_;
      Concat<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
          nthreads, top_diff, kForward, num_concats_, concat_input_size_,
          top_concat_axis, bottom_concat_axis, offset_concat_axis, bottom_diff);
=======
  const Dtype* top_diff = top[0]->gpu_diff();
  if (concat_dim_ == 0) {
    int offset_num = 0;
    for (int i = 0; i < bottom.size(); ++i) {
      Blob<Dtype>* blob = bottom[i];
      if (propagate_down[i]) {
        Dtype* bottom_diff = blob->mutable_gpu_diff();
        caffe_copy(blob->count(), top_diff + top[0]->offset(offset_num),
                       bottom_diff);
      }
      offset_num += blob->num();
    }
  } else if (concat_dim_ == 1) {
    int offset_channel = 0;
    for (int i = 0; i < bottom.size(); ++i) {
      Blob<Dtype>* blob = bottom[i];
      if (propagate_down[i]) {
        Dtype* bottom_diff = blob->mutable_gpu_diff();
        int num_elem = blob->channels()*blob->height()*blob->width();
        for (int n = 0; n < num_; ++n) {
          caffe_copy(num_elem, top_diff + top[0]->offset(n, offset_channel),
                         bottom_diff + blob->offset(n));
        }
      }
      offset_channel += blob->channels();
>>>>>>> origin/BVLC/parallel
=======
  if (bottom.size() == 1) { return; }
  const Dtype* top_diff = top[0]->gpu_diff();
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
  if (bottom.size() == 1) { return; }
  const Dtype* top_diff = top[0]->gpu_diff();
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  const bool kForward = false;
  for (int i = 0; i < bottom.size(); ++i) {
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    if (propagate_down[i]) {
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      const int bottom_concat_size = bottom_concat_axis * concat_input_size_;
      const int nthreads = bottom_concat_size * num_concats_;
      Concat<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
          nthreads, top_diff, kForward, num_concats_, concat_input_size_,
          top_concat_axis, bottom_concat_axis, offset_concat_axis, bottom_diff);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
    }
    offset_concat_axis += bottom_concat_axis;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConcatLayer);

}  // namespace caffe
