// Copyright 2014 BVLC and contributors.

#include "caffe/common.hpp"
#include "caffe/util/math_backends.hpp"

namespace caffe {

template<typename Dtype>
MathBackend<Dtype>* MathBackendFactory<Dtype>::GetMathBackend() {
	switch (Caffe::mode()) {
	case Caffe::CPU:
		return cpu_math_backend_;
	case Caffe::GPU:
		return gpu_math_backend_;
	default:
		LOG(FATAL) << "Unknown caffe mode.";
		return static_cast<MathBackend<Dtype>*>(NULL);
	}
}
template<typename Dtype>
MathBackend<Dtype>* MathBackendFactory<Dtype>::cpu_math_backend_ =
		new CPUMathBackend<Dtype>();
template<typename Dtype>
MathBackend<Dtype>* MathBackendFactory<Dtype>::gpu_math_backend_ =
		new GPUMathBackend<Dtype>();

INSTANTIATE_CLASS(MathBackendFactory);

}  // namespace caffe
