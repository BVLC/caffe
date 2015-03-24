#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

#if defined(USE_OPENCL)
#include <caffe/util/OpenCL/OpenCLDevice.hpp>
#include <caffe/util/OpenCL/sigmoid_layer.hpp>
#endif

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
void SigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = sigmoid(bottom_data[i]);
  }
}

template <typename Dtype>
void SigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      const Dtype sigmoid_x = top_data[i];
      bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x);
    }
  }
}

#if defined(USE_OPENCL)

namespace OpenCL {

template<typename T>
bool clSigmoidLayerForward(const int count, const T* bottom_data, T* top_data) {

	std::string kernel_name = clGetKernelName<T>("SigmoidForward");

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	kernel = gpu->getKernel(kernel_name);
	if ( kernel == NULL ) {
		return false;
	}

	CL_SET_KERNEL_ARG
	CL_SET_TYPE_KERNEL_ARG(int, count)
	CL_SET_ARRAY_KERNEL_ARG(&bottom_data)
	CL_SET_ARRAY_KERNEL_ARG(&top_data)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(count, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(count, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	clFinish(*queue);
	LOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
};
template bool clSigmoidLayerForward<float>(const int count, const float* bottom_data, float* top_data);
template bool clSigmoidLayerForward<double>(const int count, const double* bottom_data, double* top_data);

template<typename T>
bool clSigmoidLayerBackward(const int count, const T* top_diff, const T* top_data, T* bottom_diff) {

	std::string kernel_name = clGetKernelName<T>("SigmoidBackward");

	queue = gpu->getQueue();
	if ( ! queue ) {
		LOG(ERROR) << gpu->name() << "> failed to get OpenCL command queue";
		return false;
	}

	kernel = gpu->getKernel(kernel_name);
	if ( kernel == NULL ) {
		return false;
	}

	CL_SET_KERNEL_ARG
	CL_SET_TYPE_KERNEL_ARG(int, count)
	CL_SET_ARRAY_KERNEL_ARG(&top_diff)
	CL_SET_ARRAY_KERNEL_ARG(&top_data)
	CL_SET_ARRAY_KERNEL_ARG(&bottom_diff)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(count, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(count, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	clFinish(*queue);
	LOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
};
template bool clSigmoidLayerBackward<float>(const int count, const float* top_diff, const float* top_data, float* bottom_diff);
template bool clSigmoidLayerBackward<double>(const int count, const double* top_diff, const double* top_data, double* bottom_diff);


} // namespace OpenCL


template<typename Dtype>
void SigmoidLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = (top)[0]->mutable_gpu_data();
	const int count = bottom[0]->count();
	// NOLINT_NEXT_LINE(whitespace/operators)
	/*
	SigmoidForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, top_data);
	CUDA_POST_KERNEL_CHECK;
	*/
	BOOL_CHECK( caffe::OpenCL::clSigmoidLayerForward(count, bottom_data, top_data) );
}

template<typename Dtype>
void SigmoidLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[0]) {
		const Dtype* top_data = top[0]->gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = (bottom)[0]->mutable_gpu_diff();
		const int count = (bottom)[0]->count();
		// NOLINT_NEXT_LINE(whitespace/operators)
		/*
		SigmoidBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, top_data, bottom_diff);
		CUDA_POST_KERNEL_CHECK;
		*/
		BOOL_CHECK( caffe::OpenCL::clSigmoidLayerBackward(count, top_diff, top_data, bottom_diff) );
	}
}

#endif // USE_OPENCL

#if defined(CPU_ONLY) && ! defined(USE_OPENCL)
STUB_GPU(SigmoidLayer);
#endif

INSTANTIATE_CLASS(SigmoidLayer);


}  // namespace caffe
