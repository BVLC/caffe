#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

#if defined(USE_OPENCL)
#include <caffe/util/OpenCL/OpenCLDevice.hpp>
#include <caffe/util/OpenCL/relu_layer.hpp>
#include <caffe/util/OpenCL/definitions.hpp>
#include <caffe/util/benchmark.hpp>
#endif

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}

#if defined(USE_OPENCL)

namespace OpenCL {

template<typename T>
bool clReLULayerForward(const int count, const T* bottom_data, T* top_data, T negative_slope) {
  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();
  cl_command_queue* queue = device.getCurrentCommandQueue();

	std::string kernel_name = clGetKernelName<T>("ReLUForward");

  if ( ! queue ) {
    LOG(ERROR) << device.name() << "> failed to get OpenCL command queue";
		return false;
	}

  cl_kernel* kernel = device.getKernel(kernel_name);
	if ( kernel == NULL ) {
		return false;
	}

	CL_SET_KERNEL_ARG
  CL_SET_TYPE_KERNEL_ARG(int, count, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&bottom_data, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&top_data, kernel)
  CL_SET_TYPE_KERNEL_ARG(T, negative_slope, kernel)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(count, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(count, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
    LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<device.name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}

  DLOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<device.name();

	CL_SET_KERNEL_ARG_END

	return true;
};
template bool clReLULayerForward<float>(const int count, const float* bottom_data, float* top_data, float negative_slope);
template bool clReLULayerForward<double>(const int count, const double* bottom_data, double* top_data, double negative_slope);

template<typename T>
bool clReLULayerBackward(const int count, const T* top_diff, const T* bottom_data, T* bottom_diff, T negative_slope) {
  OpenCLDevice& device = OpenCLManager::CurrentPlatform()->CurrentDevice();

	std::string kernel_name = clGetKernelName<T>("ReLUBackward");

  cl_command_queue* queue = device.getCurrentCommandQueue();
	if ( ! queue ) {
    LOG(ERROR) << device.name() << "> failed to get OpenCL command queue";
		return false;
	}

  cl_kernel* kernel = device.getKernel(kernel_name);
	if ( kernel == NULL ) {
		return false;
	}

	CL_SET_KERNEL_ARG
  CL_SET_TYPE_KERNEL_ARG(int, count, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&top_diff, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&bottom_data, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&bottom_diff, kernel)
  CL_SET_TYPE_KERNEL_ARG(T, negative_slope, kernel)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(count, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(count, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
    LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<device.name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}

  DLOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<device.name();

	CL_SET_KERNEL_ARG_END

	return true;
};
template bool clReLULayerBackward<float>(const int count, const float* top_diff, const float* bottom_data, float* bottom_diff, float negative_slope);
template bool clReLULayerBackward<double>(const int count, const double* top_diff, const double* bottom_data, double* bottom_diff, double negative_slope);


} // namespace OpenCL


template<typename Dtype>
void ReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = (top)[0]->mutable_gpu_data();
	const int count = bottom[0]->count();
	Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
	// NOLINT_NEXT_LINE(whitespace/operators)
	/*
	ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, top_data, negative_slope);
	CUDA_POST_KERNEL_CHECK;
	*/
  TIME("caffe::OpenCL::clReLULayerForward()", {
      BOOL_CHECK( caffe::OpenCL::clReLULayerForward(count, bottom_data, top_data, negative_slope) );
  });
}

template<typename Dtype>
void ReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[0]) {
		const Dtype* bottom_data = (bottom)[0]->gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = (bottom)[0]->mutable_gpu_diff();
		const int count = (bottom)[0]->count();
		Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
		// NOLINT_NEXT_LINE(whitespace/operators)
		/*
		ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, bottom_data, bottom_diff, negative_slope);
		CUDA_POST_KERNEL_CHECK;
		*/
	  TIME("caffe::OpenCL::clReLULayerBackward()", {
		BOOL_CHECK( caffe::OpenCL::clReLULayerBackward(count, top_diff, bottom_data, bottom_diff, negative_slope) );
	  });
	}
}

#endif // USE_OPENCL

#if defined(CPU_ONLY) && ! defined(USE_OPENCL)
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);
//REGISTER_LAYER_CLASS(ReLU);

}  // namespace caffe
