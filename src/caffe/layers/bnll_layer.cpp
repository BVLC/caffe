#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

#if defined(USE_OPENCL)
#include <caffe/util/OpenCL/OpenCLDevice.hpp>
#include <caffe/util/OpenCL/bnll_layer.hpp>
#endif

namespace caffe {

const float kBNLL_THRESHOLD = 50.;

template <typename Dtype>
void BNLLLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = bottom_data[i] > 0 ?
        bottom_data[i] + log(1. + exp(-bottom_data[i])) :
        log(1. + exp(bottom_data[i]));
  }
}

template <typename Dtype>
void BNLLLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype expval;
    for (int i = 0; i < count; ++i) {
      expval = exp(std::min(bottom_data[i], Dtype(kBNLL_THRESHOLD)));
      bottom_diff[i] = top_diff[i] * expval / (expval + 1.);
    }
  }
}

#if defined(USE_OPENCL1)

namespace OpenCL {

template<typename T>
bool clBNLLLayerForward(const int count, const T* bottom_data, T* top_data) {

	std::string kernel_name = clGetKernelName<T>("BNLLForward");

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
template bool clBNLLLayerForward<float>(const int count, const float* bottom_data, float* top_data);
template bool clBNLLLayerForward<double>(const int count, const double* bottom_data, double* top_data);

template<typename T>
bool clBNLLLayerBackward(const int count, const T* top_diff, const T* bottom_data, T* bottom_diff) {

	std::string kernel_name = clGetKernelName<T>("BNLLBackward");

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
	CL_SET_ARRAY_KERNEL_ARG(&bottom_data)
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
template bool clBNLLLayerBackward<float>(const int count, const float* top_diff, const float* bottom_data, float* bottom_diff);
template bool clBNLLLayerBackward<double>(const int count, const double* top_diff, const double* bottom_data, double* bottom_diff);

} // namespace OpenCL

template<typename Dtype>
void BNLLLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = (*top)[0]->mutable_gpu_data();
	const int count = bottom[0]->count();
	// NOLINT_NEXT_LINE(whitespace/operators)
	/*
	BNLLForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, top_data);
	CUDA_POST_KERNEL_CHECK;
	*/
	BOOL_CHECK( caffe::OpenCL::clBNLLLayerForward(count, bottom_data, top_data) );
}


template<typename Dtype>
void BNLLLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
	if (propagate_down[0]) {
		const Dtype* bottom_data = (*bottom)[0]->gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
		const int count = (*bottom)[0]->count();
		// NOLINT_NEXT_LINE(whitespace/operators)
		/*
		BNLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, bottom_data, bottom_diff);
		CUDA_POST_KERNEL_CHECK;
		*/
		BOOL_CHECK( caffe::OpenCL::clBNLLLayerBackward(count, top_diff, bottom_data, bottom_diff) );
	}
}

#endif // USE_OPENCL


#ifdef CPU_ONLY
STUB_GPU(BNLLLayer);
#endif

INSTANTIATE_CLASS(BNLLLayer);
REGISTER_LAYER_CLASS(BNLL);

}  // namespace caffe
