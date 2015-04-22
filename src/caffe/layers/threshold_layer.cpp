#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

#if defined(USE_OPENCL)
#include <caffe/util/OpenCL/OpenCLDevice.hpp>
#include <caffe/util/OpenCL/threshold_layer.hpp>
#endif

namespace caffe {

template <typename Dtype>
void ThresholdLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.threshold_param().threshold();
}

template <typename Dtype>
void ThresholdLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = (bottom_data[i] > threshold_) ? Dtype(1) : Dtype(0);
  }
}

#if defined(USE_OPENCL)

namespace OpenCL {

template<typename T>
bool clThresholdForward(const int n, const T threshold, const T* in, T* out) {
  OpenCLDevice& current_device = OpenCLManager::CurrentPlatform().CurrentDevice();
	std::string kernel_name = clGetKernelName<T>("ThresholdForward");
  cl_command_queue* queue = current_device.getQueue();
  if (!queue) {
    LOG(ERROR) << current_device.name()
               << "> failed to get OpenCL command queue";
		return false;
	}

  cl_kernel* kernel = current_device.getKernel(kernel_name);
	if ( kernel == NULL ) {
		return false;
	}

	CL_SET_KERNEL_ARG
  CL_SET_TYPE_KERNEL_ARG(int, n, kernel)
  CL_SET_TYPE_KERNEL_ARG(T, threshold, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&in, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&out, kernel)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

  err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global,
                               &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
    LOG(ERROR) << "Failed to enqueue kernel '"
               <<kernel_name.c_str()<<"' on GPU "
              << current_device.name() << " : "<<caffe::OpenCL::what(err);
		return false;
	}
	//clFinish(*queue);
  DLOG(INFO) << "kernel '" << kernel_name
             << "' executed on GPU " << current_device.name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool clThresholdForward<float>(const int n, const float threshold,
                                        const float* in, float* out);
template bool clThresholdForward<double>(const int n, const double threshold,
                                          const double* in, double* out);

} // namespace OpenCL


template<typename Dtype>
void ThresholdLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = (top)[0]->mutable_gpu_data();
	const int count = bottom[0]->count();
	/*
	// NOLINT_NEXT_LINE(whitespace/operators)
	ThresholdForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, threshold_, bottom_data, top_data);
	CUDA_POST_KERNEL_CHECK;
	*/
	BOOL_CHECK( caffe::OpenCL::clThresholdForward(count, threshold_, bottom_data, top_data) );

}

#endif

#if defined(CPU_ONLY) && ! defined(USE_OPENCL)
STUB_GPU_FORWARD(ThresholdLayer, Forward);
#endif

INSTANTIATE_CLASS(ThresholdLayer);
REGISTER_LAYER_CLASS(Threshold);

}  // namespace caffe
