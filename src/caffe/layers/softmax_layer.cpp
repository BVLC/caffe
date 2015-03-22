#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#if defined(USE_OPENCL)
#include <caffe/util/OpenCL/OpenCLDevice.hpp>
#include <caffe/util/OpenCL/softmax_layer.hpp>
#endif

namespace caffe {

template <typename Dtype>
void SoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
  sum_multiplier_.Reshape(mult_dims);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  vector<int> scale_dims = bottom[0]->shape();
  scale_dims[softmax_axis_] = 1;
  scale_.Reshape(scale_dims);
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = bottom[0]->shape(softmax_axis_);
  int dim = bottom[0]->count() / outer_num_;
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  for (int i = 0; i < outer_num_; ++i) {
    // initialize scale_data to the first plane
    caffe_copy(inner_num_, bottom_data + i * dim, scale_data);
    for (int j = 0; j < channels; j++) {
      for (int k = 0; k < inner_num_; k++) {
        scale_data[k] = std::max(scale_data[k],
            bottom_data[i * dim + j * inner_num_ + k]);
      }
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_,
        1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data);
    // exponentiation
    caffe_exp<Dtype>(dim, top_data, top_data);
    // sum after exp
    caffe_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1.,
        top_data, sum_multiplier_.cpu_data(), 0., scale_data);
    // division
    for (int j = 0; j < channels; j++) {
      caffe_div(inner_num_, top_data, scale_data, top_data);
      top_data += inner_num_;
    }
  }
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = top[0]->shape(softmax_axis_);
  int dim = top[0]->count() / outer_num_;
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  for (int i = 0; i < outer_num_; ++i) {
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
    for (int k = 0; k < inner_num_; ++k) {
      scale_data[k] = caffe_cpu_strided_dot<Dtype>(channels,
          bottom_diff + i * dim + k, inner_num_,
          top_data + i * dim + k, inner_num_);
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
        -1., sum_multiplier_.cpu_data(), scale_data, 1., bottom_diff + i * dim);
  }
  // elementwise multiplication
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}

#if defined(USE_OPENCL)

namespace OpenCL {

template<typename T>
bool clkernel_channel_max(const int num, const int channels, const int spatial_dim, const T* data, T* out) {

	std::string kernel_name = clGetKernelName<T>("kernel_channel_max");

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
	CL_SET_TYPE_KERNEL_ARG(int, num)
	CL_SET_TYPE_KERNEL_ARG(int, channels)
	CL_SET_TYPE_KERNEL_ARG(int, spatial_dim)
	CL_SET_ARRAY_KERNEL_ARG(&data)
	CL_SET_ARRAY_KERNEL_ARG(&out)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(num*spatial_dim, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(num*spatial_dim, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	clFinish(*queue);
	LOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool clkernel_channel_max<float>(const int num, const int channels, const int spatial_dim, const float* data, float* out);
template bool clkernel_channel_max<double>(const int num, const int channels, const int spatial_dim, const double* data, double* out);

template<typename T>
bool clkernel_channel_subtract(const int num, const int channels, const int spatial_dim, T* data, const T* channel_max) {

	std::string kernel_name = clGetKernelName<T>("kernel_channel_subtract");

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
	CL_SET_TYPE_KERNEL_ARG(int, num)
	CL_SET_TYPE_KERNEL_ARG(int, channels)
	CL_SET_TYPE_KERNEL_ARG(int, spatial_dim)
	CL_SET_ARRAY_KERNEL_ARG(&data)
	CL_SET_ARRAY_KERNEL_ARG(&channel_max)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(num*spatial_dim, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(num*spatial_dim, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	clFinish(*queue);
	LOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool clkernel_channel_subtract<float>(const int num, const int channels, const int spatial_dim, float* data, const float* channel_max);
template bool clkernel_channel_subtract<double>(const int num, const int channels, const int spatial_dim, double* data, const double* channel_max);

template<typename T>
bool clkernel_exp(const int count, const T* data, T* out) {

	std::string kernel_name = clGetKernelName<T>("kernel_exp");

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
	CL_SET_ARRAY_KERNEL_ARG(&data)
	CL_SET_ARRAY_KERNEL_ARG(&out)

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
}
template bool clkernel_exp<float>(const int count, const float* data, float* out);
template bool clkernel_exp<double>(const int count, const double* data, double* out);

template<typename T>
bool clkernel_channel_sum(const int num, const int channels, const int spatial_dim, const T* data, T* channel_sum) {

	std::string kernel_name = clGetKernelName<T>("kernel_channel_sum");

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
	CL_SET_TYPE_KERNEL_ARG(int, num)
	CL_SET_TYPE_KERNEL_ARG(int, channels)
	CL_SET_TYPE_KERNEL_ARG(int, spatial_dim)
	CL_SET_ARRAY_KERNEL_ARG(&data)
	CL_SET_ARRAY_KERNEL_ARG(&channel_sum)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(num*spatial_dim, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(num*spatial_dim, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	clFinish(*queue);
	LOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool clkernel_channel_sum<float>(const int num, const int channels, const int spatial_dim, const float* data, float* channel_sum);
template bool clkernel_channel_sum<double>(const int num, const int channels, const int spatial_dim, const double* data, double* channel_sum);

template<typename T>
bool clkernel_channel_div(const int num, const int channels, const int spatial_dim, T* data, const T* channel_sum) {

	std::string kernel_name = clGetKernelName<T>("kernel_channel_div");

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
	CL_SET_TYPE_KERNEL_ARG(int, num)
	CL_SET_TYPE_KERNEL_ARG(int, channels)
	CL_SET_TYPE_KERNEL_ARG(int, spatial_dim)
	CL_SET_ARRAY_KERNEL_ARG(&data)
	CL_SET_ARRAY_KERNEL_ARG(&channel_sum)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(num*spatial_dim, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(num*spatial_dim, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	clFinish(*queue);
	LOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool clkernel_channel_div<float>(const int num, const int channels, const int spatial_dim, float* data, const float* channel_sum);
template bool clkernel_channel_div<double>(const int num, const int channels, const int spatial_dim, double* data, const double* channel_sum);

template<typename T>
bool clkernel_channel_dot(const int num, const int channels, const int spatial_dim, const T* data_1, const T* data_2, T* channel_dot) {

	std::string kernel_name = clGetKernelName<T>("kernel_channel_dot");

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
	CL_SET_TYPE_KERNEL_ARG(int, num)
	CL_SET_TYPE_KERNEL_ARG(int, channels)
	CL_SET_TYPE_KERNEL_ARG(int, spatial_dim)
	CL_SET_ARRAY_KERNEL_ARG(&data_1)
	CL_SET_ARRAY_KERNEL_ARG(&data_2)
	CL_SET_ARRAY_KERNEL_ARG(&channel_dot)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(num*spatial_dim, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(num*spatial_dim, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	clFinish(*queue);
	LOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool clkernel_channel_dot<float>(const int num, const int channels, const int spatial_dim, const float* data_1, const float* data_2, float* channel_dot);
template bool clkernel_channel_dot<double>(const int num, const int channels, const int spatial_dim, const double* data_1, const double* data_2, double* channel_dot);


} // namespace OpenCL


template<typename Dtype>
void SoftmaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = (top)[0]->mutable_gpu_data();
	Dtype* scale_data = scale_.mutable_gpu_data();
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int spatial_dim = bottom[0]->height() * bottom[0]->width();
	caffe_copy(bottom[0]->count(), bottom_data, top_data);

	// We need to subtract the max to avoid numerical issues, compute the exp,
	// and then normalize.
	// compute max
	// NOLINT_NEXT_LINE(whitespace/operators)

	/*
	 kernel_channel_max<Dtype><<<CAFFE_GET_BLOCKS(num * spatial_dim),
	 CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, top_data,
	 scale_data);
	 // subtract
	 // NOLINT_NEXT_LINE(whitespace/operators)
	 kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(num * spatial_dim),
	 CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, top_data,
	 scale_data);
	 // exponentiate
	 // NOLINT_NEXT_LINE(whitespace/operators)
	 kernel_exp<Dtype><<<CAFFE_GET_BLOCKS(num * channels * spatial_dim),
	 CAFFE_CUDA_NUM_THREADS>>>(num * channels * spatial_dim, top_data,
	 top_data);
	 // sum after exp
	 // NOLINT_NEXT_LINE(whitespace/operators)
	 kernel_channel_sum<Dtype><<<CAFFE_GET_BLOCKS(num * spatial_dim),
	 CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, top_data,
	 scale_data);
	 // divide
	 // NOLINT_NEXT_LINE(whitespace/operators)
	 kernel_channel_div<Dtype><<<CAFFE_GET_BLOCKS(num * spatial_dim),
	 CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, top_data,
	 scale_data);
	 */

	 BOOL_CHECK( caffe::OpenCL::clkernel_channel_max(num, channels, spatial_dim, top_data, scale_data) );
	 // subtract
	 BOOL_CHECK( caffe::OpenCL::clkernel_channel_subtract(num, channels, spatial_dim, top_data, scale_data) );
	 // exponentiate
	 BOOL_CHECK( caffe::OpenCL::clkernel_exp(num * channels * spatial_dim, top_data, top_data) );
	 // sum after exp
	 BOOL_CHECK( caffe::OpenCL::clkernel_channel_sum(num, channels, spatial_dim, top_data, scale_data) );
	 // divide
	 BOOL_CHECK( caffe::OpenCL::clkernel_channel_div(num, channels, spatial_dim, top_data, scale_data) );
}

template<typename Dtype>
void SoftmaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype* top_diff = top[0]->gpu_diff();
	const Dtype* top_data = top[0]->gpu_data();
	Dtype* bottom_diff = (bottom)[0]->mutable_gpu_diff();
	Dtype* scale_data = scale_.mutable_gpu_data();
	int num = top[0]->num();
	int channels = top[0]->channels();
	int spatial_dim = top[0]->height() * top[0]->width();
	caffe_copy(top[0]->count(), top_diff, bottom_diff);
	// Compute inner1d(top_diff, top_data) and subtract them from the bottom diff.
	// NOLINT_NEXT_LINE(whitespace/operators)
	/*
	 kernel_channel_dot<Dtype><<<CAFFE_GET_BLOCKS(num * spatial_dim),
	 CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, top_diff, top_data,
	 scale_data);
	 // NOLINT_NEXT_LINE(whitespace/operators)
	 kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(num * spatial_dim),
	 CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, bottom_diff,
	 scale_data);
	 // elementwise multiplication
	 caffe_gpu_mul<Dtype>(top[0]->count(), bottom_diff, top_data, bottom_diff);
	 */

	BOOL_CHECK( caffe::OpenCL::clkernel_channel_dot(num, channels, spatial_dim, top_diff, top_data, scale_data) );
	BOOL_CHECK( caffe::OpenCL::clkernel_channel_subtract(num, channels, spatial_dim, bottom_diff, scale_data) );
	caffe_gpu_mul<Dtype>(top[0]->count(), bottom_diff, top_data, bottom_diff);
}

#endif // USE_OPENCL

#if defined(CPU_ONLY) && ! defined(USE_OPENCL)
STUB_GPU(SoftmaxLayer);
#endif

INSTANTIATE_CLASS(SoftmaxLayer);

}  // namespace caffe
