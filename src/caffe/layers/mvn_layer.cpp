#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

#if defined(USE_OPENCL)
#include <caffe/util/OpenCL/OpenCLDevice.hpp>
#include <caffe/util/OpenCL/mvn_layer.hpp>
#endif

namespace caffe {

template <typename Dtype>
void MVNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  mean_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      1, 1);
  variance_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      1, 1);
  temp_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  sum_multiplier_.Reshape(1, 1,
      bottom[0]->height(), bottom[0]->width());
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
}

template <typename Dtype>
void MVNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();

  int dim = bottom[0]->count() / num;
  Dtype eps = 1e-10;

  if (this->layer_param_.mvn_param().normalize_variance()) {
    // put the squares of bottom into temp_
    caffe_powx(bottom[0]->count(), bottom_data, Dtype(2),
        temp_.mutable_cpu_data());

    // computes variance using var(X) = E(X^2) - (EX)^2
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
        sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, temp_.cpu_data(),
        sum_multiplier_.cpu_data(), 0.,
        variance_.mutable_cpu_data());  // E(X^2)
    caffe_powx(mean_.count(), mean_.cpu_data(), Dtype(2),
        temp_.mutable_cpu_data());  // (EX)^2
    caffe_sub(mean_.count(), variance_.cpu_data(), temp_.cpu_data(),
        variance_.mutable_cpu_data());  // variance

    // do mean and variance normalization
    // subtract mean
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
            mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
            temp_.mutable_cpu_data());

    caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);

    // normalize variance
    caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5),
          variance_.mutable_cpu_data());

    caffe_add_scalar(variance_.count(), eps, variance_.mutable_cpu_data());

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
          temp_.mutable_cpu_data());

    caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
  } else {
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
            sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX

    // subtract mean
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
            mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
            temp_.mutable_cpu_data());

    caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);
  }
}

template <typename Dtype>
void MVNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  int num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();

  int dim = bottom[0]->count() / num;
  Dtype eps = 1e-10;

  if (this->layer_param_.mvn_param().normalize_variance()) {
    caffe_mul(temp_.count(), top_data, top_diff, bottom_diff);
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., bottom_diff,
          sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
          bottom_diff);
    caffe_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., top_diff,
            sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
            mean_.cpu_data(), sum_multiplier_.cpu_data(), 1.,
            bottom_diff);

    caffe_cpu_axpby(temp_.count(), Dtype(1), top_diff, Dtype(-1. / dim),
        bottom_diff);

    // put the squares of bottom into temp_
    caffe_powx(temp_.count(), bottom_data, Dtype(2),
        temp_.mutable_cpu_data());

    // computes variance using var(X) = E(X^2) - (EX)^2
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
        sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, temp_.cpu_data(),
        sum_multiplier_.cpu_data(), 0.,
        variance_.mutable_cpu_data());  // E(X^2)
    caffe_powx(mean_.count(), mean_.cpu_data(), Dtype(2),
        temp_.mutable_cpu_data());  // (EX)^2
    caffe_sub(mean_.count(), variance_.cpu_data(), temp_.cpu_data(),
        variance_.mutable_cpu_data());  // variance

    // normalize variance
    caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5),
          variance_.mutable_cpu_data());

    caffe_add_scalar(variance_.count(), eps, variance_.mutable_cpu_data());

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
        variance_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
        temp_.mutable_cpu_data());

    caffe_div(temp_.count(), bottom_diff, temp_.cpu_data(), bottom_diff);
  } else {
    caffe_copy(temp_.count(), top_diff, bottom_diff);
  }
}

#if defined(USE_OPENCL)

namespace OpenCL {

	template<typename T> bool clMVNLayerForwardResidual(
			const T* bottom_data, const int bottom_data_height, const int bottom_data_width,
			const T* sum_multiplier, const int sum_multiplier_width,
			const T* mean, const int mean_width,
			const T* variance, const int variance_width,
			const T eps,
			T* top_data, const int top_data_height, const int top_data_width
		    ) {

		std::string kernel_name = clGetKernelName<T>("MVNLayerForwardResidual");

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
		CL_SET_ARRAY_KERNEL_ARG(&bottom_data)
		CL_SET_TYPE_KERNEL_ARG(const int, bottom_data_height)
		CL_SET_TYPE_KERNEL_ARG(const int, bottom_data_width)
		CL_SET_ARRAY_KERNEL_ARG(&sum_multiplier)
		CL_SET_TYPE_KERNEL_ARG(const int, sum_multiplier_width)
		CL_SET_ARRAY_KERNEL_ARG(&mean)
		CL_SET_TYPE_KERNEL_ARG(const int, mean_width)
		CL_SET_ARRAY_KERNEL_ARG(&variance)
		CL_SET_TYPE_KERNEL_ARG(const int, variance_width)
		CL_SET_TYPE_KERNEL_ARG(T, eps)
		CL_SET_ARRAY_KERNEL_ARG(&top_data)
		CL_SET_TYPE_KERNEL_ARG(const int, top_data_height)
		CL_SET_TYPE_KERNEL_ARG(const int, top_data_width)

		int dim 			= 1;
		size_t global[1] = {CAFFE_GET_GLOBAL_WORKITEMS(bottom_data_width*bottom_data_height, OPENCL_LOCAL_SIZE)};
		size_t local[1]  = {CAFFE_GET_LOCAL_WORKITEMS(bottom_data_width*bottom_data_height, OPENCL_LOCAL_SIZE)};

		//int dim = 2;
		//size_t global[2] = {bottom_data_height, 256};
		//size_t local[2] = {bottom_data_height, 32};
		//LOG(ERROR)<<"global = "<<global[0]<<" x "<<global[1];
		//LOG(ERROR)<<"local  = "<<local[0]<<" x "<<local[1];

		err = clEnqueueNDRangeKernel(*queue, *kernel, dim, NULL, global, local, 0, NULL, NULL);
		if ( err != CL_SUCCESS ) {
			LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
			return false;
		}
		//clFinish(*queue);
		DLOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

		CL_SET_KERNEL_ARG_END

		return true;
	};
	template bool clMVNLayerForwardResidual<float>(const float* bottom_data, const int bottom_data_height, const int bottom_data_width, const float* sum_multiplier, const int sum_multiplier_width, const float* mean, const int mean_width, const float* variance, const int variance_width, const float eps, float* top_data, const int top_data_height, const int top_data_width);
	template bool clMVNLayerForwardResidual<double>(const double* bottom_data, const int bottom_data_height, const int bottom_data_width, const double* sum_multiplier, const int sum_multiplier_width, const double* mean, const int mean_width, const double* variance, const int variance_width, const double eps, double* top_data, const int top_data_height, const int top_data_width);




template<typename T> bool clMVNLayerForwardMV2(
		const T* data2D, const int data2D_height, const int data2D_width,
		const T* data1D, const int data1D_length,
		T* linear_term,
		T* quadratic_term
	    ) {

	std::string kernel_name = clGetKernelName<T>("MVNLayerForwardMV2");

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
	CL_SET_ARRAY_KERNEL_ARG(&data2D)
	CL_SET_TYPE_KERNEL_ARG(int, data2D_height)
	CL_SET_TYPE_KERNEL_ARG(int, data2D_width)
	CL_SET_ARRAY_KERNEL_ARG(&data1D)
	CL_SET_TYPE_KERNEL_ARG(int, data1D_length)
	CL_SET_ARRAY_KERNEL_ARG(&linear_term)
	CL_SET_ARRAY_KERNEL_ARG(&quadratic_term)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(data2D_height, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(data2D_height, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	//clFinish(*queue);
	DLOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
};
template bool clMVNLayerForwardMV2<float>(const float* data2D, const int data2D_height, const int data2D_width, const float* data1D, const int data1D_length, float* linear_term, float* quadratic_term);
template bool clMVNLayerForwardMV2<double>(const double* data2D, const int data2D_height, const int data2D_width, const double* data1D, const int data1D_length, double* linear_term, double* quadratic_term);

template<typename T> bool clMVNLayerForward(
		const T* data2D_in, const int data2D_in_height, const int data2D_in_width,
		const T* data1D_in, const int data1D_in_length,
		const T* linear_term, const int linear_term_length,
		const T* quadratic_term, const int quadratic_term_length,
		const T eps,
		T* data2D_out
	    ) {

	std::string kernel_name = clGetKernelName<T>("MVNLayerForward");

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
	CL_SET_ARRAY_KERNEL_ARG(&data2D_in)
	CL_SET_TYPE_KERNEL_ARG(int, data2D_in_height)
	CL_SET_TYPE_KERNEL_ARG(int, data2D_in_width)
	CL_SET_ARRAY_KERNEL_ARG(&data1D_in)
	CL_SET_TYPE_KERNEL_ARG(int, data1D_in_length)
	CL_SET_ARRAY_KERNEL_ARG(&linear_term)
	CL_SET_TYPE_KERNEL_ARG(int, linear_term_length)
	CL_SET_ARRAY_KERNEL_ARG(&quadratic_term)
	CL_SET_TYPE_KERNEL_ARG(int, quadratic_term_length)
	CL_SET_TYPE_KERNEL_ARG(T, eps)
	CL_SET_ARRAY_KERNEL_ARG(&data2D_out)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(data2D_in_height*data2D_in_width, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(data2D_in_height*data2D_in_width, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	//clFinish(*queue);
	DLOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
};
template bool clMVNLayerForward<float>(
		const float* data2D_in, const int data2D_in_height, const int data2D_in_width,
		const float* data1D_in, const int data1D_in_length,
		const float* linear_term, const int linear_term_length,
		const float* quadratic_term, const int quadratic_term_length,
		const float eps,
		float* data2D_out
	    );
template bool clMVNLayerForward<double>(
		const double* data2D_in, const int data2D_in_height, const int data2D_in_width,
		const double* data1D_in, const int data1D_in_length,
		const double* linear_term, const int linear_term_length,
		const double* quadratic_term, const int quadratic_term_length,
		const double eps,
		double* data2D_out
	    );

template<typename T> bool clMVNLayerForwardS2(
		const T* data2D_in, const int data2D_in_height, const int data2D_in_width,
		const T* data1D_in, const int data1D_in_length,
		T* data2D_out
	    ) {

	std::string kernel_name = clGetKernelName<T>("MVNLayerForwardS2");

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
	CL_SET_ARRAY_KERNEL_ARG(&data2D_in)
	CL_SET_TYPE_KERNEL_ARG(int, data2D_in_height)
	CL_SET_TYPE_KERNEL_ARG(int, data2D_in_width)
	CL_SET_ARRAY_KERNEL_ARG(&data1D_in)
	CL_SET_TYPE_KERNEL_ARG(int, data1D_in_length)
	CL_SET_ARRAY_KERNEL_ARG(&data2D_out)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(data2D_in_height*data2D_in_width, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(data2D_in_height*data2D_in_width, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	//clFinish(*queue);
	DLOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
};
template bool clMVNLayerForwardS2<float>(
		const float* data2D_in, const int data2D_in_height, const int data2D_in_width,
		const float* data1D_in, const int data1D_in_length,
		float* data2D_out
	    );
template bool clMVNLayerForwardS2<double>(
		const double* data2D_in, const int data2D_in_height, const int data2D_in_width,
		const double* data1D_in, const int data1D_in_length,
		double* data2D_out
	    );


template<typename T> bool clMVNLayerBackward(
		const T* data2D_in, const int data2D_in_height, const int data2D_in_width,
		const T* data1D_in, const int data1D_in_length,
		const T* linear_term, const int linear_term_length,
		const T* quadratic_term, const int quadratic_term_length,
		const T eps,
		T* data2D_out
	    ) {

	std::string kernel_name = clGetKernelName<T>("MVNLayerBackward");

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
	CL_SET_ARRAY_KERNEL_ARG(&data2D_in)
	CL_SET_TYPE_KERNEL_ARG(int, data2D_in_height)
	CL_SET_TYPE_KERNEL_ARG(int, data2D_in_width)
	CL_SET_ARRAY_KERNEL_ARG(&data1D_in)
	CL_SET_TYPE_KERNEL_ARG(int, data1D_in_length)
	CL_SET_ARRAY_KERNEL_ARG(&linear_term)
	CL_SET_TYPE_KERNEL_ARG(int, linear_term_length)
	CL_SET_ARRAY_KERNEL_ARG(&quadratic_term)
	CL_SET_TYPE_KERNEL_ARG(int, quadratic_term_length)
	CL_SET_TYPE_KERNEL_ARG(T, eps)
	CL_SET_ARRAY_KERNEL_ARG(&data2D_out)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(data2D_in_height*data2D_in_width, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(data2D_in_height*data2D_in_width, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	//clFinish(*queue);
	DLOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
};
template bool clMVNLayerBackward<float>(
		const float* data2D_in, const int data2D_in_height, const int data2D_in_width,
		const float* data1D_in, const int data1D_in_length,
		const float* linear_term, const int linear_term_length,
		const float* quadratic_term, const int quadratic_term_length,
		const float eps,
		float* data2D_out
	    );
template bool clMVNLayerBackward<double>(
		const double* data2D_in, const int data2D_in_height, const int data2D_in_width,
		const double* data1D_in, const int data1D_in_length,
		const double* linear_term, const int linear_term_length,
		const double* quadratic_term, const int quadratic_term_length,
		const double eps,
		double* data2D_out
	    );

template<typename T> bool clMVNLayerBackwardMV2(
		const T* data2D, const T* diff2D, const int data2D_height, const int data2D_width,
		const T* data1D, const int data1D_length,
		T* linear_term,
		T* quadratic_term
	    ) {

	std::string kernel_name = clGetKernelName<T>("MVNLayerBackwardMV2");

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
	CL_SET_ARRAY_KERNEL_ARG(&data2D)
	CL_SET_ARRAY_KERNEL_ARG(&diff2D)
	CL_SET_TYPE_KERNEL_ARG(int, data2D_height)
	CL_SET_TYPE_KERNEL_ARG(int, data2D_width)
	CL_SET_ARRAY_KERNEL_ARG(&data1D)
	CL_SET_TYPE_KERNEL_ARG(int, data1D_length)
	CL_SET_ARRAY_KERNEL_ARG(&linear_term)
	CL_SET_ARRAY_KERNEL_ARG(&quadratic_term)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(data2D_height, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(data2D_height, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	//clFinish(*queue);
	DLOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
};
template bool clMVNLayerBackwardMV2<float>(const float* data2D, const float* diff2D, const int data2D_height, const int data2D_width, const float* data1D, const int data1D_length, float* linear_term, float* quadratic_term);
template bool clMVNLayerBackwardMV2<double>(const double* data2D, const double* diff2D, const int data2D_height, const int data2D_width, const double* data1D, const int data1D_length, double* linear_term, double* quadratic_term);

template<typename T> bool clMVNLayerBackwardS1(
		const T* data2D_in, const T* diff2D_in, const int data2D_in_height, const int data2D_in_width,
		const T* data1D_in, const int data1D_in_length,
		const T* linear_term, const int linear_term_length,
		const T* quadratic_term, const int quadratic_term_length,
		T* data2D_out
	    ) {

	std::string kernel_name = clGetKernelName<T>("MVNLayerBackwardS1");

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
	CL_SET_ARRAY_KERNEL_ARG(&data2D_in)
	CL_SET_ARRAY_KERNEL_ARG(&diff2D_in)
	CL_SET_TYPE_KERNEL_ARG(int, data2D_in_height)
	CL_SET_TYPE_KERNEL_ARG(int, data2D_in_width)
	CL_SET_ARRAY_KERNEL_ARG(&data1D_in)
	CL_SET_TYPE_KERNEL_ARG(int, data1D_in_length)
	CL_SET_ARRAY_KERNEL_ARG(&linear_term)
	CL_SET_TYPE_KERNEL_ARG(int, linear_term_length)
	CL_SET_ARRAY_KERNEL_ARG(&quadratic_term)
	CL_SET_TYPE_KERNEL_ARG(int, quadratic_term_length)
	CL_SET_ARRAY_KERNEL_ARG(&data2D_out)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(data2D_in_height*data2D_in_width, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(data2D_in_height*data2D_in_width, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	//clFinish(*queue);
	DLOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
};
template bool clMVNLayerBackwardS1<float>(
		const float* data2D_in, const float* diff2D_in, const int data2D_in_height, const int data2D_in_width,
		const float* data1D_in, const int data1D_in_length,
		const float* linear_term, const int linear_term_length,
		const float* quadratic_term, const int quadratic_term_length,
		float* data2D_out
	    );
template bool clMVNLayerBackwardS1<double>(
		const double* data2D_in, const double* diff2D_in, const int data2D_in_height, const int data2D_in_width,
		const double* data1D_in, const int data1D_in_length,
		const double* linear_term, const int linear_term_length,
		const double* quadratic_term, const int quadratic_term_length,
		double* data2D_out
	    );

template<typename T> bool clMVNLayerForward_perf(
		const T* A2D_top, const T* A2D_top_diff, const int top_height, const int top_width,
		const T* A2D_bottom, const T* A2D_bottom_diff, const int bottom_height, const int bottom_width,
		const T* A1D_sum_multiplier, const T* A1D_buffer, const int sum_multiplier_length,
		const T eps,
		T* data2D_out
	    ) {

	std::string kernel_name = clGetKernelName<T>("MVNLayerForward_perf");

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
	CL_SET_ARRAY_KERNEL_ARG(&A2D_top)
	CL_SET_ARRAY_KERNEL_ARG(&A2D_top_diff)
	CL_SET_TYPE_KERNEL_ARG(int, top_height)
	CL_SET_TYPE_KERNEL_ARG(int, top_width)
	CL_SET_ARRAY_KERNEL_ARG(&A2D_bottom)
	CL_SET_ARRAY_KERNEL_ARG(&A2D_bottom_diff)
	CL_SET_TYPE_KERNEL_ARG(int, bottom_height)
	CL_SET_TYPE_KERNEL_ARG(int, bottom_width)
	CL_SET_ARRAY_KERNEL_ARG(&A1D_sum_multiplier)
	CL_SET_ARRAY_KERNEL_ARG(&A1D_buffer)
	CL_SET_TYPE_KERNEL_ARG(int, sum_multiplier_length)
	CL_SET_TYPE_KERNEL_ARG(T, eps)
	CL_SET_ARRAY_KERNEL_ARG(&data2D_out)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(top_height*top_width, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(top_height*top_width, OPENCL_LOCAL_SIZE);

	std::string function = __func__;
	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	//clFinish(*queue);
	DLOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
};
template bool clMVNLayerForward_perf<float>(
		const float* A2D_top, const float* A2D_top_diff, const int top_height, const int top_width,
		const float* A2D_bottom, const float* A2D_bottom_diff, const int bottom_height, const int bottom_width,
		const float* A1D_sum_multiplier, const float* A1D_buffer, const int sum_multiplier_length,
		const float eps,
		float* data2D_out
	    );
template bool clMVNLayerForward_perf<double>(
		const double* A2D_top, const double* A2D_top_diff, const int top_height, const int top_width,
		const double* A2D_bottom, const double* A2D_bottom_diff, const int bottom_height, const int bottom_width,
		const double* A1D_sum_multiplier, const double* A1D_buffer, const int sum_multiplier_length,
		const double eps,
		double* data2D_out
	    );

template<typename T> bool clMVNLayerBackward_perf(
		const T* A2D_top, const T* A2D_top_diff, const int top_height, const int top_width,
		const T* A2D_bottom, const T* A2D_bottom_diff, const int bottom_height, const int bottom_width,
		const T* A1D_sum_multiplier, const T* A1D_buffer, const int sum_multiplier_length,
		const T eps,
		T* data2D_out
	    ) {

	std::string kernel_name = clGetKernelName<T>("MVNLayerBackward_perf");

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
	CL_SET_ARRAY_KERNEL_ARG(&A2D_top)
	CL_SET_ARRAY_KERNEL_ARG(&A2D_top_diff)
	CL_SET_TYPE_KERNEL_ARG(int, top_height)
	CL_SET_TYPE_KERNEL_ARG(int, top_width)
	CL_SET_ARRAY_KERNEL_ARG(&A2D_bottom)
	CL_SET_ARRAY_KERNEL_ARG(&A2D_bottom_diff)
	CL_SET_TYPE_KERNEL_ARG(int, bottom_height)
	CL_SET_TYPE_KERNEL_ARG(int, bottom_width)
	CL_SET_ARRAY_KERNEL_ARG(&A1D_sum_multiplier)
	CL_SET_ARRAY_KERNEL_ARG(&A1D_buffer)
	CL_SET_TYPE_KERNEL_ARG(int, sum_multiplier_length)
	CL_SET_TYPE_KERNEL_ARG(T, eps)
	CL_SET_ARRAY_KERNEL_ARG(&data2D_out)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(top_height*top_width, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(top_height*top_width, OPENCL_LOCAL_SIZE);

	err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
		LOG(ERROR) << "Failed to enqueue kernel '"<<kernel_name.c_str()<<"' on GPU "<<gpu->name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	//clFinish(*queue);
	DLOG(INFO) << "kernel '"<<kernel_name.c_str()<<"' executed on GPU "<<gpu->name();

	CL_SET_KERNEL_ARG_END

	return true;
};
template bool clMVNLayerBackward_perf<float>(
		const float* A2D_top, const float* A2D_top_diff, const int top_height, const int top_width,
		const float* A2D_bottom, const float* A2D_bottom_diff, const int bottom_height, const int bottom_width,
		const float* A1D_sum_multiplier, const float* A1D_buffer, const int sum_multiplier_length,
		const float eps,
		float* data2D_out
	    );
template bool clMVNLayerBackward_perf<double>(
		const double* A2D_top, const double* A2D_top_diff, const int top_height, const int top_width,
		const double* A2D_bottom, const double* A2D_bottom_diff, const int bottom_height, const int bottom_width,
		const double* A1D_sum_multiplier, const double* A1D_buffer, const int sum_multiplier_length,
		const double eps,
		double* data2D_out
	    );


} // namespace OpenCL


template<typename Dtype>
void MVNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	//const Dtype* bottom_diff = bottom[0]->gpu_diff();

	Dtype* top_data = (top)[0]->mutable_gpu_data();
	//const Dtype* top_diff = (top)[0]->gpu_diff();

	int num_images	 = bottom[0]->num();
	int num_channels = bottom[0]->channels();
	int num_pixels	 = bottom[0]->count();

	//LOG(ERROR)<<"num_images   = "<<num_images;
	//LOG(ERROR)<<"num_channels = "<<num_channels;
	//LOG(ERROR)<<"num_pixels   = "<<num_pixels;

	int num_parts;
	if (this->layer_param_.mvn_param().across_channels()) {
		num_parts = num_images;																// number of images
	} else {
		num_parts = num_images * num_channels;								// number of frames
	}
	int num_ppp = num_pixels / num_parts;										//	number of pixels for each part

	//LOG(ERROR)<<"num_parts    = "<<num_parts;
	//LOG(ERROR)<<"num_ppp      = "<<num_ppp;

	if (this->layer_param_.mvn_param().normalize_variance()) {
		Dtype eps = 1e-10;

		// 1D array [num_pixels]
		// this step computes the square of all pixel values in bottom layer
		caffe_gpu_powx(num_pixels, bottom_data, Dtype(2), temp_.mutable_gpu_data());
	  //snap2D("bottom", bottom[0]->cpu_data(),num_ppp, num_parts);
	  //snap2D("squared", temp_.cpu_data(),num_ppp, num_parts);

		// 2D array [num_parts x num_ppp]
		// this step computes the dot product between each part of bottom layer and sum_multiplier and normalizes by number of pixels per part
		caffe_gpu_gemv<Dtype>(CblasNoTrans, num_parts, num_ppp, 1. / num_ppp, bottom_data, sum_multiplier_.gpu_data(), 0., mean_.mutable_gpu_data());
	  //snap2D("mean", mean_.cpu_data(),1, num_parts);

		// 2D array [num_parts x num_ppp]
		// this step computes the dot product between each part of bottom layer (squared) and sum_multiplier and normalizes by number of pixels per part
		caffe_gpu_gemv<Dtype>(CblasNoTrans, num_parts, num_ppp, 1. / num_ppp, temp_.gpu_data(), sum_multiplier_.gpu_data(), 0., variance_.mutable_gpu_data());  // E(X^2)
	  //snap2D("variance", variance_.cpu_data(),1, num_parts);

		/*
		// 1D array [num_parts]
		// this step computes the squre of the mean
		caffe_gpu_powx(mean_.count(), mean_.gpu_data(), Dtype(2), temp_.mutable_gpu_data());

		// 1D array [num_parts]
		// this step computes the difference between variance and quare of mean vector
		caffe_gpu_sub(mean_.count(), variance_.gpu_data(), temp_.gpu_data(), variance_.mutable_gpu_data());

		// [num_parts x num_ppp] = [num_parts x 1] * [1 x num_ppp]
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_parts, num_ppp, 1, -1., mean_.gpu_data(), sum_multiplier_.gpu_data(), 0., temp_.mutable_gpu_data());

		// 1D array [num_pixels]
		caffe_gpu_add(temp_.count(), bottom_data, temp_.gpu_data(), top_data);

		// 1D array [num_parts]
		// square root of variance
		caffe_gpu_powx(variance_.count(), variance_.gpu_data(), Dtype(0.5), variance_.mutable_gpu_data());

		// 1D array [num_parts]
		// add small eps value to variance
		caffe_gpu_add_scalar(variance_.count(), eps, variance_.mutable_gpu_data());

		// [num_parts x num_ppp] = [num_parts x 1] * [1 x num_ppp]
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_parts, num_ppp, 1, 1., variance_.gpu_data(), sum_multiplier_.gpu_data(), 0., temp_.mutable_gpu_data());

		// [num_pixels]
		caffe_gpu_div(temp_.count(), top_data, temp_.gpu_data(), top_data);
		*/

		caffe::OpenCL::clMVNLayerForwardResidual(
				bottom_data, num_parts, num_ppp,
				sum_multiplier_.gpu_data(), num_ppp,
				mean_.gpu_data(), num_parts,
			  variance_.gpu_data(), num_parts,
			  eps,
				top_data, num_parts, num_ppp
		);

		//caffe::OpenCL::clMVNLayerForwardMV2(bottom_data, num_parts, num_ppp, sum_multiplier_.gpu_data(), num_ppp, (Dtype*) mean_.mutable_gpu_data(), (Dtype*) variance_.mutable_gpu_data());
		//caffe::OpenCL::clMVNLayerForward(bottom_data, num_parts, num_ppp, sum_multiplier_.gpu_data(), num_ppp, (Dtype*) mean_.mutable_gpu_data(), num_parts, (Dtype*) variance_.mutable_gpu_data(), num_parts, eps, top_data);

		//caffe::OpenCL::clMVNLayerForward_perf(top_data, top_diff, num_parts, num_ppp, bottom_data, bottom_diff, num_parts, num_ppp, sum_multiplier_.gpu_data(), (Dtype*) temp_.mutable_gpu_data(), num_ppp, eps, top_data);

	} else {

		/*
		caffe_gpu_gemv<Dtype>(CblasNoTrans, num_parts, num_ppp, 1. / num_ppp, bottom_data, sum_multiplier_.gpu_data(), 0., mean_.mutable_gpu_data());  // EX
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_parts, num_ppp, 1, -1., mean_.gpu_data(), sum_multiplier_.gpu_data(), 0., temp_.mutable_gpu_data());
		caffe_gpu_add(temp_.count(), bottom_data, temp_.gpu_data(), top_data);
		*/
		caffe::OpenCL::clMVNLayerForwardS2(bottom_data, num_parts, num_ppp, sum_multiplier_.gpu_data(), num_ppp, top_data);

	}
}

template<typename Dtype>
void MVNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype* top_diff = top[0]->gpu_diff();
	const Dtype* top_data = top[0]->gpu_data();
	const Dtype* bottom_data = (bottom)[0]->gpu_data();
	Dtype* bottom_diff = (bottom)[0]->mutable_gpu_diff();

	int num;
	if (this->layer_param_.mvn_param().across_channels())
		num = (bottom)[0]->num();
	else
		num = (bottom)[0]->num() * (bottom)[0]->channels();

	int dim = (bottom)[0]->count() / num;

	Dtype eps = 1e-10;

	if (this->layer_param_.mvn_param().normalize_variance()) {

		/*
		caffe_gpu_mul(temp_.count(), top_data, top_diff, bottom_diff);
		caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., bottom_diff, sum_multiplier_.gpu_data(), 0., mean_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1., mean_.gpu_data(), sum_multiplier_.gpu_data(), 0., bottom_diff);
		caffe_gpu_mul(temp_.count(), top_data, bottom_diff, bottom_diff);
		caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., top_diff, sum_multiplier_.gpu_data(), 0., mean_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1., mean_.gpu_data(), sum_multiplier_.gpu_data(), 1., bottom_diff);
		caffe_gpu_axpby(temp_.count(), Dtype(1), top_diff, Dtype(-1. / dim), bottom_diff);
		caffe_gpu_powx(temp_.count(), bottom_data, Dtype(2), temp_.mutable_gpu_data());
		caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data, sum_multiplier_.gpu_data(), 0., mean_.mutable_gpu_data());  // EX
		caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, temp_.gpu_data(), sum_multiplier_.gpu_data(), 0., variance_.mutable_gpu_data());  // E(X^2)
		caffe_gpu_powx(mean_.count(), mean_.gpu_data(), Dtype(2), temp_.mutable_gpu_data());  // (EX)^2
		caffe_gpu_sub(mean_.count(), variance_.gpu_data(), temp_.gpu_data(), variance_.mutable_gpu_data());  // variance
		caffe_gpu_powx(variance_.count(), variance_.gpu_data(), Dtype(0.5), variance_.mutable_gpu_data());
		caffe_gpu_add_scalar(variance_.count(), eps, variance_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1., variance_.gpu_data(), sum_multiplier_.gpu_data(), 0., temp_.mutable_gpu_data());
		caffe_gpu_div(temp_.count(), bottom_diff, temp_.gpu_data(), bottom_diff);
		*/

		/*
		caffe::OpenCL::clMVNLayerBackwardMV2(top_data, top_diff, num, dim, sum_multiplier_.gpu_data(), dim, (Dtype*) mean_.mutable_gpu_data(), (Dtype*) variance_.mutable_gpu_data());
		caffe::OpenCL::clMVNLayerBackwardS1(top_data, top_diff, num, dim, sum_multiplier_.gpu_data(), dim, (Dtype*) mean_.mutable_gpu_data(), num, (Dtype*) variance_.mutable_gpu_data(), num, bottom_diff);
		caffe::OpenCL::clMVNLayerForwardMV2(bottom_data, num, dim, sum_multiplier_.gpu_data(), dim, (Dtype*) mean_.mutable_gpu_data(), (Dtype*) variance_.mutable_gpu_data());
		caffe::OpenCL::clMVNLayerBackward(bottom_diff, num, dim, sum_multiplier_.gpu_data(), dim, (Dtype*) mean_.mutable_gpu_data(), num, (Dtype*) variance_.mutable_gpu_data(), num, eps, bottom_diff);
		*/

		caffe::OpenCL::clMVNLayerBackward_perf(top_data, top_diff, num, dim, bottom_data, bottom_diff, num, dim, sum_multiplier_.gpu_data(), (Dtype*) temp_.mutable_gpu_data(), dim, eps, bottom_diff);
	} else {
		caffe_copy(temp_.count(), top_diff, bottom_diff);
	}
}

#endif // USE_OPENCL

#if defined(CPU_ONLY) && ! defined(USE_OPENCL)
STUB_GPU(MVNLayer);
#endif

INSTANTIATE_CLASS(MVNLayer);
REGISTER_LAYER_CLASS(MVN);

}  // namespace caffe
