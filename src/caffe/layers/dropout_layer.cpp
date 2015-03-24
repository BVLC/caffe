// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/benchmark.hpp"
#if defined(USE_OPENCL)
#include <caffe/util/OpenCL/OpenCLDevice.hpp>
#include <caffe/util/OpenCL/dropout_layer.hpp>
#endif

namespace caffe {

template <typename Dtype>
void DropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
}

template <typename Dtype>
void DropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  rand_vec_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  unsigned int* mask = rand_vec_.mutable_cpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    // Create random numbers
    caffe_rng_bernoulli(count, 1. - threshold_, mask);
    for (int i = 0; i < count; ++i) {
      top_data[i] = bottom_data[i] * mask[i] * scale_;
    }
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (this->phase_ == TRAIN) {
      const unsigned int* mask = rand_vec_.cpu_data();
      const int count = bottom[0]->count();
      for (int i = 0; i < count; ++i) {
        bottom_diff[i] = top_diff[i] * mask[i] * scale_;
      }
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}

#if defined(USE_OPENCL)

namespace OpenCL {

template<typename T>
bool clDropoutLayerForward(const int count, const T* bottom_data, const unsigned int* mask, const unsigned int threshold, const T scale, T* top_data) {

	std::string kernel_name = clGetKernelName<T>("DropoutForward");

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
	CL_SET_ARRAY_KERNEL_ARG(&mask)
	CL_SET_TYPE_KERNEL_ARG(const unsigned int, threshold)
	CL_SET_TYPE_KERNEL_ARG(T, scale)
	CL_SET_ARRAY_KERNEL_ARG(&top_data)

	size_t global = count;//CAFFE_GET_GLOBAL_WORKITEMS(count, OPENCL_LOCAL_SIZE);
	size_t local  = 1;//CAFFE_GET_LOCAL_WORKITEMS(count, OPENCL_LOCAL_SIZE);

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
template bool clDropoutLayerForward<float>(const int count, const float* bottom_data, const unsigned int* mask, const unsigned int threshold, const float scale, float* top_data);
template bool clDropoutLayerForward<double>(const int count, const double* bottom_data, const unsigned int* mask, const unsigned int threshold, const double scale, double* top_data);

template<typename T>
bool clDropoutLayerBackward(const int count, const T* top_diff, const unsigned int* mask, const unsigned int threshold, const T scale, T* bottom_diff) {

	std::string kernel_name = clGetKernelName<T>("DropoutBackward");

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
	CL_SET_ARRAY_KERNEL_ARG(&mask)
	CL_SET_TYPE_KERNEL_ARG(unsigned int, threshold)
	CL_SET_TYPE_KERNEL_ARG(T, scale)
	CL_SET_ARRAY_KERNEL_ARG(&bottom_diff)

	size_t global = count;//CAFFE_GET_GLOBAL_WORKITEMS(count, OPENCL_LOCAL_SIZE);
	size_t local  = 1;//CAFFE_GET_LOCAL_WORKITEMS(count, OPENCL_LOCAL_SIZE);

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
template bool clDropoutLayerBackward<float>(const int count, const float* top_diff, const unsigned int* mask, const unsigned int threshold, const float scale, float* bottom_diff);
template bool clDropoutLayerBackward<double>(const int count, const double* top_diff, const unsigned int* mask, const unsigned int threshold, const double scale, double* bottom_diff);

} // namespace OpenCL


template<typename Dtype>
void DropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    unsigned int* mask = static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
    caffe_gpu_rng_bernoulli(count, 1. - threshold_, mask);

    //caffe_gpu_rng_uniform(count, mask);

    /*
    // set thresholds
    // NOLINT_NEXT_LINE(whitespace/operators)
    DropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, mask, uint_thres_, scale_, top_data);
    CUDA_POST_KERNEL_CHECK;
    */
    BOOL_CHECK( caffe::OpenCL::clDropoutLayerForward(count, bottom_data, mask, uint_thres_, scale_, top_data));
  } else {
    caffe_copy(count, bottom_data, top_data);
  }
}

template<typename Dtype>
void DropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    if (this->phase_ == TRAIN) {
      const unsigned int* mask = static_cast<const unsigned int*>(rand_vec_.gpu_data());
      const int count = bottom[0]->count();
      /*
      // NOLINT_NEXT_LINE(whitespace/operators)
      DropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
          count, top_diff, mask, uint_thres_, scale_, bottom_diff);
      CUDA_POST_KERNEL_CHECK;
      */
			BOOL_CHECK( caffe::OpenCL::clDropoutLayerBackward(count, top_diff, mask, uint_thres_, scale_, bottom_diff) );
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}

#endif // USE_OPENCL


#if defined(CPU_ONLY) && ! defined(USE_OPENCL)
STUB_GPU(DropoutLayer);
#endif

INSTANTIATE_CLASS(DropoutLayer);
REGISTER_LAYER_CLASS(Dropout);

}  // namespace caffe
