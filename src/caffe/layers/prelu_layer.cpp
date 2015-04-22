#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

#if defined(USE_OPENCL)
#include <caffe/util/OpenCL/OpenCLDevice.hpp>
#include <caffe/util/OpenCL/prelu_layer.hpp>
#endif

namespace caffe {

template <typename Dtype>
void PReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  PReLUParameter prelu_param = this->layer_param().prelu_param();
  int channels = bottom[0]->channels();
  channel_shared_ = prelu_param.channel_shared();
  if (this->blobs_.size() > 0) {
    DLOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    if (channel_shared_) {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(0)));
    } else {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, channels)));
    }
    shared_ptr<Filler<Dtype> > filler;
    if (prelu_param.has_filler()) {
      filler.reset(GetFiller<Dtype>(prelu_param.filler()));
    } else {
      FillerParameter filler_param;
      filler_param.set_type("constant");
      filler_param.set_value(0.25);
      filler.reset(GetFiller<Dtype>(filler_param));
    }
    filler->Fill(this->blobs_[0].get());
  }
  if (channel_shared_) {
    CHECK_EQ(this->blobs_[0]->count(), 1)
        << "Negative slope size is inconsistent with prototxt config";
  } else {
    CHECK_EQ(this->blobs_[0]->count(), channels)
        << "Negative slope size is inconsistent with prototxt config";
  }

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  multiplier_.Reshape(vector<int>(1, bottom[0]->count() / bottom[0]->num()));
  caffe_set(multiplier_.count(), Dtype(1), multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void PReLULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
  if (bottom[0] == top[0]) {
    // For in-place computation
    bottom_memory_.ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void PReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const Dtype* slope_data = this->blobs_[0]->cpu_data();

  // For in-place computation
  if (bottom[0] == top[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_cpu_data());
  }

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? channels : 1;
  for (int i = 0; i < count; ++i) {
    int c = (i / dim) % channels / div_factor;
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + slope_data[c] * std::min(bottom_data[i], Dtype(0));
  }
}

template <typename Dtype>
void PReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* slope_data = this->blobs_[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.cpu_data();
  }

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? channels : 1;

  // Propagte to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[0]) {
    Dtype* slope_diff = this->blobs_[0]->mutable_cpu_diff();
    caffe_set(this->blobs_[0]->count(), Dtype(0), slope_diff);
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      slope_diff[c] += top_diff[i] * bottom_data[i] * (bottom_data[i] <= 0);
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + slope_data[c] * (bottom_data[i] <= 0));
    }
  }
}

#if defined(USE_OPENCL)

namespace OpenCL {

template<typename T>
bool clPReLUForward(const int n, const int channels, const int dim,
    const T* in, T* out, const T* slope_data,
    const int div_factor) {
  OpenCLDevice& current_device = OpenCLManager::CurrentPlatform().CurrentDevice();
	std::string kernel_name = clGetKernelName<T>("PReLUForward");
  cl_command_queue* queue = current_device.getQueue();
  if (!queue) {
    LOG(ERROR) << current_device.name()
               << "> failed to get OpenCL command queue";
		return false;
	}

  cl_kernel* kernel = current_device.getKernel(kernel_name);
  if (kernel == NULL) {
		return false;
	}

	CL_SET_KERNEL_ARG
  CL_SET_TYPE_KERNEL_ARG(int, n, kernel)
  CL_SET_TYPE_KERNEL_ARG(int, channels, kernel)
  CL_SET_TYPE_KERNEL_ARG(int, dim, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&in, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&out, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&slope_data, kernel)
  CL_SET_TYPE_KERNEL_ARG(int, div_factor, kernel)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

  err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global,
                               &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
    LOG(ERROR) << "Failed to enqueue kernel '"
               << kernel_name.c_str() << "' on GPU "<< current_device.name()
               << " : " << caffe::OpenCL::what(err);
		return false;
	}
	//clFinish(*queue);
  DLOG(INFO) << "kernel '" << kernel_name.c_str()
             << "' executed on GPU " << current_device.name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool clPReLUForward<float>(const int n, const int channels, const int dim,
    const float* in, float* out, const float* slope_data,
    const int div_factor);
template bool clPReLUForward<double>(const int n, const int channels, const int dim,
    const double* in, double* out, const double* slope_data,
    const int div_factor);

template<typename T>
bool clPReLUBackward(const int n, const int channels, const int dim,
    const T* in_diff, const T* in_data, T* out_diff,
    const T* slope_data, const int div_factor) {
  OpenCLDevice& current_device = OpenCLManager::CurrentPlatform().CurrentDevice();
	std::string kernel_name = clGetKernelName<T>("PReLUBackward");
  cl_command_queue* queue = current_device.getQueue();
  if (!queue) {
    LOG(ERROR) << current_device.name() << "> failed to get OpenCL command queue";
		return false;
	}

  cl_kernel* kernel = current_device.getKernel(kernel_name);
	if ( kernel == NULL ) {
		return false;
	}

	CL_SET_KERNEL_ARG
  CL_SET_TYPE_KERNEL_ARG(int, n, kernel)
  CL_SET_TYPE_KERNEL_ARG(int, channels, kernel)
  CL_SET_TYPE_KERNEL_ARG(int, dim, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&in_diff, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&in_data, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&out_diff, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&slope_data, kernel)
  CL_SET_TYPE_KERNEL_ARG(int, div_factor, kernel)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

  err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL,
                               &global, &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
    LOG(ERROR) << "Failed to enqueue kernel '"
               << kernel_name << "' on GPU "
               << current_device.name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	//clFinish(*queue);
  DLOG(INFO) << "kernel '" << kernel_name
             << "' executed on GPU " << current_device.name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool clPReLUBackward<float>(const int n, const int channels, const int dim,
    const float* in_diff, const float* in_data, float* out_diff,
    const float* slope_data, const int div_factor);
template bool clPReLUBackward<double>(const int n, const int channels, const int dim,
    const double* in_diff, const double* in_data, double* out_diff,
    const double* slope_data, const int div_factor);

template<typename T>
bool clPReLUParamBackward(const int n, const T* in_diff,
    const T* in_data, T* out_diff) {
  OpenCLDevice& current_device = OpenCLManager::CurrentPlatform().CurrentDevice();
	std::string kernel_name = clGetKernelName<T>("PReLUParamBackward");
  cl_command_queue* queue = current_device.getQueue();
  if (!queue) {
    LOG(ERROR) << current_device.name() << "> failed to get OpenCL command queue";
		return false;
	}

  cl_kernel* kernel = current_device.getKernel(kernel_name);
  if (kernel == NULL) {
		return false;
	}

	CL_SET_KERNEL_ARG
  CL_SET_TYPE_KERNEL_ARG(int, n, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&in_diff, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&in_data, kernel)
  CL_SET_ARRAY_KERNEL_ARG(&out_diff, kernel)

	size_t global = CAFFE_GET_GLOBAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);
	size_t local  = CAFFE_GET_LOCAL_WORKITEMS(n, OPENCL_LOCAL_SIZE);

  err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, &global,
                               &local, 0, NULL, NULL);
	if ( err != CL_SUCCESS ) {
    LOG(ERROR) << "Failed to enqueue kernel '"
               << kernel_name.c_str() << "' on GPU "
               << current_device.name()<<" : "<<caffe::OpenCL::what(err);
		return false;
	}
	//clFinish(*queue);
  DLOG(INFO) << "kernel '" << kernel_name.c_str()
             << "' executed on GPU " << current_device.name();

	CL_SET_KERNEL_ARG_END

	return true;
}
template bool clPReLUParamBackward<float>(const int n, const float* in_diff, const float* in_data, float* out_diff);
template bool clPReLUParamBackward<double>(const int n, const double* in_diff, const double* in_data, double* out_diff);

} // namespace OpenCL

template <typename Dtype>
void PReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const Dtype* slope_data = this->blobs_[0]->gpu_data();
  const int div_factor = channel_shared_ ? channels : 1;

  // For in-place computation
  if (top[0] == bottom[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
  }

  /*
  // NOLINT_NEXT_LINE(whitespace/operators)
  PReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, channels, dim, bottom_data, top_data, slope_data, div_factor);
  CUDA_POST_KERNEL_CHECK;
  */
  BOOL_CHECK(caffe::OpenCL::clPReLUForward(count, channels, dim, bottom_data, top_data, slope_data, div_factor));
}

template <typename Dtype>
void PReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.gpu_data();
  }

  // Propagte to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[0]) {
    Dtype* slope_diff = this->blobs_[0]->mutable_gpu_diff();
    // slope_diff is set as 0, then accumulated over batches
    caffe_gpu_set<Dtype>(this->blobs_[0]->count(), Dtype(0), slope_diff);
    int cdim = channels * dim;
    Dtype dsum = 0.;
    for (int n = 0; n < bottom[0]->num(); ++n) {
      /*
       Dtype* temp_buff = multiplier_.mutable_gpu_diff();
      // compute element-wise diff
      // NOLINT_NEXT_LINE(whitespace/operators)
      PReLUParamBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
          CAFFE_CUDA_NUM_THREADS>>>(
          cdim, top_diff + top[0]->offset(n),
          bottom_data + bottom[0]->offset(n), multiplier_.mutable_gpu_diff());
      CUDA_POST_KERNEL_CHECK;
      */
      BOOL_CHECK(caffe::OpenCL::clPReLUParamBackward(cdim, top_diff + top[0]->offset(n), bottom_data + bottom[0]->offset(n), multiplier_.mutable_gpu_diff()));
      if (channel_shared_) {
        Dtype d;
        caffe_gpu_dot<Dtype>(channels * dim, multiplier_.gpu_diff(),
            multiplier_.gpu_data(), &d);
        dsum += d;
      } else {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
            multiplier_.gpu_diff(), multiplier_.gpu_data(), 1.,
            slope_diff);
      }
    }
    if (channel_shared_) {
      caffe_gpu_set(this->blobs_[0]->count(), Dtype(dsum), slope_diff);
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* slope_data = this->blobs_[0]->gpu_data();
    int div_factor = channel_shared_ ? channels : 1;

    /*
    // NOLINT_NEXT_LINE(whitespace/operators)
    PReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
        count, channels, dim, top_diff, bottom_data, bottom_diff, slope_data,
        div_factor);
    CUDA_POST_KERNEL_CHECK;
    */
    BOOL_CHECK(caffe::OpenCL::clPReLUBackward(count, channels, dim, top_diff, bottom_data, bottom_diff, slope_data, div_factor));
  }
}

#endif // USE_OPENCL

#if defined(CPU_ONLY) && ! defined(USE_OPENCL)
STUB_GPU(PReLULayer);
#endif

INSTANTIATE_CLASS(PReLULayer);
REGISTER_LAYER_CLASS(PReLU);

}  // namespace caffe
