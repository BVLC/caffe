#include <boost/date_time/posix_time/posix_time.hpp>

#include "caffe/common.hpp"
#include "caffe/device.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

Timer::Timer()
    : initted_(false), running_(false), has_run_at_least_once_(false) {
  Init();
}

Timer::~Timer() {
  if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
#ifdef USE_CUDA
    if (Caffe::GetDefaultDevice()->backend() == BACKEND_CUDA) {
      CUDA_CHECK(cudaEventDestroy(start_gpu_cuda_));
      CUDA_CHECK(cudaEventDestroy(stop_gpu_cuda_));
    }
#endif  // USE_CUDA
#ifdef USE_GREENTEA
    if (Caffe::GetDefaultDevice()->backend() == BACKEND_OpenCL) {
      clWaitForEvents(1, &start_gpu_cl_);
      clWaitForEvents(1, &stop_gpu_cl_);
      clReleaseEvent(start_gpu_cl_);
      clReleaseEvent(stop_gpu_cl_);
    }
#endif  // USE_GREENTEA
#else
    NO_GPU;
#endif
  }
}

void Timer::Start() {
  if (!running()) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
#ifdef USE_CUDA
      if (Caffe::GetDefaultDevice()->backend() == BACKEND_CUDA) {
        CUDA_CHECK(cudaEventRecord(start_gpu_cuda_, 0));
      }
#endif  // USE_CUDA
#ifdef USE_GREENTEA
      if (Caffe::GetDefaultDevice()->backend() == BACKEND_OpenCL) {
        clWaitForEvents(1, &start_gpu_cl_);
        clReleaseEvent(start_gpu_cl_);
        viennacl::ocl::context &ctx = viennacl::ocl::get_context(
            Caffe::GetDefaultDevice()->id());
        viennacl::ocl::program &program = Caffe::GetDefaultDevice()->program();
        viennacl::ocl::kernel &kernel = program.get_kernel("null_kernel_float");
        clEnqueueTask(ctx.get_queue().handle().get(), kernel.handle().get(), 0,
                        NULL, &start_gpu_cl_);
        clFinish(ctx.get_queue().handle().get());
      }
#endif
#else
      NO_GPU;
#endif
    } else {
      start_cpu_ = boost::posix_time::microsec_clock::local_time();
    }
    running_ = true;
    has_run_at_least_once_ = true;
  }
}

void Timer::Stop() {
  if (running()) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
#ifdef USE_CUDA
      if (Caffe::GetDefaultDevice()->backend() == BACKEND_CUDA) {
        CUDA_CHECK(cudaEventRecord(stop_gpu_cuda_, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_gpu_cuda_));
      }
#endif  // USE_CUDA
#ifdef USE_GREENTEA
      if (Caffe::GetDefaultDevice()->backend() == BACKEND_OpenCL) {
        clWaitForEvents(1, &stop_gpu_cl_);
        clReleaseEvent(stop_gpu_cl_);
        viennacl::ocl::context &ctx = viennacl::ocl::get_context(
            Caffe::GetDefaultDevice()->id());
        viennacl::ocl::program &program = Caffe::GetDefaultDevice()->program();
        viennacl::ocl::kernel &kernel = program.get_kernel("null_kernel_float");
        clEnqueueTask(ctx.get_queue().handle().get(), kernel.handle().get(), 0,
                        NULL, &stop_gpu_cl_);
        clFinish(ctx.get_queue().handle().get());
      }
#endif
#else
      NO_GPU;
#endif
    } else {
      stop_cpu_ = boost::posix_time::microsec_clock::local_time();
    }
    running_ = false;
  }
}

float Timer::MicroSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING)<< "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
  if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
#ifdef USE_CUDA
    if (Caffe::GetDefaultDevice()->backend() == BACKEND_CUDA) {
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_milliseconds_, start_gpu_cuda_,
              stop_gpu_cuda_));
      // Cuda only measure milliseconds
      elapsed_microseconds_ = elapsed_milliseconds_ * 1000;
    }
#endif  // USE_CUDA
#ifdef USE_GREENTEA
    if (Caffe::GetDefaultDevice()->backend() == BACKEND_OpenCL) {
      cl_ulong startTime, stopTime;
      clWaitForEvents(1, &stop_gpu_cl_);
      clGetEventProfilingInfo(start_gpu_cl_, CL_PROFILING_COMMAND_END,
          sizeof startTime, &startTime, NULL);
      clGetEventProfilingInfo(stop_gpu_cl_, CL_PROFILING_COMMAND_START,
          sizeof stopTime, &stopTime, NULL);
      double us = static_cast<double>(stopTime - startTime) / 1000.0;
      elapsed_microseconds_ = static_cast<float>(us);
    }
#endif
#else
    NO_GPU;
#endif
  } else {
    elapsed_microseconds_ = (stop_cpu_ - start_cpu_).total_microseconds();
  }
  return elapsed_microseconds_;
}

float Timer::MilliSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING)<< "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
  if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
#ifdef USE_CUDA
    if (Caffe::GetDefaultDevice()->backend() == BACKEND_CUDA) {
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_milliseconds_, start_gpu_cuda_,
              stop_gpu_cuda_));
    }
#endif  // USE_CUDA
#ifdef USE_GREENTEA
    if (Caffe::GetDefaultDevice()->backend() == BACKEND_OpenCL) {
      cl_ulong startTime = 0, stopTime = 0;
      clGetEventProfilingInfo(start_gpu_cl_, CL_PROFILING_COMMAND_END,
          sizeof startTime, &startTime, NULL);
      clGetEventProfilingInfo(stop_gpu_cl_, CL_PROFILING_COMMAND_START,
          sizeof stopTime, &stopTime, NULL);
      double ms = static_cast<double>(stopTime - startTime) / 1000000.0;
      elapsed_milliseconds_ = static_cast<float>(ms);
    }
#endif
#else
    NO_GPU;
#endif
  } else {
    elapsed_milliseconds_ = (stop_cpu_ - start_cpu_).total_milliseconds();
  }
  return elapsed_milliseconds_;
}

float Timer::Seconds() {
  return MilliSeconds() / 1000.;
}

void Timer::Init() {
  if (!initted()) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
#ifdef USE_CUDA
      if (Caffe::GetDefaultDevice()->backend() == BACKEND_CUDA) {
        CUDA_CHECK(cudaEventCreate(&start_gpu_cuda_));
        CUDA_CHECK(cudaEventCreate(&stop_gpu_cuda_));
      }
#endif  // USE_CUDA
#ifdef USE_GREENTEA
      if (Caffe::GetDefaultDevice()->backend() == BACKEND_OpenCL) {
        start_gpu_cl_ = 0;
        stop_gpu_cl_ = 0;
      }
#endif
#else
      NO_GPU;
#endif
    }
    initted_ = true;
  }
}

CPUTimer::CPUTimer() {
  this->initted_ = true;
  this->running_ = false;
  this->has_run_at_least_once_ = false;
}

void CPUTimer::Start() {
  if (!running()) {
    this->start_cpu_ = boost::posix_time::microsec_clock::local_time();
    this->running_ = true;
    this->has_run_at_least_once_ = true;
  }
}

void CPUTimer::Stop() {
  if (running()) {
    this->stop_cpu_ = boost::posix_time::microsec_clock::local_time();
    this->running_ = false;
  }
}

float CPUTimer::MilliSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING)<< "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
  this->elapsed_milliseconds_ = (this->stop_cpu_ -
      this->start_cpu_).total_milliseconds();
  return this->elapsed_milliseconds_;
}

float CPUTimer::MicroSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
  this->elapsed_microseconds_ = (this->stop_cpu_ -
      this->start_cpu_).total_microseconds();
  return this->elapsed_microseconds_;
}

}  // namespace caffe
