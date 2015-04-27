#include <boost/date_time/posix_time/posix_time.hpp>

#include "caffe/common.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

Timer::Timer()
    : initted_(false),
      running_(false),
      has_run_at_least_once_(false) {
  Init();
}

Timer::~Timer() {
  if (Caffe::mode() == Caffe::GPU ) { //&& Caffe::GPU_USE_CUDA ) {
#if defined(USE_CUDA)
    CUDA_CHECK(cudaEventDestroy(start_gpu_));
    CUDA_CHECK(cudaEventDestroy(stop_gpu_));
#endif
  }
}

void Timer::Start() {

  if (!running()) {
#ifdef USE_CUDA
    if (Caffe::mode() == Caffe::GPU ) { //&& Caffe::GPU_USE_CUDA ) {
      CUDA_CHECK(cudaEventRecord(start_gpu_, 0));
    } else {
#endif
      start_cpu_ = boost::posix_time::microsec_clock::local_time();
#ifdef USE_CUDA
    }
#endif
    running_ = true;
    has_run_at_least_once_ = true;
  }
}

void Timer::Stop() {
  if (running()) {
#if defined(USE_CUDA)
    if (Caffe::mode() == Caffe::GPU ) { //&& Caffe::GPU_USE_CUDA) {
      CUDA_CHECK(cudaEventRecord(stop_gpu_, 0));
      CUDA_CHECK(cudaEventSynchronize(stop_gpu_));
    } else {
#endif
      stop_cpu_ = boost::posix_time::microsec_clock::local_time();
#if defined(USE_CUDA)
    }
#endif
    running_ = false;
  }
}


float Timer::MicroSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
#ifdef USE_CUDA
  if (Caffe::mode() == Caffe::GPU ) { //&& Caffe::GPU_USE_CUDA ) {
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_milliseconds_, start_gpu_,
                                    stop_gpu_));
    // Cuda only measure milliseconds
    elapsed_microseconds_ = elapsed_milliseconds_ * 1000;
  } else {
#endif
    elapsed_microseconds_ = (stop_cpu_ - start_cpu_).total_microseconds();
#ifdef USE_CUDA
  }
#endif
  return elapsed_microseconds_;
}

float Timer::MilliSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }

#ifdef USE_CUDA
  if ( Caffe::mode() == Caffe::GPU ) { //&& Caffe::GPU_USE_CUDA {
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_milliseconds_, start_gpu_,
                                    stop_gpu_));
  } else {
#endif
    elapsed_milliseconds_ = (stop_cpu_ - start_cpu_).total_milliseconds();
#ifdef USE_CUDA
  }
#endif
  return elapsed_milliseconds_;
}

float Timer::Seconds() {
  return MilliSeconds() / 1000.;
}

void Timer::Init() {
  if (!initted()) {
    if (Caffe::mode() == Caffe::GPU ) { //&& Caffe::GPU_USE_CUDA) {
#ifdef USE_CUDA
      CUDA_CHECK(cudaEventCreate(&start_gpu_));
      CUDA_CHECK(cudaEventCreate(&stop_gpu_));
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
    LOG(WARNING) << "Timer has never been run before reading time.";
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

bool Timer::log(std::string file, record result) {

	// check on file
	bool file_is_new = true;
	if ( std::ifstream(file.c_str()) ) {
		file_is_new = false;
	}

	// open file output stream
	std::ofstream ofs;
	ofs.open(file.c_str(), std::ios::app);
	if ( ! ofs.is_open() ) {
		LOG(ERROR)<<"failed to open file '"<<file.c_str()<<"' for writing.";
		return false;
	}

	if ( file_is_new ) {
		ofs << "\"Device\",\"SDK\",\"CAFFE_OPENCL\",\"File\",\"Function\",\"DataType\",\"Images[#]\",\"Channels[#]\",\"Width\",\"Height\",\"Time[ms]\"" << std::endl;
	}

#ifdef CPU_ONLY
	result.device = "CPU";
	result.sdk	  = "C++";
#endif

#ifdef USE_CUDA
	if ( Caffe::mode() == Caffe::CPU ) {
		result.device = "CPU";
		result.sdk	  = "C++";
	}
	if ( Caffe::mode() == Caffe::GPU ) {
		extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
		result.device = CAFFE_TEST_CUDA_PROP.name;
		result.sdk	  = "CUDA";
	}
#endif

#ifdef USE_OPENCL
	if ( Caffe::mode() == Caffe::CPU ) {
		result.device = "CPU";
    if ( OpenCLManager::CurrentPlatform().getNumDevices(CL_DEVICE_TYPE_CPU) > 0 ) {
      if ( OpenCLManager::CurrentPlatform().getDevice(CL_DEVICE_TYPE_CPU, 0) != NULL ) {
        result.device = OpenCLManager::CurrentPlatform().getDevice(CL_DEVICE_TYPE_CPU, 0)->name();
			}
		}
		result.sdk	  = "C++";
	}
	if ( Caffe::mode() == Caffe::GPU ) {
    result.device = OpenCLManager::CurrentPlatform().getDevice(CL_DEVICE_TYPE_GPU, 0)->name();
    result.sdk = OpenCLManager::CurrentPlatform().version();
	}
#endif

	ofs << result.device <<",";
	ofs << result.sdk <<",";
#ifdef USE_OPENCL
	ofs << CAFFE_OPENCL_VERSION <<",";
#else
	ofs << "0.0" <<",";
#endif
	ofs << result.file <<",";
	ofs << result.function <<",";
	ofs << result.type <<",";
	ofs << result.num_images <<",";
	ofs << result.num_channels <<",";
	ofs << result.img_width <<",";
	ofs << result.img_height <<",";
	ofs << result.time;
	ofs << std::endl;

	ofs.close();
	return true;
}

}  // namespace caffe
