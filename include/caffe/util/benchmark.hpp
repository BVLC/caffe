#ifndef CAFFE_UTIL_BENCHMARK_H_
#define CAFFE_UTIL_BENCHMARK_H_

#include <boost/date_time/posix_time/posix_time.hpp>

#include "caffe/util/device_alternate.hpp"

namespace caffe {

class Timer {
 public:
  Timer();
  virtual ~Timer();
  virtual void Start();
  virtual void Stop();
  virtual float MilliSeconds();
  virtual float MicroSeconds();
  virtual float Seconds();

  inline bool initted() { return initted_; }
  inline bool running() { return running_; }
  inline bool has_run_at_least_once() { return has_run_at_least_once_; }

 protected:
  void Init();

  bool initted_;
  bool running_;
  bool has_run_at_least_once_;
#ifndef CPU_ONLY
  cudaEvent_t start_gpu_;
  cudaEvent_t stop_gpu_;
#endif
  boost::posix_time::ptime start_cpu_;
  boost::posix_time::ptime stop_cpu_;
  float elapsed_milliseconds_;
  float elapsed_microseconds_;
};

class CPUTimer : public Timer {
 public:
  explicit CPUTimer();
  virtual ~CPUTimer() {}
  virtual void Start();
  virtual void Stop();
  virtual float MilliSeconds();
  virtual float MicroSeconds();
};

}  // namespace caffe

#if defined(CPU_ONLY) && ! defined(USE_OPENCL)
#define	BENCH(result, this)\
{\
	struct timeval s;\
	double bgn = 0.0;\
	if (gettimeofday(&s, 0) == 0) {\
		bgn = s.tv_sec * 1.0 + s.tv_usec * 1.e-6;\
	}\
	(this); \
	double end = 0.0;\
	if (gettimeofday(&s, 0) == 0) {\
		end = s.tv_sec * 1.0 + s.tv_usec * 1.e-6;\
	}\
	result.time 	= ((float) floor(1000*(1000*(end-bgn))))/1000;\
	result.function = __func__;\
	result.file     = __FILE__;\
	LOG(INFO) << "TIME::C++::"<<result.function.c_str()<<" = "<<result.time<<"ms";\
	caffe::Timer::log("bench.csv", result);\
}
#endif // CPU_ONLY


#if defined(USE_CUDA)

#define	BENCH(result, this)\
{\
	struct timeval s;\
	double bgn = 0.0;\
	if (gettimeofday(&s, 0) == 0) {\
		bgn = s.tv_sec * 1.0 + s.tv_usec * 1.e-6;\
	}\
	(this); \
	cudaDeviceSynchronize();\
	double end = 0.0;\
	if (gettimeofday(&s, 0) == 0) {\
		end = s.tv_sec * 1.0 + s.tv_usec * 1.e-6;\
	}\
	result.time 	= ((float) floor(1000*(1000*(end-bgn))))/1000;\
	result.function = __func__;\
	result.file     = __FILE__;\
	LOG(INFO) << "TIME::CUDA::"<<result.function.c_str()<<" = "<<result.time<<"ms";\
	caffe::Timer::log("bench.csv", result);\
}

#endif // USE_CUDA

#if defined(USE_OPENCL)

#define	BENCH(result, this)\
{\
	struct timeval s;\
	double bgn = 0.0;\
	if (gettimeofday(&s, 0) == 0) {\
		bgn = s.tv_sec * 1.0 + s.tv_usec * 1.e-6;\
	}\
	(this); \
	double end = 0.0;\
	if (gettimeofday(&s, 0) == 0) {\
		end = s.tv_sec * 1.0 + s.tv_usec * 1.e-6;\
	}\
	result.time 	= ((float) floor(1000*(1000*(end-bgn))))/1000;\
	result.function = __func__;\
	result.file     = __FILE__;\
	LOG(INFO) << "TIME::OpenCL::"<<result.function.c_str()<<" = "<<result.time<<"ms";\
	caffe::Timer::log("bench.csv", result);\
}

#endif // USE_OPENCL

#endif   // CAFFE_UTIL_BENCHMARK_H_
