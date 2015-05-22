#ifndef CAFFE_UTIL_BENCHMARK_H_
#define CAFFE_UTIL_BENCHMARK_H_

#include <boost/date_time/posix_time/posix_time.hpp>
#include "caffe/util/device_alternate.hpp"
#include <typeinfo>

struct record {
  std::string device;
  std::string sdk;
  std::string function;
  std::string file;
  std::string type;
  int		  num_images;
  int		  num_channels;
  int		  img_width;
  int		  img_height;
  float		  time;
};

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
  static bool log(std::string file, record result);

 protected:
  void Init();

  bool initted_;
  bool running_;
  bool has_run_at_least_once_;
#ifdef USE_CUDA
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

#ifdef USE_TIMER
#define TIME(name, this) {\
\
struct timeval s;\
\
double bgn = 0.0;\
if (gettimeofday(&s, 0) == 0) {\
  bgn = s.tv_sec * 1.0 + s.tv_usec * 1.e-6;\
}\
\
(this); \
\
double end = 0.0;\
if (gettimeofday(&s, 0) == 0) {\
  end = s.tv_sec * 1.0 + s.tv_usec * 1.e-6;\
}\
\
LOG(INFO) << "TIME("<<name<<") = "<<((float) floor(1000*(1000*(end-bgn))))/1000<<"ms"; \
\
}
#else
#define TIME(name, this) {\
(this); \
}
#endif

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


#define	BENCH(result, this)\
{\
	struct timeval s;\
	double bgn = 0.0;\
	if (gettimeofday(&s, 0) == 0) {\
		bgn = s.tv_sec * 1.0 + s.tv_usec * 1.e-6;\
	}\
	(this); \
  caffe::Caffe::DeviceSync(); \
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

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

#define SNAP_LENGTH 30
#define snap(name, array, length) \
	{ \
		char buffer[1024];\
		std::cout<<name<<"[" << length << "] = ";\
		int limit_length  = length < SNAP_LENGTH ? length : SNAP_LENGTH;\
		for( int i = 0; i < limit_length; i++ ) {\
			sprintf(buffer, "%+6.4f ", (array)[i]);\
			std::cout<<buffer;\
		}\
		if ( limit_length < length ) {\
			sprintf(buffer, "%+6.4f ", (array)[length-1]);\
			std::cout<<" ... "<<buffer;\
		}\
		std::cout<<std::endl;\
	}\

#define snap2D(name, array, width, height) \
	{ \
		char buffer[1024];\
		std::cout<<name<<"[" << width << " x " << height << "] = " <<std::endl;\
		int limit_width  = width < SNAP_LENGTH ? width : SNAP_LENGTH;\
		int limit_height = height < SNAP_LENGTH ? height : SNAP_LENGTH;\
		for( int i = 0; i < limit_height; i++ ) {\
			for( int j = 0; j < limit_width; j++ ) {\
				sprintf(buffer, "%+5.1f ", (array)[i*width+j]);\
				std::cout<<buffer;\
			}\
			if ( limit_width < width ) {\
				sprintf(buffer, "%+5.1f ", (array)[i*width+width-1]);\
				std::cout<<" ... "<<buffer;\
			}\
			std::cout<<std::endl;\
		}\
		if ( limit_height < height ) {\
			std::cout<<" ..."<<std::endl;\
			for( int j = 0; j < limit_width; j++ ) {\
				sprintf(buffer, "%+5.1f ", (array)[(height-1)*width+j]);\
				std::cout<<buffer;\
			}\
			if ( limit_width < width ) {\
				sprintf(buffer, "%+5.1f ", (array)[(height-1)*width+width-1]);\
				std::cout<<" ... "<<buffer;\
			}\
			std::cout<<std::endl;\
		}\
	}\

#define diff2D(name, array1, array2, width, height) \
  { \
    char buffer[1024];\
    std::cout<<name<<"[" << width << " x " << height << "] = " <<std::endl;\
    int limit_width  = width < SNAP_LENGTH ? width : SNAP_LENGTH;\
    int limit_height = height < SNAP_LENGTH ? height : SNAP_LENGTH;\
    double delta = 0.0;\
    double epsilon = 0.01;\
    for( int i = 0; i < limit_height; i++ ) {\
      for( int j = 0; j < limit_width; j++ ) {\
        delta = (array1)[i*width+j] - (array2)[i*width+j];\
        sprintf(buffer, "%+5.1f ", delta);\
        if ( fabs(delta) < epsilon ) {\
          std::cout<<buffer;\
        } else {\
          std::cout<<KRED<<buffer<<KNRM;\
        }\
      }\
      if ( limit_width < width ) {\
        delta = (array1)[i*width+width-1] - (array2)[i*width+width-1];\
        sprintf(buffer, "%+5.1f ", delta);\
        if ( fabs(delta) < epsilon ) {\
          std::cout<<" ... "<<buffer;\
        } else {\
          std::cout<<" ... "<<KRED<<buffer<<KNRM;\
        }\
      }\
      std::cout<<std::endl;\
    }\
    if ( limit_height < height ) {\
      std::cout<<" ..."<<std::endl;\
      for( int j = 0; j < limit_width; j++ ) {\
        delta = (array1)[(height-1)*width+j] - (array2)[(height-1)*width+j];\
        sprintf(buffer, "%+5.1f ", delta);\
        if ( fabs(delta) < epsilon ) {\
          std::cout<<buffer;\
        } else {\
          std::cout<<KRED<<buffer<<KNRM;\
        }\
      }\
      if ( limit_width < width ) {\
        delta = (array1)[(height-1)*width+width-1] - (array2)[(height-1)*width+width-1];\
        sprintf(buffer, "%+5.1f ", delta);\
        if ( fabs(delta) < epsilon ) {\
          std::cout<<" ... "<<buffer;\
        } else {\
          std::cout<<" ... "<<KRED<<buffer<<KNRM;\
        }\
      }\
      std::cout<<std::endl;\
    }\
  }\

#endif   // CAFFE_UTIL_BENCHMARK_H_
