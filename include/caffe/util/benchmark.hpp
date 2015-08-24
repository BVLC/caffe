#ifndef CAFFE_UTIL_BENCHMARK_H_
#define CAFFE_UTIL_BENCHMARK_H_

#include <boost/date_time/posix_time/posix_time.hpp>
#include <caffe/util/device_alternate.hpp>
#include <string>
#include <typeinfo>

struct record {
    std::string device;
    std::string sdk;
    std::string function;
    std::string file;
    std::string type;
    int num_images;
    int num_channels;
    int img_width;
    int img_height;
    float time;
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

    inline bool initted() {
      return initted_;
    }
    inline bool running() {
      return running_;
    }
    inline bool has_run_at_least_once() {
      return has_run_at_least_once_;
    }
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

class CPUTimer: public Timer {
 public:
    explicit CPUTimer();
    virtual ~CPUTimer() {
    }
    virtual void Start();
    virtual void Stop();
    virtual float MilliSeconds();
    virtual float MicroSeconds();
};

}  // namespace caffe

#ifdef USE_TIMER

#define TIME_INIT() \
struct timeval s;\
double bgn = 0.0;\
double end = 0.0;\

#define TIME_BGN() \
if (gettimeofday(&s, 0) == 0) {\
  bgn = s.tv_sec * 1.0 + s.tv_usec * 1.e-6;\
}

#define TIME_END(name) \
if (gettimeofday(&s, 0) == 0) {\
  end = s.tv_sec * 1.0 + s.tv_usec * 1.e-6;\
}\
LOG(INFO) << "TIME(" << name << ") = "\
           << (static_cast<float>(floor(1000*(1000*(end-bgn)))))/1000 << "ms";

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
caffe::Caffe::DeviceSync();\
\
double end = 0.0;\
if (gettimeofday(&s, 0) == 0) {\
  end = s.tv_sec * 1.0 + s.tv_usec * 1.e-6;\
}\
\
LOG(INFO) << "TIME(" << name << ") = "\
           << (static_cast<float>(floor(1000*(1000*(end-bgn)))))/1000 << "ms";\
\
}

#define TIMENOSYNC(name, this) {\
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
LOG(INFO) << "TIMENOSYNC(" << name << ") = "\
           << (static_cast<float>(floor(1000*(1000*(end-bgn)))))/1000 << "ms";\
}

#define FLOPS(name, ops, bytes, this) {\
\
struct timeval s;\
\
double bgn = 0.0;\
if (gettimeofday(&s, 0) == 0) {\
  bgn = s.tv_sec * 1.0 + s.tv_usec * 1.e-6;\
}\
\
(this); \
caffe::Caffe::DeviceSync();\
\
double end = 0.0;\
if (gettimeofday(&s, 0) == 0) {\
  end = s.tv_sec * 1.0 + s.tv_usec * 1.e-6;\
}\
\
  float time_s = end-bgn;\
  float time_ms = 1000*time_s;\
  float GFLOP = (uint64_t) ops/ 1000000000f;\
  float GBYTE = (uint64_t) bytes/ 1000000000f;\
  LOG(INFO) << "GBYTE = " << GBYTE << " FLOPS(" << name << ") = " \
            << (static_cast<float>(floor(1000*time_ms)))/1000 \
            << "ms " << (static_cast<float>(floor(100*GFLOP/time_s)))/100 \
            << " GFlop/s " << (static_cast<float>(floor(100*GBYTE/time_s)))/100\
            << " GByte/s"; \
\
}

#else
#define TIME(name, this) {\
(this); \
}

#define FLOPS(name, ops, this) {\
(this); \
}

#define TIME_INIT()
#define TIME_BGN()
#define TIME_END(name)

#define TIMENOSYNC(name, this) {\
(this); \
}


#endif

#if defined(CPU_ONLY) && !defined(USE_OPENCL)
#define  BENCH(result, this)\
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
  result.time   = (static_cast<float>(floor(1000*(1000*(end-bgn)))))/1000;\
  result.function = __func__;\
  result.file     = __FILE__;\
  LOG(INFO) << "TIME::C++::" << result.function.c_str()\
            << " = " << result.time << "ms";\
  caffe::Timer::log("bench.csv", result);\
}
#endif  // CPU_ONLY

#if defined(USE_OPENCL)
#define  BENCH(result, this)\
{\
  struct timeval s;\
  double bgn = 0.0;\
  if (gettimeofday(&s, 0) == 0) {\
    bgn = s.tv_sec * 1.0 + s.tv_usec * 1.e-6;\
  }\
  (this); \
  caffe::Caffe::DeviceSync();\
  double end = 0.0;\
  if (gettimeofday(&s, 0) == 0) {\
    end = s.tv_sec * 1.0 + s.tv_usec * 1.e-6;\
  }\
  result.time   = (static_cast<float>(floor(10000*(1000*(end-bgn)))))/10000;\
  result.function = __func__;\
  result.file     = __FILE__;\
  LOG(INFO) << "TIME::OpenCL::" << result.function.c_str() \
            << " = " << result.time << "ms";\
  caffe::Timer::log("bench.csv", result);\
}
#endif

#if defined(USE_CUDA)
#define BENCH(result, this)\
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
  result.time   = (static_cast<float>(floor(1000*(1000*(end-bgn)))))/1000;\
  result.function = __func__;\
  result.file     = __FILE__;\
  LOG(INFO) << "TIME::CUDA::" << result.function.c_str() \
            << " = " << result.time << "ms";\
  caffe::Timer::log("bench.csv", result);\
}
#endif

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

#define SNAP_LENGTH 32

#define iSNAPSHOT(name, array, length) \
  { \
    char buffer[1024];\
    std::cout << name << "[" << length << "] = ";\
    int limit_length  = length < SNAP_LENGTH ? length : SNAP_LENGTH;\
    for ( int i = 0; i < limit_length; i++ ) {\
      snprintf(buffer, sizeof(buffer), "%d ", (array)[i]);\
      std::cout << buffer;\
    }\
    if ( limit_length < length ) {\
      snprintf(buffer, sizeof(buffer), "%d ", (array)[length-1]);\
      std::cout << " ... " << buffer;\
    }\
    std::cout << std::endl;\
  }\

#define SNAPSHOT(name, array, length) \
  { \
    char buffer[1024];\
    std::cout << name << "[" << length << "] = ";\
    int limit_length  = length < SNAP_LENGTH ? length : SNAP_LENGTH;\
    for ( int i = 0; i < limit_length; i++ ) {\
      snprintf(buffer, sizeof(buffer), "%+6.4f ", (array)[i]);\
      std::cout << buffer;\
    }\
    if ( limit_length < length ) {\
      snprintf(buffer, sizeof(buffer), "%+6.4f ", (array)[length-1]);\
      std::cout << " ... " << buffer;\
    }\
    std::cout << std::endl;\
  }\

#define SNAPSHOT2D(name, array, height, width) \
  { \
    char buffer[1024];\
    std::cout << name \
              << "[" << height << " x " << width << "] = "  << std::endl;\
    int limit_width  = width < SNAP_LENGTH ? width : SNAP_LENGTH;\
    int limit_height = height < SNAP_LENGTH ? height : SNAP_LENGTH;\
    for ( int i = 0; i < limit_height; i++ ) {\
      for ( int j = 0; j < limit_width; j++ ) {\
        snprintf(buffer, sizeof(buffer), "%+5.1f ", (array)[i*width+j]);\
        std::cout << buffer;\
      }\
      if ( limit_width < width ) {\
        snprintf(buffer, sizeof(buffer), "%+5.1f ", (array)[i*width+width-1]);\
        std::cout << " ... " << buffer;\
      }\
      std::cout << std::endl;\
    }\
    if ( limit_height < height ) {\
      std::cout << " ..." << std::endl;\
      for ( int j = 0; j < limit_width; j++ ) {\
        snprintf(buffer, sizeof(buffer), "%+5.1f ", \
                (array)[(height-1)*width+j]);\
        std::cout << buffer;\
      }\
      if ( limit_width < width ) {\
        snprintf(buffer, sizeof(buffer), "%+5.1f ", \
                (array)[(height-1)*width+width-1]);\
        std::cout << " ... " << buffer;\
      }\
      std::cout << std::endl;\
    }\
  }\

#define DIFFSHOT2D(name, array1, array2, height, width) \
  { \
    char buffer[1024];\
    std::cout << name \
              << "[" << height << " x " << width << "] = "  << std::endl;\
    int limit_width  = width < SNAP_LENGTH ? width : SNAP_LENGTH;\
    int limit_height = height < SNAP_LENGTH ? height : SNAP_LENGTH;\
    double delta = 0.0;\
    double epsilon = 0.01;\
    for ( int i = 0; i < limit_height; i++ ) {\
      for ( int j = 0; j < limit_width; j++ ) {\
        delta = (array1)[i*width+j] - (array2)[i*width+j];\
        snprintf(buffer, sizeof(buffer), "%+5.1f ", delta);\
        if ( fabs(delta) < epsilon ) {\
          std::cout << KGRN << buffer << KNRM;\
        } else {\
          std::cout << KRED << buffer << KNRM;\
        }\
      }\
      if ( limit_width < width ) {\
        delta = (array1)[i*width+width-1] - (array2)[i*width+width-1];\
        snprintf(buffer, sizeof(buffer), "%+5.1f ", delta);\
        if ( fabs(delta) < epsilon ) {\
          std::cout << " ... " << KGRN << buffer << KNRM;\
        } else {\
          std::cout << " ... " << KRED << buffer << KNRM;\
        }\
      }\
      std::cout << std::endl;\
    }\
    if ( limit_height < height ) {\
      std::cout << " ..." << std::endl;\
      for ( int j = 0; j < limit_width; j++ ) {\
        delta = (array1)[(height-1)*width+j] - (array2)[(height-1)*width+j];\
        snprintf(buffer, sizeof(buffer), "%+5.1f ", delta);\
        if ( fabs(delta) < epsilon ) {\
          std::cout << KGRN << buffer << KNRM;\
        } else {\
          std::cout << KRED << buffer << KNRM;\
        }\
      }\
      if ( limit_width < width ) {\
        delta = (array1)[(height-1)*width+width-1] \
                - (array2)[(height-1)*width+width-1];\
        snprintf(buffer, sizeof(buffer), "%+5.1f ", delta);\
        if ( fabs(delta) < epsilon ) {\
          std::cout << " ... " << KGRN << buffer << KNRM;\
        } else {\
          std::cout << " ... " << KRED << buffer << KNRM;\
        }\
      }\
      std::cout << std::endl;\
    }\
  }\

#define DIFFSHOT(name, array1, array2, width) \
  { \
    char buffer[1024];\
    std::cout << name << "[" << width << "] = "  << std::endl;\
    int limit_width  = width < SNAP_LENGTH ? width : SNAP_LENGTH;\
    double delta = 0.0;\
    double epsilon = 0.01;\
    for ( int j = 0; j < limit_width; j++ ) {\
      delta = (array1)[j] - (array2)[j];\
      snprintf(buffer, sizeof(buffer), "%+5.1f ", delta);\
      if ( fabs(delta) < epsilon ) {\
        std::cout << KGRN << buffer << KNRM;\
      } else {\
        std::cout << KRED << buffer << KNRM;\
      }\
    }\
    if ( limit_width < width ) {\
      delta = (array1)[width-1] - (array2)[width-1];\
      snprintf(buffer, sizeof(buffer), "%+5.1f ", delta);\
      if ( fabs(delta) < epsilon ) {\
        std::cout << " ... " << KGRN << buffer << KNRM;\
      } else {\
        std::cout << " ... " << KRED << buffer << KNRM;\
      }\
    }\
    std::cout << std::endl;\
  }\

#define F1(value)\
  floor((value*10.0)+0.5)/10.0

#define F2(value)\
  floor((value*100.0)+0.5)/100.0

#define F3(value)\
  floor((value*1000.0)+0.5)/1000.0

#endif  // CAFFE_UTIL_BENCHMARK_H_
