#ifndef __OPENCL_MANAGER_HPP__
#define __OPENCL_MANAGER_HPP__

#include <CL/cl.hpp>
#include <vector>
#include <caffe/util/OpenCL/OpenCLPlatform.hpp>

namespace caffe {

typedef std::vector<caffe::OpenCLPlatform> Platforms;
typedef Platforms::iterator PlatformIter;
class OpenCLManager {
 public:
  static bool Init();
  static void Print();
  static int GetNumPlatforms();
  static OpenCLPlatform& CurrentPlatform();
  static void SetDeviceId(int device_id);
 private:
  void Check();
  OpenCLManager();
  ~OpenCLManager();
  OpenCLPlatform* getPlatform(unsigned int idx);
  bool Query();
  //cl::Platform current_platform_; // cl_platform_id* 					platformPtr;
  int current_platform_index_;
  Platforms platforms_;
  int device_id_;
  bool initialized_;
  // singleton instance.
  static OpenCLManager instance_;
};

}

#endif // __OPENCL_MANAGER_HPP__
