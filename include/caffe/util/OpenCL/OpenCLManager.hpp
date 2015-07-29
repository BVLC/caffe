#ifndef __OPENCL_MANAGER_HPP__
#define __OPENCL_MANAGER_HPP__

#include <CL/cl.hpp>
#include <tr1/memory>

#include <caffe/util/OpenCL/OpenCLPlatform.hpp>

#include <vector>

namespace caffe {

typedef std::vector<std::tr1::shared_ptr<caffe::OpenCLPlatform> > Platforms;
typedef Platforms::iterator PlatformIter;
class OpenCLManager {
 public:
    static bool Init();
    static void Print();
    static int GetNumPlatforms();
    static std::tr1::shared_ptr<OpenCLPlatform> CurrentPlatform();
    static void SetDeviceId(int device_id);
 private:
    void Check();
    OpenCLManager();
    ~OpenCLManager();
    std::tr1::shared_ptr<OpenCLPlatform> getPlatform(unsigned int idx);
    bool Query();
    int current_platform_index_;
    Platforms platforms_;
    int device_id_;
    bool initialized_;
    // singleton instance.
    static OpenCLManager instance_;
};
}  // namespace caffe

#endif  // __OPENCL_MANAGER_HPP__
