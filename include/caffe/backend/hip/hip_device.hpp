#ifndef CAFFE_BACKEND_HIP_HIP_DEVICE_HPP_
#define CAFFE_BACKEND_HIP_HIP_DEVICE_HPP_

#include "caffe/backend/opencl/caffe_opencl.hpp"
#include "caffe/backend/device.hpp"

namespace caffe {

#ifdef USE_HIP

class HipDevice : public Device {
 public:
 protected:
 private:
};


#endif  // USE_HIP

}

#endif  // CAFFE_BACKEND_HIP_HIP_DEVICE_HPP_
