#ifndef CAFFE_BACKEND_DEVICE_PROGRAM_HPP_
#define CAFFE_BACKEND_DEVICE_PROGRAM_HPP_

namespace caffe {

class device_program {

public:
  virtual void compile() = 0;

};


#endif  // CAFFE_BACKEND_DEVICE_PROGRAM_HPP_
