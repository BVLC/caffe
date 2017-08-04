#ifndef CAFFE_BACKEND_DEV_PTR_HPP_
#define CAFFE_BACKEND_DEV_PTR_HPP_

#include <cstddef>
#include "caffe/definitions.hpp"

namespace caffe {

template<typename Mtype> class dev_ptr {
 public:
  virtual void increment(int_tp val) = 0;
  virtual void decrement(int_tp val) = 0;
};

}  // namespace caffe

#endif  // CAFFE_BACKEND_DEV_PTR_HPP_
