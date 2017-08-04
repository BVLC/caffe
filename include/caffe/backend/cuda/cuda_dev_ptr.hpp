#ifndef CAFFE_BACKEND_CUDA_CUDA_DEV_PTR_HPP_
#define CAFFE_BACKEND_CUDA_CUDA_DEV_PTR_HPP_

#include "caffe/backend/dev_ptr.hpp"

#ifdef USE_CUDA

namespace caffe {

template<typename Mtype>
class cuda_dev_ptr : public dev_ptr<Mtype> {
 public:
  explicit cuda_dev_ptr(Mtype* raw_ptr);
  template<typename Otype>
  explicit cuda_dev_ptr(dev_ptr<Otype>* other);

  Mtype* get();

  void increment(int_tp val);
  void decrement(int_tp val);

 private:
  Mtype* raw_ptr_;
};

}  // namespace caffe

#endif  // USE_CUDA

#endif  // CAFFE_BACKEND_CUDA_CUDA_DEV_PTR_HPP_
