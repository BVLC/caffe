#ifndef CAFFE_CUDA_DEV_PTR_HPP_
#define CAFFE_CUDA_DEV_PTR_HPP_

#include "caffe/dev_ptr.hpp"

#ifdef USE_CUDA

namespace caffe {

template<typename Type> class cuda_dev_ptr : public dev_ptr<Type> {
 public:
  explicit cuda_dev_ptr(Type* ptr);

  void* get();
  int_tp off();

 private:
  Type* raw_ptr_;
};

}  // namespace caffe

#endif  // USE_CUDA

#endif /* CAFFE_CUDA_DEV_PTR_HPP_ */
