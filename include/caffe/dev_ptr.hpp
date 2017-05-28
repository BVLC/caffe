#ifndef CAFFE_DEVPTR_HPP_
#define CAFFE_DEVPTR_HPP_

#include <cstddef>
#include "caffe/definitions.hpp"

namespace caffe {

/*
 *  dev_ptr class should be constructed similarly to shared_ptr of Boost.
 *  (but excluding the smart pointer features, so memory management
 *  is explicit, and only support types (float, void, double, char, int_tp, ...))
 *  It should be possible to use this object just like pointers,
 *  independently of the backend and device used.
 *  Dereferencing (although inefficient on some backends) should also
 *  be supported.
 * */
template<typename Type> class dev_ptr {
 public:
  virtual Type* get() = 0;
  virtual std::size_t off() = 0;

  // Comparators
  virtual inline bool operator==(dev_ptr const &other) = 0;
  virtual inline bool operator!=(dev_ptr const &other) = 0;
  virtual inline bool operator>(dev_ptr const &other) = 0;
  virtual inline bool operator<(dev_ptr const &other) = 0;
  virtual inline bool operator<=(dev_ptr const &other) = 0;
  virtual inline bool operator>=(dev_ptr const &other) = 0;
};

}  // namespace caffe

#endif /* CAFFE_DEVPTR_HPP_ */
