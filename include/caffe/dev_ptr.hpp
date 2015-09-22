#ifndef CAFFE_DEVPTR_HPP_
#define CAFFE_DEVPTR_HPP_

namespace caffe {

/*
 *  dev_ptr class should be constructed similarly to shared_ptr of Boost.
 *  (but excluding the smart pointer features, so memory management
 *  is explicit, and only support types (float, void, double, char, int, ...))
 *  It should be possible to use this object just like pointers,
 *  independently of the backend and device used.
 *  Dereferencing (although inefficient on some backends) should also
 *  be supported.
 * */
template<typename Dtype> class dev_ptr {
 public:
  // Explicit constructors and destructors
  virtual dev_ptr();
  virtual dev_ptr(dev_ptr const& other);
  virtual ~dev_ptr();

  /* Comparators should act like comparators on normal pointers.
   /* This can depend on the offset and cl_mem object for OpenCL,
   /* and wrap around pointer comparison for CPU and CUDA.
   * */
  template<typename T> virtual inline bool operator==(dev_ptr<Dtype> const &a,
                                                      dev_ptr<T> const &b);
  template<typename T> virtual inline bool operator!=(dev_ptr<Dtype> const &a,
                                                      dev_ptr<T> const &b);
  template<typename T> virtual inline bool operator>(dev_ptr<Dtype> const &a,
                                                     dev_ptr<T> const &b);
  // TODO: Remaining cases

  // TODO: Dereference, increment, bracket and other C++ operators

  // TODO: Explicit casting template conversions
};

}  // namespace caffe


#endif /* CAFFE_DEVPTR_HPP_ */
