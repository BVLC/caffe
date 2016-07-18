#ifndef CAFFE_UTIL_CL_STATE_HPP
#define CAFFE_UTIL_CL_STATE_HPP
#include <CL/cl.h>

#include <string>
#include <utility>
#include <vector>

namespace caffe {

/**
 * @brief Virtual addressing with Opencl memory objects and their offsets. After allocating
 *        Opencl memory, the memory object and the memory size are mapped into a virtual
 *        address and the virtual address is returned. The memory object and its offset can
 *        easily be determined by passing a single address in order to enable the use of Caffe's
 *        pointer offsetting just as the approach in CUDA path. By this mechanism, many redundant
 *        code caused by the difference between Opencl and CUDA path can be reduced.
 */

template<typename T>
struct ClMemOff {
  cl_mem memobj;
  size_t offset;  // offset in elements
};

class ClState {
 public:
  ClState();
  ~ClState();

  void* create_buffer(int dev_id, cl_mem_flags flags, size_t size,
                      void* host_ptr, cl_int *errcode);
  void destroy_buffer(void* buffer);
  size_t get_buffer_size(const void* buffer);
  ClMemOff<uint8_t> get_buffer_mem(const void* ptr);
  int get_mem_dev(cl_mem memobj);

  template<typename T>
  ClMemOff<T> get_buffer_mem(const T* ptr) {
    ClMemOff<uint8_t> m = get_buffer_mem(static_cast<const void*>(ptr));
    ClMemOff<T> mT = {m.memobj, m.offset / sizeof (T)};
    return mT;
  }

  cl_mem create_subbuffer(const void* ptr, size_t offset, cl_mem_flags flags);
  template <typename T> cl_mem create_subbuffer(T* ptr, int offset) {
    return create_subbuffer(ptr, offset * sizeof(T), CL_MEM_READ_WRITE);
  }
  template <typename T> cl_mem create_subbuffer(const T* ptr, int offset) {
    return create_subbuffer(ptr, offset * sizeof(T), CL_MEM_READ_ONLY);
  }

 private:
  ClState(const ClState&);

  struct Impl;
  Impl* impl_;
};

}  // namespace caffe
#endif
