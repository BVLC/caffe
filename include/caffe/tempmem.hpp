#ifndef CAFFE_TEMPMEM_HPP_
#define CAFFE_TEMPMEM_HPP_

#include <cstdlib>

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

/**
 * @brief Holds a block of temporary memory that can be shared between
 *        different parts of caffe. The CPU and GPU memory is *not*
 *        synchronized.
 *
 * TODO(dox): more thorough description.
 */
template<typename Dtype>
class TemporaryMemory {
 public:
  TemporaryMemory():cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0) {}
  explicit TemporaryMemory(size_t size);
  ~TemporaryMemory();

  void acquire_gpu();
  void release_gpu();
  void acquire_cpu();
  void release_cpu();
  const Dtype* cpu_data() const;
  const Dtype* gpu_data() const;
  Dtype* mutable_cpu_data();
  Dtype* mutable_gpu_data();
  size_t size() { return size_; }
  void resize(size_t size);
 private:
  Dtype* cpu_ptr_;
  Dtype* gpu_ptr_;
  size_t size_;

  DISABLE_COPY_AND_ASSIGN(TemporaryMemory);
};  // class TemporaryMemory

}  // namespace caffe

#endif  // CAFFE_TEMPMEM_HPP_
