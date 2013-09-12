#ifndef CAFFEINE_SYNCEDMEM_HPP
#define CAFFEINE_SYNCEDMEM_HPP

namespace caffeine {

class SyncedMemory {
 public:
  SyncedMemory()
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(SYNCED) {};
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), head_(SYNCED), size_(size) {};
  ~SyncedMemory();
  const void* cpu_data();
  const void* gpu_data();
  void* mutable_cpu_data();
  void* mutable_gpu_data();
 private:
  void to_cpu();
  void to_gpu();
  void* cpu_ptr_;
  void* gpu_ptr_;
  size_t size_;
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head_;
};  // class SyncedMemory

}  // namespace caffeine

#endif  // CAFFEINE_SYNCEDMEM_HPP_
