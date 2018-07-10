#ifdef USE_LIBDNN
#include "caffe/libdnn/libdnn_blas.hpp"

namespace caffe {

template<typename MItype, typename MOtype>
LibDNNBlas<MItype, MOtype>::LibDNNBlas(Device* dev_ptr)
    : LibDNN<MItype, MOtype>(dev_ptr) {

}

template<typename MItype, typename MOtype>
int_tp LibDNNBlas<MItype, MOtype>::get_id(string identifier) {
  boost::shared_lock<boost::shared_mutex> lock(program_mutex_);
  std::unordered_map<string, size_t>::iterator it =
      program_map_.find(identifier);
  if (it == program_map_.end()) {
    lock.unlock();
    return -1;
  }
  lock.unlock();
  return program_map_[identifier];
}

template<typename MItype, typename MOtype>
int_tp LibDNNBlas<MItype, MOtype>::get_id_or_new(string identifier) {
  boost::unique_lock<boost::shared_mutex> ulock(program_mutex_);
  std::unordered_map<string, size_t>::iterator it =
      program_map_.find(identifier);
  if (it == program_map_.end()) {
    size_t id = program_map_.size();
    program_map_[identifier] = id;
    program_tuners_.push_back(make_shared<LibDNNTuner>());
    programs_.push_back(this->dev_ptr_->CreateProgram());
    program_ready_.push_back(false);
    ulock.unlock();
    return id;
  }
  ulock.unlock();
  return program_map_[identifier];
}

INSTANTIATE_CLASS_2T_GUARDED(LibDNNBlas, PROTO_TYPES, PROTO_TYPES);

}  // namespace caffe

#endif  // USE_LIBDNN
