#include "caffe/libdnn/libdnn_blas.hpp"

namespace caffe {

template<typename MItype, typename MOtype>
LibDNNBlas<MItype, MOtype>::LibDNNBlas(Device* dev_ptr)
    : LibDNN<MItype, MOtype>(dev_ptr) {

}

template<typename MItype, typename MOtype>
size_t LibDNNBlas<MItype, MOtype>::get_id(string identifier) {
  std::unordered_map<string, size_t>::iterator it =
      program_map_.find(identifier);
  if (it == program_map_.end()) {
    size_t id = program_map_.size();
    program_map_[identifier] = id;
    program_tuners_.push_back(std::make_shared<LibDNNTuner>());
    programs_.push_back(this->dev_ptr_->CreateProgram());
    program_ready_.push_back(false);
    return id;
  }
  return program_map_[identifier];
}

INSTANTIATE_CLASS_2T_GUARDED(LibDNNBlas, PROTO_TYPES, PROTO_TYPES);


}  // namespace caffe
