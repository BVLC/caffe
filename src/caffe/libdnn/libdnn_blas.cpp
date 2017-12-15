#include "caffe/libdnn/libdnn_blas.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
LibDNNBlas<Dtype, MItype, MOtype>::LibDNNBlas(Device* dev_ptr)
    : LibDNN<Dtype, MItype, MOtype>(dev_ptr) {

}

template<typename Dtype, typename MItype, typename MOtype>
size_t LibDNNBlas<Dtype, MItype, MOtype>::get_id(string identifier) {
  map<string,size_t>::iterator it = program_map_.find(identifier);
  if (it != program_map_.end()) {
    size_t id = program_map_.size();
    program_map_[identifier] = id;
    program_tuners_.push_back(std::make_shared<LibDNNTuner>());
    programs_.push_back(this->dev_ptr_->CreateProgram());
    program_ready_.push_back(false);
    return id;
  }
  return program_map_[identifier];
}


INSTANTIATE_CLASS_3T(LibDNNBlas, (half_fp), VARIANT_TYPES, VARIANT_TYPES);
INSTANTIATE_CLASS_3T(LibDNNBlas, (float), VARIANT_TYPES, VARIANT_TYPES);
INSTANTIATE_CLASS_3T(LibDNNBlas, (double), VARIANT_TYPES, VARIANT_TYPES);
INSTANTIATE_CLASS_3T(LibDNNBlas, (int8_t), VARIANT_TYPES, VARIANT_TYPES);
INSTANTIATE_CLASS_3T(LibDNNBlas, (int16_t), VARIANT_TYPES, VARIANT_TYPES);
INSTANTIATE_CLASS_3T(LibDNNBlas, (int32_t), VARIANT_TYPES, VARIANT_TYPES);
INSTANTIATE_CLASS_3T(LibDNNBlas, (int64_t), VARIANT_TYPES, VARIANT_TYPES);

}  // namespace caffe
