#ifndef CAFFE_QUANTIZER_HPP_
#define CAFFE_QUANTIZER_HPP_

#include "caffe/common.hpp"
#include "caffe/backend/vptr.hpp"

namespace caffe {

template<typename MItype, typename MOtype>
class Quantizer {
 public:
  void Forward_cpu(MItype* input, MOtype* output);
  void Backward_cpu(MOtype* input, MItype* output);
  void Forward_gpu(vptr<MItype> input, vptr<MOtype> output);
  void Backward_gpu(vptr<MOtype> input, vptr<MItype> output);
 private:
};

}  // namespace caffe

#endif
