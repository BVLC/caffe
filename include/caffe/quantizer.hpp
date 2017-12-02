#ifndef CAFFE_QUANTIZER_HPP_
#define CAFFE_QUANTIZER_HPP_

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/backend/vptr.hpp"
#include "caffe/backend/device_program.hpp"

namespace caffe {

class QuantizerBase {
 public:
  virtual void Forward_cpu(size_t n, const void* input, void* output) = 0;
  virtual void Backward_cpu(size_t n, const void* input, void* output) = 0;
  virtual void Forward_gpu(size_t n, vptr<const void> input,
                           vptr<void> output) = 0;
  virtual void Backward_gpu(size_t n, vptr<const void> input,
                            vptr<void> output) = 0;
 protected:
  explicit QuantizerBase(QuantizerParameter& param);
 private:
  QuantizerParameter quant_param_;
  shared_ptr<DeviceProgram> quantizer_program_;
  Device* device_;
};

template<typename MItype, typename MOtype>
class Quantizer : public QuantizerBase {
 public:
  explicit Quantizer(QuantizerParameter& param);

  void Forward(Blob<MItype>* input, Blob<MOtype>* output,
               bool fw_data, bool fw_diff);
  void Backward(Blob<MOtype>* input, Blob<MItype>* output,
                bool bw_data, bool bw_diff);
  void Forward_cpu(Blob<MItype>* input, Blob<MOtype>* output,
                   bool fw_data, bool fw_diff);
  void Backward_cpu(Blob<MOtype>* input, Blob<MItype>* output,
                    bool bw_data, bool bw_diff);
  void Forward_gpu(Blob<MItype>* input, Blob<MOtype>* output,
                   bool fw_data, bool fw_diff);
  void Backward_gpu(Blob<MOtype>* input, Blob<MItype>* output,
                    bool bw_data, bool bw_diff);
  virtual void Forward_cpu(size_t n, const void* input, void* output);
  virtual void Backward_cpu(size_t n, const void* input, void* output);
  virtual void Forward_gpu(size_t n, vptr<const void> input,
                           vptr<void> output);
  virtual void Backward_gpu(size_t n, vptr<const void> input,
                            vptr<void> output);
  void Forward_cpu(size_t n, const MItype* input, MOtype* output);
  void Backward_cpu(size_t n, const MOtype* input, MItype* output);
  void Forward_gpu(size_t n, vptr<const MItype> input, vptr<MOtype> output);
  void Backward_gpu(size_t n, vptr<const MOtype> input, vptr<MItype> output);
};

}  // namespace caffe

#endif  // CAFFE_QUANTIZER_HPP_
