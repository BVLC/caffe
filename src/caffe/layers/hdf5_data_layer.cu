#ifdef USE_HDF5
/*
 TODO:
 - only load parts of the file, in accordance with a prototxt param "max_mem"
 */

#include <stdint.h>
#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/layers/hdf5_data_layer.hpp"

namespace caffe {


// Fast data path (native loading)
template<typename Dtype, typename MOtype>
inline typename std::enable_if<std::is_same<Dtype, MOtype>::value, void>::type
data_copy_to_top(int_tp n, const Dtype* data, vptr<MOtype> top,
                             QuantizerBase* quant, Device* dev) {
  if (!quant->needs_quantization()) {
    dev->template copy(n, data, top);
  } else {
    int_tp buffer_id = -1;
    shared_ptr<Blob<Dtype> > buff = dev->template Buffer<Dtype>(
                                         std::vector<int_tp>(1, n), &buffer_id);
    dev->template copy(n, data, buff->mutable_gpu_data());
    quant->Forward_gpu(n, buff->gpu_data(), top);
    dev->unlock_buffer(&buffer_id);
  }
}

// Slow data path (non-native loading)
template<typename Dtype, typename MOtype>
inline typename std::enable_if<!std::is_same<Dtype, MOtype>::value, void>::type
data_copy_to_top(int_tp n, const Dtype* data, vptr<MOtype> top,
                             QuantizerBase* quant, Device* dev) {
  int_tp buffer_id = -1;
  shared_ptr<Blob<Dtype> > buff = dev->template Buffer<Dtype>(
                                       std::vector<int_tp>(1, n), &buffer_id);
  dev->template copy(n, data, buff->mutable_gpu_data());
  quant->Forward_gpu(n, buff->gpu_data(), top);
  dev->unlock_buffer(&buffer_id);
}

template<typename Dtype, typename MItype, typename MOtype>
void HDF5DataLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                         const vector<Blob<MItype>*>& bottom,
                                         const vector<Blob<MOtype>*>& top) {
  const int_tp batch_size = this->layer_param_.hdf5_data_param().batch_size();
  for (int_tp i = 0; i < batch_size; ++i) {
    while (Skip()) {
      Next();
    }
    for (int j = 0; j < this->layer_param_.top_size(); ++j) {
      int data_dim = top[j]->count() / top[j]->shape(0);
      vptr<MOtype> top_data = top[j]->mutable_gpu_data() + i * data_dim;
      data_copy_to_top<Dtype, MOtype>(data_dim,
         &hdf_blobs_[j]->cpu_data()[data_permutation_[current_row_] * data_dim],
         top_data, this->top_quants_[j].get(), this->device_);
    }
    Next();
  }
}


INSTANTIATE_CLASST_FUNC_3T_GUARDED(HDF5DataLayer, Forward_gpu,
                                  (half_fp), (half_fp),
                                  (half_fp)(float)(double));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(HDF5DataLayer, Forward_gpu,
                                  (float), (float),
                                  (half_fp)(float)(double));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(HDF5DataLayer, Forward_gpu,
                                  (double), (double),
                                  (half_fp)(float)(double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(HDF5DataLayer, Backward_gpu,
                                  (half_fp), (half_fp),
                                  (half_fp)(float)(double));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(HDF5DataLayer, Backward_gpu,
                                  (float), (float),
                                  (half_fp)(float)(double));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(HDF5DataLayer, Backward_gpu,
                                  (double), (double),
                                  (half_fp)(float)(double));

}  // namespace caffe
#endif  // USE_HDF5
