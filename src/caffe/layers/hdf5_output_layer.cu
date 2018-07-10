#ifdef USE_HDF5
#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/layers/hdf5_output_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void HDF5OutputLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                         const vector<Blob<MItype>*>& bottom,
                                         const vector<Blob<MOtype>*>& top) {
  CHECK_GE(bottom.size(), 2);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  data_blob_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                     bottom[0]->height(), bottom[0]->width());
  label_blob_.Reshape(bottom[1]->num(), bottom[1]->channels(),
                      bottom[1]->height(), bottom[1]->width());
  const int_tp data_datum_dim = bottom[0]->count() / bottom[0]->num();
  const int_tp label_datum_dim = bottom[1]->count() / bottom[1]->num();

  for (int_tp i = 0; i < bottom[0]->num(); ++i) {
    this->device_->template copy<Dtype>(data_datum_dim,
                       bottom[0]->gpu_data() + i * data_datum_dim,
                       &data_blob_.mutable_cpu_data()[i * data_datum_dim]);
    this->device_->template copy<Dtype>(label_datum_dim,
                       bottom[1]->gpu_data() + i * label_datum_dim,
                       &label_blob_.mutable_cpu_data()[i * label_datum_dim]);
  }
  SaveBlobs();
}

template<typename Dtype, typename MItype, typename MOtype>
void HDF5OutputLayer<Dtype, MItype, MOtype>::Backward_gpu(
                                         const vector<Blob<MOtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<MItype>*>& bottom) {
  return;
}


INSTANTIATE_CLASST_FUNC_3T_GUARDED(HDF5OutputLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(HDF5OutputLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(HDF5OutputLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(HDF5OutputLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(HDF5OutputLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(HDF5OutputLayer, Backward_gpu,
                                  (double), (double), (double));

}  // namespace caffe
#endif  // USE_HDF5
