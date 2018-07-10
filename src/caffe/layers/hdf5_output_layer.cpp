#ifdef USE_HDF5
#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/layers/hdf5_output_layer.hpp"
#include "caffe/util/hdf5.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void HDF5OutputLayer<Dtype, MItype, MOtype>::LayerSetUp(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  file_name_ = this->layer_param_.hdf5_output_param().file_name();
  file_id_ = H5Fcreate(file_name_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                       H5P_DEFAULT);
  CHECK_GE(file_id_, 0) << "Failed to open HDF5 file" << file_name_;
  file_opened_ = true;
  this->InitializeQuantizers(bottom, top);
}

template<typename Dtype, typename MItype, typename MOtype>
HDF5OutputLayer<Dtype, MItype, MOtype>::
    ~HDF5OutputLayer<Dtype, MItype, MOtype>() {
  if (file_opened_) {
    herr_t status = H5Fclose(file_id_);
    CHECK_GE(status, 0) << "Failed to close HDF5 file " << file_name_;
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void HDF5OutputLayer<Dtype, MItype, MOtype>::SaveBlobs() {
  // TODO: no limit on the number of blobs
  LOG(INFO) << "Saving HDF5 file " << file_name_;
  CHECK_EQ(data_blob_.num(), label_blob_.num()) <<
      "data blob and label blob must have the same batch size";
  hdf5_save_nd_dataset(file_id_, HDF5_DATA_DATASET_NAME, data_blob_);
  hdf5_save_nd_dataset(file_id_, HDF5_DATA_LABEL_NAME, label_blob_);
  LOG(INFO) << "Successfully saved " << data_blob_.num() << " rows";
}

template<typename Dtype, typename MItype, typename MOtype>
void HDF5OutputLayer<Dtype, MItype, MOtype>::Forward_cpu(
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
    caffe_copy(data_datum_dim, &bottom[0]->cpu_data()[i * data_datum_dim],
        &data_blob_.mutable_cpu_data()[i * data_datum_dim]);
    caffe_copy(label_datum_dim, &bottom[1]->cpu_data()[i * label_datum_dim],
        &label_blob_.mutable_cpu_data()[i * label_datum_dim]);
  }
  SaveBlobs();
}

template<typename Dtype, typename MItype, typename MOtype>
void HDF5OutputLayer<Dtype, MItype, MOtype>::Backward_cpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<MItype>*>& bottom) {
  return;
}

#ifdef CPU_ONLY
STUB_GPU(HDF5OutputLayer);
#endif

INSTANTIATE_CLASS_3T_GUARDED(HDF5OutputLayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(HDF5OutputLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(HDF5OutputLayer, (double), (double), (double));

REGISTER_LAYER_CLASS(HDF5Output);
REGISTER_LAYER_CLASS_INST(HDF5Output, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(HDF5Output, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(HDF5Output, (double), (double), (double));

}  // namespace caffe
#endif  // USE_HDF5
