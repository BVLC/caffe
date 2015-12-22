#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#include "caffe/layers/hdf5_output_layer.hpp"
=======
#include "caffe/data_layers.hpp"
>>>>>>> pod-caffe-pod.hpp-merge
=======
#include "caffe/data_layers.hpp"
>>>>>>> pod/caffe-merge
=======
#include "caffe/data_layers.hpp"
>>>>>>> pod-caffe-pod.hpp-merge
=======
#include "caffe/data_layers.hpp"
>>>>>>> device-abstraction
=======
#include "caffe/data_layers.hpp"
>>>>>>> pod/post-rebase-error-fix
=======
#include "caffe/data_layers.hpp"
>>>>>>> pod-caffe-pod.hpp-merge
#include "caffe/util/hdf5.hpp"

namespace caffe {

template <typename Dtype>
void HDF5OutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  file_name_ = this->layer_param_.hdf5_output_param().file_name();
  file_id_ = H5Fcreate(file_name_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                       H5P_DEFAULT);
  CHECK_GE(file_id_, 0) << "Failed to open HDF5 file" << file_name_;
  file_opened_ = true;
}

template <typename Dtype>
HDF5OutputLayer<Dtype>::~HDF5OutputLayer<Dtype>() {
  if (file_opened_) {
    herr_t status = H5Fclose(file_id_);
    CHECK_GE(status, 0) << "Failed to close HDF5 file " << file_name_;
  }
}

template <typename Dtype>
void HDF5OutputLayer<Dtype>::SaveBlobs() {
  // TODO: no limit on the number of blobs
  LOG(INFO) << "Saving HDF5 file " << file_name_;
  CHECK_EQ(data_blob_.num(), label_blob_.num()) <<
      "data blob and label blob must have the same batch size";
  hdf5_save_nd_dataset(file_id_, HDF5_DATA_DATASET_NAME, data_blob_);
  hdf5_save_nd_dataset(file_id_, HDF5_DATA_LABEL_NAME, label_blob_);
  LOG(INFO) << "Successfully saved " << data_blob_.num() << " rows";
}

template <typename Dtype>
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
void HDF5OutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
=======
=======
>>>>>>> BVLC/device-abstraction
Dtype HDF5OutputLayer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> device-abstraction
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> pod/post-rebase-error-fix
=======
>>>>>>> pod-caffe-pod.hpp-merge
Dtype HDF5OutputLayer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
=======
void HDF5OutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
>>>>>>> BVLC/master
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
void HDF5OutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
>>>>>>> BVLC/master
=======
void HDF5OutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
>>>>>>> BVLC/master
=======
void HDF5OutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
>>>>>>> master
=======
void HDF5OutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
>>>>>>> caffe
=======
void HDF5OutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
>>>>>>> master
=======
void HDF5OutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
>>>>>>> master
=======
void HDF5OutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
>>>>>>> BVLC/master
=======
void HDF5OutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
>>>>>>> master
=======
void HDF5OutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
>>>>>>> master
=======
void HDF5OutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
>>>>>>> origin/BVLC/parallel
=======
void HDF5OutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod/post-rebase-error-fix
=======
>>>>>>> pod-caffe-pod.hpp-merge
  CHECK_GE(bottom.size(), 2);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  data_blob_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                     bottom[0]->height(), bottom[0]->width());
  label_blob_.Reshape(bottom[1]->num(), bottom[1]->channels(),
                     bottom[1]->height(), bottom[1]->width());
  const int data_datum_dim = bottom[0]->count() / bottom[0]->num();
  const int label_datum_dim = bottom[1]->count() / bottom[1]->num();

  for (int i = 0; i < bottom[0]->num(); ++i) {
    this->device_->copy(data_datum_dim,
        &bottom[0]->cpu_data()[i * data_datum_dim],
        &data_blob_.mutable_data()[i * data_datum_dim]);
    this->device_->copy(label_datum_dim,
        &bottom[1]->cpu_data()[i * label_datum_dim],
        &label_blob_.mutable_data()[i * label_datum_dim]);
  }
  SaveBlobs();
}

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/caffe-merge
=======
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/caffe-merge
=======
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> pod/post-rebase-error-fix
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
template <typename Dtype>
void HDF5OutputLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  return;
}

#ifdef CPU_ONLY
STUB_GPU(HDF5OutputLayer);
#endif

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
INSTANTIATE_CLASS(HDF5OutputLayer);
=======
>>>>>>> BVLC/master
INSTANTIATE_CLASS(HDF5OutputLayer);
<<<<<<< HEAD
=======
>>>>>>> BVLC/master
INSTANTIATE_CLASS(HDF5OutputLayer);
<<<<<<< HEAD
=======
>>>>>>> BVLC/master
INSTANTIATE_CLASS(HDF5OutputLayer);
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/device-abstraction
INSTANTIATE_CLASS(HDF5OutputLayer);
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> BVLC/master
INSTANTIATE_CLASS(HDF5OutputLayer);
>>>>>>> device-abstraction
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> BVLC/master
INSTANTIATE_CLASS(HDF5OutputLayer);
=======
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
INSTANTIATE_CLASS(HDF5OutputLayer);
=======
>>>>>>> BVLC/master
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
INSTANTIATE_CLASS(HDF5OutputLayer);
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
INSTANTIATE_CLASS(HDF5OutputLayer);
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/device-abstraction
INSTANTIATE_CLASS(HDF5OutputLayer);
>>>>>>> device-abstraction
=======
>>>>>>> BVLC/master
INSTANTIATE_CLASS(HDF5OutputLayer);
>>>>>>> pod/post-rebase-error-fix
=======
>>>>>>> BVLC/master
INSTANTIATE_CLASS(HDF5OutputLayer);
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
REGISTER_LAYER_CLASS(HDF5Output);

=======
REGISTER_LAYER_CLASS(HDF5_OUTPUT, HDF5OutputLayer);
>>>>>>> origin/BVLC/parallel
}  // namespace caffe
