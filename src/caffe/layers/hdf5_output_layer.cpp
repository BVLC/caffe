#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/layers/hdf5_output_layer.hpp"
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

  for (int i = 0; i < bottom.size(); i++)
    data_blobs_.push_back(new Blob<Dtype>());
}

template <typename Dtype>
HDF5OutputLayer<Dtype>::~HDF5OutputLayer<Dtype>() {
  if (file_opened_) {
    herr_t status = H5Fclose(file_id_);
    CHECK_GE(status, 0) << "Failed to close HDF5 file " << file_name_;
  }
  for (int i = 0; i < data_blobs_.size(); i++) {
    delete data_blobs_[i];
  }
}

template <typename Dtype>
void HDF5OutputLayer<Dtype>::SaveBlobs() {
  // TODO: no limit on the number of blobs
  LOG(INFO) << "Saving HDF5 file " << file_name_;
  for (int i = 0; i < data_blobs_.size(); i++) {
    stringstream ss;
    ss << "blob" << i;
    hdf5_save_nd_dataset(file_id_, ss.str(), *(data_blobs_[i]));
  }
  LOG(INFO) << "Successfully saved " << data_blobs_[0]->num() << " rows";
}

template <typename Dtype>
void HDF5OutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom.size(), 1);

  for (int i = 0; i < bottom.size(); ++i) {
    CHECK_EQ(bottom[0]->num(), bottom[i]->num());
    data_blobs_[i]->Reshape(bottom[i]->num(), bottom[i]->channels(),
                       bottom[i]->height(), bottom[i]->width());
    caffe_copy(bottom[i]->count(), bottom[i]->cpu_data(),
               data_blobs_[i]->mutable_cpu_data());
  }
  SaveBlobs();
}

template <typename Dtype>
void HDF5OutputLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  return;
}

#ifdef CPU_ONLY
STUB_GPU(HDF5OutputLayer);
#endif

INSTANTIATE_CLASS(HDF5OutputLayer);
REGISTER_LAYER_CLASS(HDF5Output);

}  // namespace caffe
