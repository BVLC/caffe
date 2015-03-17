#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
HDF5OutputLayer<Dtype>::HDF5OutputLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      file_name_(param.hdf5_output_param().file_name()) {
  /* create a HDF5 file */
  file_id_ = H5Fcreate(file_name_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                       H5P_DEFAULT);
  CHECK_GE(file_id_, 0) << "Failed to open HDF5 file" << file_name_;
  current_batch_ = 0;
}

template <typename Dtype>
HDF5OutputLayer<Dtype>::~HDF5OutputLayer<Dtype>() {
  herr_t status = H5Fclose(file_id_);
  CHECK_GE(status, 0) << "Failed to close HDF5 file " << file_name_;
}

template <typename Dtype>
void HDF5OutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(this->layer_param_.bottom_size(), bottom.size());
  for (int i = 0; i < bottom.size(); ++i) {
    stringstream batch_id;
    batch_id << this->layer_param_.bottom(i) << "_" << current_batch_;
    LOG_FIRST_N(INFO, bottom.size()) << "Saving batch " << batch_id.str()
        << " to HDF5 file " << file_name_;
    hdf5_save_nd_dataset(file_id_, batch_id.str(), *bottom[i]);
  }
  H5Fflush(file_id_, H5F_SCOPE_GLOBAL);
  current_batch_++;
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
REGISTER_LAYER_CLASS(HDF5_OUTPUT, HDF5OutputLayer);
}  // namespace caffe
