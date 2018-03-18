#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV

#include <vector>

#include "caffe/layers/memory_data_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void MemoryDataLayer<Dtype, MItype, MOtype>::DataLayerSetUp(
    const vector<Blob<MItype>*>& bottom, const vector<Blob<MOtype>*>& top) {
  MemoryDataParameter mem_param = this->layer_param_.memory_data_param();

  has_label_ = false;

  // Old 4D (2D spatial) parameters
  shape_.clear();
  shape_.push_back(mem_param.batch_size());
  shape_.push_back(mem_param.channels());
  shape_.push_back(mem_param.height());
  shape_.push_back(mem_param.width());

  // New ND parameters
  if (mem_param.dim_size() > 0) {
    shape_.clear();
    for (int_tp i = 0; i < mem_param.dim_size(); ++i) {
      shape_.push_back(mem_param.dim(i));
    }
  }

  top[0]->Reshape(shape_);
  added_data_.Reshape(shape_);
  data_ = NULL;
  added_data_.cpu_data();

  if (top.size() == 2) {
    has_label_ = true;
    // Labels have shape batch_size, 1, 1, ..., 1
    label_shape_.push_back(shape_[0]);
    size_ = 1;
    // All sizes except the batch index
    for (int_tp i = 1; i < shape_.size(); ++i) {
      size_ *= shape_[i];
      label_shape_.push_back(1);
    }
    top[1]->Reshape(label_shape_);
    added_label_.Reshape(label_shape_);
    labels_ = NULL;
    added_label_.cpu_data();
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void MemoryDataLayer<Dtype, MItype, MOtype>::AddDatumVector(
    const vector<Datum>& datum_vector) {
  CHECK(!has_new_data_) <<
  "Can't add data until current data has been consumed.";
  uint_tp num = datum_vector.size();
  CHECK_GT(num, 0)<< "There is no datum to add.";
  CHECK_EQ(num % shape_[0], 0)<<
  "The added data must be a multiple of the batch size.";
  vector<int_tp> added_shape = shape_;
  added_shape[0] = num;
  added_data_.Reshape(added_shape);

  // Apply data transformations (mirror, scale, crop...)
  this->data_transformer_->Transform(datum_vector, &added_data_);
  Dtype* top_data = added_data_.mutable_cpu_data();
  Dtype* top_label = nullptr;

  if (has_label_) {
    vector<int_tp> added_label_shape = label_shape_;
    added_label_shape[0] = num;
    added_label_.Reshape(added_label_shape);
    // Copy Labels
    top_label = added_label_.mutable_cpu_data();
    for (int_tp item_id = 0; item_id < num; ++item_id) {
      top_label[item_id] = datum_vector[item_id].label();
    }
  }
  // num_images == batch_size_
  Reset(top_data, top_label, num);
  has_new_data_ = true;
}

#ifdef USE_OPENCV
template<typename Dtype, typename MItype, typename MOtype>
void MemoryDataLayer<Dtype, MItype, MOtype>::AddMatVector(
    const vector<cv::Mat>& mat_vector, const vector<int_tp>& labels) {
  uint_tp num = mat_vector.size();
  CHECK(!has_new_data_) <<
  "Can't add mat until current data has been consumed.";
  CHECK_GT(num, 0) << "There is no mat to add";
  CHECK_EQ(num % shape_[0], 0) <<
  "The added data must be a multiple of the batch size.";
  vector<int_tp> added_shape = shape_;
  added_shape[0] = num;
  added_data_.Reshape(added_shape);

  // Apply data transformations (mirror, scale, crop...)
  this->data_transformer_->Transform(mat_vector, &added_data_);
  Dtype* top_data = added_data_.mutable_cpu_data();
  Dtype* top_label = nullptr;

  if (has_label_) {
    vector<int_tp> added_label_shape = label_shape_;
    added_label_shape[0] = num;
    added_label_.Reshape(added_label_shape);
    // Copy Labels
    top_label = added_label_.mutable_cpu_data();
    for (int_tp item_id = 0; item_id < num; ++item_id) {
      top_label[item_id] = labels[item_id];
    }
  }
  // num_images == batch_size_
  Reset(top_data, top_label, num);
  has_new_data_ = true;
}
#endif  // USE_OPENCV

template<typename Dtype, typename MItype, typename MOtype>
void MemoryDataLayer<Dtype, MItype, MOtype>::Reset(Dtype* data,
                                                   Dtype* labels, int_tp n) {
  CHECK(data);
  if (has_label_) {
    CHECK(labels);
  }
  CHECK_EQ(n % shape_[0], 0)<< "n must be a multiple of batch size";
  // Warn with transformation parameters since a memory array is meant to
  // be generic and no transformations are done with Reset().
  if (this->layer_param_.has_transform_param()) {
    LOG(WARNING)<< this->type() << " does not transform array data on Reset()";
  }
  data_ = data;
  labels_ = labels;
  n_ = n;
  pos_ = 0;
}

template<typename Dtype, typename MItype, typename MOtype>
void MemoryDataLayer<Dtype, MItype, MOtype>::set_batch_size(int_tp new_size) {
  CHECK(!has_new_data_) <<
  "Can't change batch_size until current data has been consumed.";
  shape_[0] = new_size;
  added_data_.Reshape(shape_);
  if (has_label_) {
    label_shape_[0] = new_size;
    added_label_.Reshape(label_shape_);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void MemoryDataLayer<Dtype, MItype, MOtype>::Forward_cpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  CHECK(data_) << "MemoryDataLayer needs to be initialized by calling Reset";
  top[0]->Reshape(shape_);
  top[0]->set_cpu_data(data_ + pos_ * size_);
  if (top.size() == 2 && has_label_) {
    top[1]->Reshape(label_shape_);
    top[1]->set_cpu_data(labels_ + pos_);
  }
  pos_ = (pos_ + shape_[0]) % n_;
  if (pos_ == 0) {
    has_new_data_ = false;
  }
}

INSTANTIATE_CLASS_3T_GUARDED(MemoryDataLayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(MemoryDataLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(MemoryDataLayer, (double), (double), (double));
INSTANTIATE_CLASS_3T_GUARDED(MemoryDataLayer, (uint8_t), (uint8_t), (uint8_t));
INSTANTIATE_CLASS_3T_GUARDED(MemoryDataLayer,
                             (uint16_t), (uint16_t), (uint16_t));
INSTANTIATE_CLASS_3T_GUARDED(MemoryDataLayer,
                             (uint32_t), (uint32_t), (uint32_t));
INSTANTIATE_CLASS_3T_GUARDED(MemoryDataLayer,
                             (uint64_t), (uint64_t), (uint64_t));


REGISTER_LAYER_CLASS(MemoryData);
REGISTER_LAYER_CLASS_INST(MemoryData, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(MemoryData, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(MemoryData, (double), (double), (double));
REGISTER_LAYER_CLASS_INST(MemoryData, (uint8_t), (uint8_t), (uint8_t));
REGISTER_LAYER_CLASS_INST(MemoryData, (uint16_t), (uint16_t), (uint16_t));
REGISTER_LAYER_CLASS_INST(MemoryData, (uint32_t), (uint32_t), (uint32_t));
REGISTER_LAYER_CLASS_INST(MemoryData, (uint64_t), (uint64_t), (uint64_t));


}  // namespace caffe
