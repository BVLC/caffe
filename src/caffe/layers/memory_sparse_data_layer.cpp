#include <vector>

#include "caffe/layers/memory_sparse_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void MemorySparseDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
  batch_size_ = this->layer_param_.memory_data_param().batch_size();
  channels_ = this->layer_param_.memory_data_param().channels();
  height_ = this->layer_param_.memory_data_param().height();
  width_ = this->layer_param_.memory_data_param().width();
  vector<int> label_shape(1, batch_size_);
  if (SparseBlob<Dtype>* sparseBlob = dynamic_cast<SparseBlob<Dtype>*>(top[0]))
    {
      std::cerr << "batch_size_=" << batch_size_ << " / channels_=" << channels_ << " / height_=" << height_ << " / width_=" << width_ << std::endl;
      sparseBlob->Reshape(batch_size_, channels_, height_, width_);
      std::cerr << "label_shape=" << label_shape[0] << " / " << label_shape[1] << std::endl;
      top[1]->Reshape(label_shape);
    } else {
    LOG(FATAL)<< "The top blob in the memory sparse data layer is not sparse\n";
  }
  added_data_.Reshape(batch_size_, channels_, height_, width_);
  added_label_.Reshape(label_shape);
  data_ = NULL;
  labels_ = NULL;
  added_data_.cpu_data();
  added_label_.cpu_data();
}

template <typename Dtype>
void MemorySparseDataLayer<Dtype>::AddDatumVector(const vector<SparseDatum>& datum_vector) {
  CHECK(!has_new_data_) <<
      "Can't add data until current data has been consumed.";
  size_t datum_num = datum_vector.size();
  CHECK_GT(datum_num, 0) << "There is no datum to add.";
  CHECK_EQ(datum_num % batch_size_, 0) <<
      "The added data must be a multiple of the batch size.";
  vector<int> top_shape = {static_cast<int>(datum_num),static_cast<int>(datum_vector.at(0).size())};
  
  int nnz = 0;
  for (int i = 0; i < datum_num; i++) {
    nnz += datum_vector[i].nnz();
  }
  added_data_.Reshape(top_shape,nnz);
  added_label_.Reshape(datum_num, 1, 1, 1);
  
  const int num = added_data_.num();
  Dtype *top_data = added_data_.mutable_cpu_data();
  int* indices = added_data_.mutable_cpu_indices();
  int* ptr = added_data_.mutable_cpu_ptr();
  ptr[0] = 0;
  int pos = 0;
  for (int item_id = 0; item_id < (int)datum_num; ++item_id) {
    SparseDatum datum = datum_vector.at(item_id);
    for (int k=0;k<datum.nnz();k++) {
      top_data[k + pos] = datum.data(k);
      indices[k + pos] = datum.indices(k);
    }
    pos += datum.nnz();
    ptr[item_id + 1] = pos;
  }
  
  // Copy Labels
  Dtype* top_label = added_label_.mutable_cpu_data();
  for (int item_id = 0; item_id < num; ++item_id) {
    top_label[item_id] = datum_vector[item_id].label();
  }
  Reset(top_data, indices, ptr, top_label, num, nnz);
  has_new_data_ = true;
}

template <typename Dtype>
void MemorySparseDataLayer<Dtype>::Reset(Dtype* data, int *indices, int *ptr, Dtype* labels, int n, int nnz) {
  CHECK(data);
  CHECK(labels);
  CHECK_EQ(n % batch_size_, 0) << "n must be a multiple of batch size";
  data_ = data;
  indices_ = indices;
  ptr_ = ptr;
  labels_ = labels;
  nnz_ = nnz;
  n_ = n;
  pos_ = 0;
}

template <typename Dtype>
void MemorySparseDataLayer<Dtype>::set_batch_size(int new_size) {
  CHECK(!has_new_data_) <<
      "Can't change batch_size until current data has been consumed.";
  batch_size_ = new_size;
  added_data_.Reshape(batch_size_, channels_, height_, width_);
  added_label_.Reshape(batch_size_, 1, 1, 1);
}

template <typename Dtype>
void MemorySparseDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  std::cerr << "Forward_cpu\n";
  CHECK(data_) << "MemorySparseDataLayer needs to be initalized by calling Reset";
  if (SparseBlob<Dtype>* sparseBlob = dynamic_cast<SparseBlob<Dtype>*>(top[0]))
    {
      sparseBlob->Reshape(batch_size_, channels_, height_, width_);
      sparseBlob->set_cpu_data(data_, indices_, ptr_, nnz_);
    } else {
    LOG(FATAL) << "The top blob in the memory sparse data layer is not sparse";
    }
 
  DLOG(INFO) << "Prefetch sparse copied (forward)";
  if (this->output_labels_) {
    top[1]->Reshape(batch_size_, 1, 1, 1);
    top[1]->set_cpu_data(labels_);
  }

  /*if (pos_ == 0) // XXX: not sure we need this.
    has_new_data_ = false;*/
}

INSTANTIATE_CLASS(MemorySparseDataLayer);
REGISTER_LAYER_CLASS(MemorySparseData);

}  // namespace caffe
