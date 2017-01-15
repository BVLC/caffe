#include <string>
#include <vector>

#include "caffe/layers/external_lib_data_layer.hpp"
#include "caffe/util/external_lib_data_source.hpp"
#include "caffe/util/external_lib_data_source_util.hpp"
#include "glog/logging.h"

namespace caffe {

  // Helper structs to map actual blob type to BlobType.
  template <typename T>
  struct BlobTypeMap {
    static BlobType GetType() {
      LOG(FATAL) << "Blob type mapping not implemented.";
      return BlobTypeFLOAT;
    }
  };
  // Explicit instantiation for float.
  template <>
  struct BlobTypeMap<float> {
    static BlobType GetType() {
      return BlobTypeFLOAT;
    }
  };
  // Explicit instantiation for double.
  template <>
  struct BlobTypeMap<double> {
    static BlobType GetType() {
      return BlobTypeDOUBLE;
    }
  };

  template <typename Dtype>
  ExternalLibDataLayer<Dtype>::ExternalLibDataLayer(
    const LayerParameter& param)
    : BasePrefetchingDataLayer<Dtype>(param) {
    }

  template <typename Dtype>
  ExternalLibDataLayer<Dtype>::~ExternalLibDataLayer<Dtype>() {
    this->StopInternalThread();
  }

  template <typename Dtype>
  void ExternalLibDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    // Get data source object.
    ExternalLibDataParameter ext_lib_data_param =
      this->layer_param_.external_lib_data_param();
    external_lib_interface_ = GetExternalLib(
      ext_lib_data_param.external_lib_path(),
      ext_lib_data_param.factory_method_name(),
      ext_lib_data_param.external_lib_param());
    data_source_ = external_lib_interface_->GetDataSource();

    // Convert names of tops to be fetched to GPU to corresponding indices.
    if (ext_lib_data_param.gpu_prefetch_top_size() == 0) {
      // In case nothing is defined we default to prefetching just first top
      // like in other data layers.
      gpu_prefetch_indices_.push_back(0);
    } else {
      for (int it = 0; it < this->layer_param_.top_size(); it++) {
        const string& top_name = this->layer_param_.top(it);
        bool found = false;
        for (int in = 0; in < ext_lib_data_param.gpu_prefetch_top_size();
          in++) {
          if (top_name == ext_lib_data_param.gpu_prefetch_top(in)) {
            gpu_prefetch_indices_.push_back(it);
            found = true;
            break;
          }
        }
        CHECK(found);
      }
    }

    // Make sure we have number of blobs equal to number of tops.
    for (int ip = 0; ip < this->PREFETCH_COUNT; ip++) {
      CHECK(this->prefetch_[ip].data_.size() <= this->layer_param_.top_size());
      for (int il = this->prefetch_[ip].data_.size();
        il < this->layer_param_.top_size(); il++) {
        this->prefetch_[ip].data_.push_back(
          shared_ptr<Blob<Dtype> >(new Blob<Dtype>()));
      }
    }

    // Now we need to reshape.
    const int batch_size = ext_lib_data_param.batch_size();
    IDatum* example = data_source_->GetCurrent();
    for (int it = 0; it < this->layer_param_.top_size(); it++) {
      const string& top_name = this->layer_param_.top(it);
      const int* shape = NULL;
      int shape_count = 0;
      example->GetBlobShape(top_name.c_str(), &shape, &shape_count);
      vector<int> shape_vec(shape, shape + shape_count);
      shape_vec.insert(shape_vec.begin(), batch_size);
      top[it]->Reshape(shape_vec);
      for (int ip = 0; ip < this->PREFETCH_COUNT; ip++) {
        this->prefetch_[ip].data_[it]->Reshape(shape_vec);
      }
    }
  }

  // This function is called on prefetch thread.
  template <typename Dtype>
  void ExternalLibDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
    ExternalLibDataParameter external_lib_data_param =
      this->layer_param_.external_lib_data_param();

    // Take one example to be able to take shapes.
    IDatum* example = data_source_->GetCurrent();

    // Go over all batch blobs and reshape them accordingly.
    const int batch_size = external_lib_data_param.batch_size();
    for (int il = 0; il < this->layer_param_.top_size(); il++) {
      // Take blob data from example for blob with the given name.
      const string& top_name = this->layer_param_.top(il);
      const int* shape = NULL;
      int shape_count = 0;
      example->GetBlobShape(top_name.c_str(), &shape, &shape_count);

      vector<int> top_shape(shape, shape + shape_count);
      top_shape.insert(top_shape.begin(), batch_size);
      batch->data_[il]->Reshape(top_shape);
    }

    // Go over the batch and load data.
    for (int item_id = 0; item_id < batch_size; ++item_id) {
      // Take current batch example.
      example = data_source_->GetCurrent();

      // Go over all top blobs.
      for (int il = 0; il < this->layer_param_.top_size(); il++) {
        // Take blob shape from example for blob with the given name.
        const string& top_name = this->layer_param_.top(il);
        const int* shape = NULL;
        int shape_count = 0;
        example->GetBlobShape(top_name.c_str(), &shape, &shape_count);
        vector<int> top_shape(shape, shape + shape_count);
        top_shape.insert(top_shape.begin(), batch_size);
        // Make sure we have example shaped appropriately.
        CHECK(batch->data_[il]->shape() == top_shape);

        // Load example data into memory.
        int datum_size = batch->data_[il]->count(1);
        example->GetBlobData(top_name.c_str(),
          batch->data_[il]->mutable_cpu_data() + item_id * datum_size,
          BlobTypeMap<Dtype>::GetType());
      }

      // Move to the next example.
      data_source_->MoveToNext();
    }
  }

  template <typename Dtype>
  bool ExternalLibDataLayer<Dtype>::PrefetchToGpu(int index) {
    return find(gpu_prefetch_indices_.begin(), gpu_prefetch_indices_.end(),
      index) != gpu_prefetch_indices_.end();
  }

  template <typename Dtype>
  shared_ptr<IExternalLib> ExternalLibDataLayer<Dtype>::GetExternalLib(
    const string& ext_lib_path,
    const string& factory_name,
    const string& ext_lib_param) {
    return GetDataSourceLibraryWrapper(ext_lib_path, factory_name,
      ext_lib_param);
  }

  INSTANTIATE_CLASS(ExternalLibDataLayer);
  REGISTER_LAYER_CLASS(ExternalLibData);

}  // namespace caffe
