#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
DataLayer<Dtype>::DataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
    // Check user override in param file
    if (this->layer_param_.data_param().threads() == 0) {
#ifndef CPU_ONLY
        // In GPU mode, default to # CPU cores / # GPUs
        if (Caffe::mode() == Caffe::GPU) {
            int cuda_devices;
            unsigned int cpu_count;
            unsigned int prefetch_threads;
            cpu_count = boost::thread::hardware_concurrency();
            CUDA_CHECK(cudaGetDeviceCount(&cuda_devices));
            CHECK_GE(cuda_devices, 1) << "No CUDA devices found";
            DLOG(INFO) << "CPU threads: " << cpu_count
                       << " GPU devices visible: " << cuda_devices;
            prefetch_threads = cpu_count > cuda_devices ?
                cpu_count / cuda_devices + (cpu_count % cuda_devices != 0) : 1;
            DLOG(INFO) << "Data processing threads: " << prefetch_threads;
            pool_.reset(new ThreadPool(prefetch_threads));
        } else {
            // In CPU mode, 1 thread seems to be enough
            DLOG(INFO) << "Data processing threads: 1";
            pool_.reset(new ThreadPool(1));
        }
#else
        DLOG(INFO) << "Data processing threads: 1";
        pool_.reset(new ThreadPool(1));
#endif
    } else {
        DLOG(INFO) << "Data processing threads: "
                   << this->layer_param_.data_param().threads();
        pool_.reset(new ThreadPool(this->layer_param_.data_param().threads()));
    }
}

template <typename Dtype>
DataLayer<Dtype>::~DataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  bool use_gpu_transform = this->transform_param_.use_gpu_transform() &&
                           (Caffe::mode() == Caffe::GPU);
  // Read a data point, and use it to initialize the top blob.
  string& str = *(reader_.full().peek());
  // Parse this data point so we can use the datum to infer things
  Datum datum;
  datum.ParseFromString(str);

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum,
                                                   use_gpu_transform);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape = this->data_transformer_->InferBlobShape(datum);
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);

  LOG(INFO) << "ReshapePrefetch " << top_shape[0] << ", " << top_shape[1]
      << ", " << top_shape[2] << ", " << top_shape[3];
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  if (use_gpu_transform) {
    LOG(INFO) << "prefetch data size: " << top_shape[0] << ","
        << top_shape[1] << "," << top_shape[2] << ","
        << top_shape[3];
  }
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void DataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  bool use_gpu_transform = this->transform_param_.use_gpu_transform() &&
                           (Caffe::mode() == Caffe::GPU);

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  // vector to store generated random numbers
  if (use_gpu_transform) {
    vector<int> random_vec_shape_;
    random_vec_shape_.push_back(batch_size * 3);
    batch->random_vec_.Reshape(random_vec_shape_);
  }

  string& str = *(reader_.full().peek());
  Datum datum;
  datum.ParseFromString(str);
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum,
                                                   use_gpu_transform);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
      timer.Start();
      // Get a datum
      string* str = (reader_.full().pop("Waiting for data"));
      read_time += timer.MicroSeconds();
      // Copy label.
      Dtype* label_ptr = NULL;
      if (this->output_labels_) {
          label_ptr = &top_label[item_id];
      }

      // Get data offset for this datum to hand off to transform thread
      int offset = batch->data_.offset(item_id);
      this->transformed_data_.set_cpu_data(top_data + offset);
      Dtype* ptr = this->transformed_data_.mutable_cpu_data();

      // Precalculate the necessary random draws so that they are
      // drawn deterministically
      int rand1 = 0, rand2 = 0, rand3 = 0;
      if (this->transform_param_.mirror()) {
         rand1 = this->data_transformer_->Rand(RAND_MAX)+1;
      }
      if (this->phase_ == TRAIN && this->transform_param_.crop_size()) {
         rand2 = this->data_transformer_->Rand(RAND_MAX)+1;
         rand3 = this->data_transformer_->Rand(RAND_MAX)+1;
      }

      if (use_gpu_transform) {
        // store the generated random numbers and enqueue the copy
        batch->random_vec_.mutable_cpu_data()[item_id*3    ] = rand1;
        batch->random_vec_.mutable_cpu_data()[item_id*3 + 1] = rand2;
        batch->random_vec_.mutable_cpu_data()[item_id*3 + 2] = rand3;
        pool_->runTask(boost::bind(&DataTransformer<Dtype>::CopyPtrEntry,
                                  this->data_transformer_.get(), str, ptr,
                                  this->output_labels_, label_ptr,
                                  &(reader_.free())));
      } else {
        pool_->runTask(boost::bind(&DataTransformer<Dtype>::TransformPtrEntry,
                                  this->data_transformer_.get(), str, ptr,
                                  rand1, rand2, rand3,
                                  this->output_labels_, label_ptr,
                                  &(reader_.free())));
      }
  }
  timer.Stop();

  // Need to make sure we have completed all work before returning or
  // completing timimg
  pool_->waitWorkComplete();
  batch_timer.Stop();

  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe
