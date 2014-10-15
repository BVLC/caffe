#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
void ProtobufDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  caffe::Timer timer;
  timer.Start();
  this->JoinPrefetchThread();
  timer.Stop();
  LOG(INFO) << "prefetch was behind by " << timer.MilliSeconds()
      << " milliseconds";
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
      top.at(0)->mutable_gpu_data());
  if (this->output_labels_) {
    for (int i = 0; i < num_labels_; ++i) {
      caffe_copy(prefetch_labels_.at(i)->count(),
                 prefetch_labels_.at(i)->cpu_data(),
                 top.at(i + 1)->mutable_gpu_data());
    }
    for (int i = 0; i < prefetch_weights_.size(); ++i) {
      caffe_copy(prefetch_weights_.at(i)->count(),
          prefetch_weights_.at(i)->cpu_data(),
          top.at(i + num_labels_ + 1)->mutable_gpu_data());
    }
  }
  // Start a new prefetch thread
  this->CreatePrefetchThread();
}

INSTANTIATE_LAYER_GPU_FORWARD(ProtobufDataLayer);

}  // namespace caffe
