#ifdef USE_CUDNN
#include <vector>
#include <chrono>

#include "caffe/layers/cudnn_conv_layer.hpp"
#include "caffe/syncedmem.hpp"

namespace caffe {

__global__ void sync_conv_groups() { }

static inline uint64_t get_current_time_ms() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}
template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    auto begin_ms=get_current_time_ms();

  std::cout<<__LINE__<<" cudnn used size="<<SyncedMemory::get_used_size()<<std::endl;
  const Dtype* weight = this->blobs_[0]->gpu_data();
  std::cout<<__LINE__<<" cudnn used size="<<SyncedMemory::get_used_size()<<std::endl;
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
  std::cout<<__LINE__<<" cudnn used size="<<SyncedMemory::get_used_size()<<std::endl;
    Dtype* top_data = top[i]->mutable_gpu_data();
  std::cout<<__LINE__<<"ptr = "<<top[i]  <<" cudnn used size="<<SyncedMemory::get_used_size()<<std::endl;

    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      // Filters.
      CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], bottom_data + bottom_offset_ * g,
            filter_desc_, weight + this->weight_offset_ * g,
            conv_descs_[i],
            fwd_algo_[i], workspace[g], workspace_fwd_sizes_[i],
            cudnn::dataType<Dtype>::zero,
            top_descs_[i], top_data + top_offset_ * g));

      // Bias.
      if (this->bias_term_) {
        const Dtype* bias_data = this->blobs_[1]->gpu_data();
  std::cout<<__LINE__<<" cudnn used size="<<SyncedMemory::get_used_size()<<std::endl;
        CUDNN_CHECK(cudnnAddTensor(handle_[g],
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_descs_[i], top_data + top_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups<<<1, 1>>>();
  }
    auto end_ms=get_current_time_ms();
//    if(end_ms-begin_ms>10) {
      std::cout<<"process  CuDNNConvolutionLayer ms="<< end_ms-begin_ms<<std::endl;
 //   }
}


INSTANTIATE_LAYER_GPU_FUNCS(CuDNNConvolutionLayer);

}  // namespace caffe
#endif
