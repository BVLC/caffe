#include "caffe/layers/ctc_loss_layer.hpp"
#include <numeric>
namespace caffe {
  template <>
  void CtcLossLayer<double>::Forward_gpu(
      const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top) {
      CAFFE1_NOT_IMPLEMENTED;
  }

  template <typename Dtype>
  void CtcLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      cudaDeviceSynchronize();
      auto options = ctcOptions{};
      options.loc = CTC_GPU;
      CUDA_CHECK(cudaStreamCreate(&(options.stream)));
      options.blank_label = blank_label_;
      int mini_batch = bottom[0]->shape()[1];
      int alphabet_size = alphabet_size_;
      const Dtype* const activations = bottom[0]->gpu_data();
      Dtype* gradients = bottom[0]->mutable_gpu_diff();
      CHECK(gradients != NULL) << "Oops, gradients is null";

      FlattenLabels(bottom[1]);
      size_t size_bytes;
      CHECK_CTC_STATUS(get_workspace_size(label_lengths_.data(),
                      input_lengths_.data(), alphabet_size,
                      mini_batch, options, &size_bytes));
      void* workspace;
      CUDA_CHECK(cudaMalloc(&workspace, size_bytes));
      vector<Dtype> cost(mini_batch);
      CHECK_CTC_STATUS(compute_ctc_loss(activations, gradients,
                       flat_labels_.data(),
                       label_lengths_.data(), input_lengths_.data(),
                       alphabet_size, mini_batch, cost.data(),
                       workspace, options));
      Dtype loss = std::accumulate(cost.begin(), cost.end(), Dtype(0));
      top[0]->mutable_cpu_data()[0] = loss / mini_batch;

      CUDA_CHECK(cudaFree(workspace));
      CUDA_CHECK(cudaStreamDestroy(options.stream));
      CUDA_POST_KERNEL_CHECK;
  }

  template <>
  void CtcLossLayer<double>::Backward_gpu(const vector<Blob<double>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<double>*>& bottom) {
      CAFFE1_NOT_IMPLEMENTED;
  }

  template <typename Dtype>
  void CtcLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      if(propagate_down[0]) {
          cudaDeviceSynchronize();
          caffe_gpu_scal(bottom[0]->count(), top[0]->cpu_diff()[0],
                         bottom[0]->mutable_gpu_diff());
          CUDA_POST_KERNEL_CHECK;
      }
  }
  INSTANTIATE_LAYER_GPU_FUNCS(CtcLossLayer);
}
