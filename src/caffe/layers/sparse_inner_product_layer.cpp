#include <vector>

#include "caffe/blob.hpp"
#include "caffe/sparse_blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/sparse_inner_product_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void SparseInnerProductLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  SparseBlob<Dtype> * bottomSparseBlob =
        dynamic_cast<SparseBlob<Dtype>*>(bottom[0]);

  if (bottomSparseBlob == 0) {  // fall back to dense computation
    std::cerr << "\nFalling back to dense computation\n";
    InnerProductLayer<Dtype>::Forward_cpu(bottom, top);
    return;
  }
  const Dtype* bottom_data = bottomSparseBlob->cpu_data();
  const int* bottom_indices = bottomSparseBlob->cpu_indices();
  const int* bottom_ptr = bottomSparseBlob->cpu_ptr();
  const int nnz = bottomSparseBlob->nnz();

  //debug
  /*std::cerr << "inner_product sparse bottom_indices size=" << nnz << " / bottom_indices=";
  for (int ind=0;ind<nnz;ind++)
    std::cerr << bottom_indices[ind] << " ";
    std::cerr << std::endl;*/
  //debug

  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();

  caffe_cpu_csr_gemm<Dtype>(CblasNoTrans, CblasTrans, this->M_,
                            this->N_,
                             this->K_, (Dtype) 1., nnz, bottom_data,
                             bottom_indices, bottom_ptr, weight,
                             (Dtype) 0.,
                             top_data, CblasRowMajor);

  if (this->bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->M_, this->N_, 1,
                          (Dtype) 1., this->bias_multiplier_.cpu_data(),
                          this->blobs_[1]->cpu_data(), (Dtype) 1., top_data);
  }
}

template <typename Dtype>
void SparseInnerProductLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  SparseBlob<Dtype> * bottomSparseBlob =
          dynamic_cast<SparseBlob<Dtype>*>(bottom[0]);
  // fall back to dense computation
  if (bottomSparseBlob == 0) {
      InnerProductLayer<Dtype>::Backward_cpu(top, propagate_down, bottom);
      return;
  }
  if (this->param_propagate_down_[0]) {
      // Gradient with respect to weight
      const Dtype* top_diff = top[0]->cpu_diff();
      const Dtype* bottom_data = bottomSparseBlob->cpu_data();
      const int* bottom_indices = bottomSparseBlob->cpu_indices();
      const int* bottom_ptr = bottomSparseBlob->cpu_ptr();
      const int nnz = bottomSparseBlob->nnz();
      caffe_cpu_csr_gemm<Dtype>(CblasTrans, CblasNoTrans, this->K_,
                                this->N_,
                                this->M_, (Dtype) 1., nnz, bottom_data,
                                bottom_indices, bottom_ptr, top_diff,
                                (Dtype) 0.,
                                this->blobs_[0]->mutable_cpu_diff(),
                                CblasColMajor);
    }

    if (this->bias_term_ && this->param_propagate_down_[1]) {
      // Gradient with respect to bias
      const Dtype* top_diff = top[0]->cpu_diff();
      caffe_cpu_gemv<Dtype>(CblasTrans, this->M_, this->N_, (Dtype) 1.,
                            top_diff,
                            this->bias_multiplier_.cpu_data(),
                            (Dtype) 0.,
                            this->blobs_[1]->mutable_cpu_diff());
    }
    if (propagate_down[0]) {
      // there is a bug in the code because this is called no matter what!
      LOG(ERROR) << "propagate down not supported for sparse inner product";
      LOG(FATAL) << "fatal error";
    }
}

#ifdef CPU_ONLY
STUB_GPU(SparseInnerProductLayer);
#endif

INSTANTIATE_CLASS(SparseInnerProductLayer);
REGISTER_LAYER_CLASS(SparseInnerProduct);

}  // namespace caffe


