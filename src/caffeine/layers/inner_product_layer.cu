#include <mkl.h>
#include <cublas_v2.h>

#include "caffeine/blob.hpp"
#include "caffeine/common.hpp"
#include "caffeine/filler.hpp"
#include "caffeine/layer.hpp"
#include "caffeine/vision_layers.hpp"

namespace caffeine {

template <typename Dtype>
void InnerProductLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "IP Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "IP Layer takes a single blob as output.";
	const int num_output = this->layer_param_.num_output();
	const bool gemm_last_dim = this->layer_param_.gemm_last_dim();
	biasterm_ = this->layer_param_.biasterm();
	// Figure out the dimensions
	if (gemm_last_dim) {
		M_ = bottom[0]->count() / bottom[0]->channels();
  	K_ = bottom[0]->channels();
  	N_ = num_output;
  	(*top)[0]->Reshape(bottom[0]->num(), bottom[0]->height(),
  			bottom[0]->width(), num_output);
	} else {
		M_ = bottom[0]->num();
		K_ = bottom[0]->count() / bottom[0]->num();
		N_ = num_output;
		(*top)[0]->Reshape(bottom[0]->num(), 1, 1, num_output);
	}
  if (biasterm_) {
    this->blobs_.resize(2);
  } else {
    this->blobs_.resize(1);
  }
	// Intialize the weight
  this->blobs_[0].Reshape(1, 1, K_, N_);
	// fill the weights
	shared_ptr<Filler<Dtype> > weight_filler(
			GetFiller<Dtype>(this->layer_param_.weight_filler()));
	weight_filler->Fill(&this->blobs_[0]);
	// If necessary, intiialize and fill the bias term
	if (biasterm_) {
    this->blobs_[1].Reshape(1, 1, 1, N_);
		shared_ptr<Filler<Dtype> > bias_filler(
				GetFiller<Dtype>(this->layer_param_.bias_filler()));
		bias_filler->Fill(&this->blobs_[1]);
	}
};

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0].cpu_data();
  const Dtype* bias = NULL;
  if (biasterm_) {
  	bias = this->blobs_[1].cpu_data();
  }
  switch(sizeof(Dtype)) {
  case sizeof(float):
  	// matrix multiply
  	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M_, N_, K_,
  			1., (const float*)bottom_data, K_, (const float*)weight, N_, 0.,
  			(float*)top_data, N_);
  	if (bias) {
  		// add bias
  		for (int i = 0; i < M_; ++i) {
  			cblas_saxpy(N_, 1., (const float*)bias, 1,
  					(float*)(top_data) + (N_ * i), 1);
  		}
  	}
    break;
  case sizeof(double):
    // matrix multiply
  	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M_, N_, K_,
  			1., (const double*)bottom_data, K_, (const double*)weight, N_, 0.,
  			(double*)top_data, N_);
  	if (bias) {
  		// add bias
  		for (int i = 0; i < M_; ++i) {
  			cblas_daxpy(N_, 1., (const double*)bias, 1,
  					(double*)(top_data) + (N_ * i), 1);
  		}
  	}
    break;
  default:
  	CHECK(false) << "Unknown data type.";
  }
}

template <typename Dtype>
Dtype InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  CHECK(false);
  return Dtype(0);
}

template <typename Dtype>
__global__ void BroadcastRow(const int total, const int vec_len,
	const Dtype* in_vec, Dtype* out_matrix) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < total) {
  	int v_index = index % vec_len;
  	out_matrix[index] = in_vec[v_index];
  }
}



template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0].gpu_data();
  const Dtype* bias = NULL;
  Dtype alpha = 1., beta = 0.;
  if (biasterm_) {
  	bias = this->blobs_[1].gpu_data();
  	beta = 1.;
  	const int count = (*top)[0]->count();
  	// we pre-copy the bias to the results, and then call gemm.
  	BroadcastRow<<<CAFFEINE_GET_BLOCKS(count), CAFFEINE_CUDA_NUM_THREADS>>>(
  			count, N_, bias, top_data);
  }
  switch(sizeof(Dtype)) {
  case sizeof(float):
  	// matrix multiply: since cublas uses Fortran major, we actually do
  	// C' = B' A'
  	CUBLAS_CHECK(cublasSgemm(Caffeine::cublas_handle(), CUBLAS_OP_N,
  			CUBLAS_OP_N, N_, M_, K_, (float*)&alpha, (const float*)weight, N_,
  			(const float*)bottom_data, K_, (float*)&beta, (float*)top_data, N_));
    break;
  case sizeof(double):
    // matrix multiply
  	CUBLAS_CHECK(cublasDgemm(Caffeine::cublas_handle(), CUBLAS_OP_N,
  			CUBLAS_OP_N, N_, M_, K_, (double*)&alpha, (const double*)weight, N_,
  			(const double*)bottom_data, K_, (double*)&beta, (double*)top_data, N_));
    break;
  default:
  	CHECK(false) << "Unknown data type.";
  }
}

template <typename Dtype>
Dtype InnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  CHECK(false);
  return Dtype(0.);
}

INSTANTIATE_CLASS(InnerProductLayer);

}  // namespace caffeine
