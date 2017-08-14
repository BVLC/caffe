#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include <glog/logging.h>

namespace caffe {

// CUDA kernele for forward
template <typename Dtype>
__global__ void SReLUForward(const int n, const int channels, const int dim,
    const Dtype* in, Dtype* out, const Dtype negative_slope,
    const Dtype* thresh_data, const Dtype* pslope_data, const Dtype* nslope_data, const Dtype* nthresh_data,
    const int thresh_div_factor, const int pslope_div_factor, const int nslope_div_factor, const int nthresh_div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int pt_c = (index / dim) % channels / thresh_div_factor;
    int ps_c = (index / dim) % channels / pslope_div_factor;
    int ns_c = (index / dim) % channels / nslope_div_factor;
    int nt_c = (index / dim) % channels / nthresh_div_factor;
    if (in[index] >= thresh_data[pt_c])
      out[index] = thresh_data[pt_c] + pslope_data[ps_c] * (in[index] - thresh_data[pt_c]);
    else if (in[index] <= nthresh_data[nt_c])
      out[index] = nthresh_data[nt_c] + nslope_data[ns_c] * (in[index] - nthresh_data[nt_c]);
    else 
      out[index] = in[index];
  }
}

// CUDA kernel for bottom backward
template <typename Dtype>
__global__ void SReLUBackward(const int n, const int channels, const int dim,
    const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff, const Dtype negative_slope,
    const Dtype* thresh_data, const Dtype* pslope_data, const Dtype* nslope_data, const Dtype* nthresh_data,
    const int thresh_div_factor, const int pslope_div_factor, const int nslope_div_factor, const int nthresh_div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int pt_c = (index / dim) % channels / thresh_div_factor;
    int ps_c = (index / dim) % channels / pslope_div_factor;
    int ns_c = (index / dim) % channels / nslope_div_factor;
    int nt_c = (index / dim) % channels / nthresh_div_factor;
    if (in_data[index] >= thresh_data[pt_c])
      out_diff[index] = pslope_data[ps_c] * in_diff[index];
    else if (in_data[index] <= nthresh_data[nt_c])
      out_diff[index] = nslope_data[ns_c] * in_diff[index];
    else
      out_diff[index] = in_diff[index];
  }
}

// CUDA kernel for element-wise parameter backward
template <typename Dtype>
__global__ void SReLUParamBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, const int dim,
    Dtype* thresh_out_diff, Dtype* pslope_out_diff, Dtype* nslope_out_diff, Dtype* nthresh_out_diff,
    const Dtype* thresh_data, const Dtype* pslope_data, const Dtype* nslope_data, const Dtype* nthresh_data, 
    const bool thresh_channel_shared, const bool pslope_channel_shared, const bool nslope_channel_shared, const bool nthresh_channel_shared) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = index / dim;
    Dtype local_thresh  = thresh_channel_shared  ? thresh_data[0]  : thresh_data[c];
    Dtype local_pslope  = pslope_channel_shared  ? pslope_data[0]  : pslope_data[c];
    Dtype local_nslope  = nslope_channel_shared  ? nslope_data[0]  : nslope_data[c];
    Dtype local_nthresh = nthresh_channel_shared ? nthresh_data[0] : nthresh_data[c];
    if (in_data[index] >= local_thresh) {
      thresh_out_diff[index]  = in_diff[index] * (1. - local_pslope);
      pslope_out_diff[index]  = in_diff[index] * (in_data[index] - local_thresh);
      nslope_out_diff[index]  = 0.;
      nthresh_out_diff[index] = 0.;
    } else if(in_data[index] <= local_nthresh){
      nthresh_out_diff[index] = in_diff[index] * (1. - local_nslope);
      nslope_out_diff[index]  = in_diff[index] * (in_data[index] - local_nthresh);
      thresh_out_diff[index]  = 0.;
      pslope_out_diff[index]  = 0.;
    } else {                   
      thresh_out_diff[index]  = 0.;
      pslope_out_diff[index]  = 0.;
      nslope_out_diff[index]  = 0.;
      nthresh_out_diff[index] = 0.;
    }
  }
}

template <typename Dtype>
void SReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->height() * bottom[0]->width();
  const int channels = bottom[0]->channels();
  const Dtype* thresh_data  = this->blobs_[0]->gpu_data();
  const Dtype* pslope_data  = this->blobs_[1]->gpu_data();
  const Dtype* nslope_data  = this->blobs_[2]->gpu_data();
  const Dtype* nthresh_data = this->blobs_[3]->gpu_data();
  const int thresh_div_factor  = thresh_channel_shared_  ? channels : 1;
  const int pslope_div_factor  = pslope_channel_shared_  ? channels : 1;
  const int nslope_div_factor  = nslope_channel_shared_  ? channels : 1;
  const int nthresh_div_factor = nthresh_channel_shared_ ? channels : 1;
  // For in-place computation
  if (top[0] == bottom[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
  }

  // NOLINT_NEXT_LINE(whitespace/operators)
  SReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, channels, dim, bottom_data, top_data, negative_slope_,
      thresh_data, pslope_data, nslope_data, nthresh_data,
      thresh_div_factor, pslope_div_factor, nslope_div_factor, nthresh_div_factor);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void SReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->height() * bottom[0]->width();
  const int channels = bottom[0]->channels();
  const Dtype* thresh_data  = this->blobs_[0]->gpu_data();
  const Dtype* pslope_data  = this->blobs_[1]->gpu_data();
  const Dtype* nslope_data  = this->blobs_[2]->gpu_data();
  const Dtype* nthresh_data = this->blobs_[3]->gpu_data();
  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.gpu_data();
  }

  // Propagate to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
//  if (this->param_propagate_down_[0] && this->param_propagate_down_[1]) {
    Dtype* thresh_diff  = this->blobs_[0]->mutable_gpu_diff();
    Dtype* pslope_diff  = this->blobs_[1]->mutable_gpu_diff();
    Dtype* nslope_diff  = this->blobs_[2]->mutable_gpu_diff();
    Dtype* nthresh_diff = this->blobs_[3]->mutable_gpu_diff();
    int cdim = channels * dim;
    Dtype thresh_dsum = 0., pslope_dsum = 0., nslope_dsum = 0., nthresh_dsum = 0.;
    for (int n = 0; n < bottom[0]->num(); ++n) {
      // compute element-wise diff
      // NOLINT_NEXT_LINE(whitespace/operators)
      SReLUParamBackward<Dtype><<<CAFFE_GET_BLOCKS(cdim),
          CAFFE_CUDA_NUM_THREADS>>>(
          cdim, top_diff + top[0]->offset(n),
          bottom_data + bottom[0]->offset(n), dim,
           thresh_backward_buff_.mutable_gpu_diff(),
           pslope_backward_buff_.mutable_gpu_diff(),
           nslope_backward_buff_.mutable_gpu_diff(),
          nthresh_backward_buff_.mutable_gpu_diff(),
          thresh_data, pslope_data, nslope_data, nthresh_data,
          thresh_channel_shared_, pslope_channel_shared_, nslope_channel_shared_, nthresh_channel_shared_);
      CUDA_POST_KERNEL_CHECK;
      if (thresh_channel_shared_) {
        Dtype d;
        caffe_gpu_dot<Dtype>(channels * dim, thresh_backward_buff_.gpu_diff(),
            multiplier_.gpu_data(), &d);
        thresh_dsum += d;
      } else {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
            thresh_backward_buff_.gpu_diff(), multiplier_.gpu_data(), 1.,
            thresh_diff);
      }
      if (pslope_channel_shared_) {
        Dtype d;
        caffe_gpu_dot<Dtype>(channels * dim, pslope_backward_buff_.gpu_diff(),
            multiplier_.gpu_data(), &d);
        pslope_dsum += d;
      } else {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
            pslope_backward_buff_.gpu_diff(), multiplier_.gpu_data(), 1.,
            pslope_diff);
      }
      if (nslope_channel_shared_) {
        Dtype d;
        caffe_gpu_dot<Dtype>(channels * dim, nslope_backward_buff_.gpu_diff(),
            multiplier_.gpu_data(), &d);
        nslope_dsum += d;
      } else {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
            nslope_backward_buff_.gpu_diff(), multiplier_.gpu_data(), 1.,
            nslope_diff);
      }
      if (nthresh_channel_shared_) {
        Dtype d;
        caffe_gpu_dot<Dtype>(channels * dim, nthresh_backward_buff_.gpu_diff(),
            multiplier_.gpu_data(), &d);
        nthresh_dsum += d;
      } else {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
            nthresh_backward_buff_.gpu_diff(), multiplier_.gpu_data(), 1.,
            nthresh_diff);
      }
    }
    if (thresh_channel_shared_) {
      caffe_gpu_add_scalar(this->blobs_[0]->count(), Dtype(thresh_dsum),  thresh_diff);
    }
    if (pslope_channel_shared_) {
      caffe_gpu_add_scalar(this->blobs_[1]->count(), Dtype(pslope_dsum),  pslope_diff);
    }
    if (nslope_channel_shared_) {
      caffe_gpu_add_scalar(this->blobs_[2]->count(), Dtype(nslope_dsum),  nslope_diff);
    }
    if (nthresh_channel_shared_) {
      caffe_gpu_add_scalar(this->blobs_[3]->count(), Dtype(nthresh_dsum), nthresh_diff);
    }

//  } else {
//    LOG(INFO) << this->param_propagate_down_[0] << " " << this->param_propagate_down_[1];
//    LOG(INFO) << "Learn thresh and slope at the same time, otherwise, unsupported";
//  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    int thresh_div_factor  = thresh_channel_shared_  ? channels : 1;
    int pslope_div_factor  = pslope_channel_shared_  ? channels : 1;
    int nslope_div_factor  = nslope_channel_shared_  ? channels : 1;
    int nthresh_div_factor = nthresh_channel_shared_ ? channels : 1;
    // NOLINT_NEXT_LINE(whitespace/operators)
    SReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
        count, channels, dim, top_diff, bottom_data, bottom_diff, negative_slope_,
        thresh_data, pslope_data, nslope_data, nthresh_data,
        thresh_div_factor, pslope_div_factor, nslope_div_factor, nthresh_div_factor);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SReLULayer);


}  // namespace caffe
