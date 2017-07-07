/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef MKL2017_SUPPORTED
#include <algorithm>
#include <cstdlib>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/mkl_layers.hpp"
#include "caffe/util/performance.hpp"
#include "mkl_service.h"
#ifdef _OPENMP
#include <omp.h>
#endif


static int getMKLBuildDate() {
  static int build = 0;
  if (build == 0) {
    MKLVersion v;
    mkl_get_version(&v);
    build = atoi(v.Build);
  }
  return build;
}

namespace caffe {
template <typename Dtype>
MKLDeconvolutionLayer<Dtype>::MKLDeconvolutionLayer(
  const LayerParameter& param)
      : DeconvolutionLayer<Dtype>(param),
        fwd_bottom_data(new MKLData<Dtype>()),
        fwd_top_data(new MKLData<Dtype>()),
        fwd_filter_data(new MKLData<Dtype>()),
        fwd_bias_data(new MKLData<Dtype>()),
        convolutionFwd(NULL),
        bwdd_top_diff(new MKLDiff<Dtype>()),
        bwdd_bottom_diff(new MKLDiff<Dtype>()),
        bwdd_filter_data(new MKLData<Dtype>()),
        convolutionBwdData(static_cast<dnnPrimitive_t>(NULL)),
        bwdf_top_diff(new MKLDiff<Dtype>()),
        bwdf_filter_diff(new MKLDiff<Dtype>()),
        bwdf2fwd_filter_diff(new MKLDiff<Dtype>()),
        bwdf_bottom_data(new MKLData<Dtype>()),
        convolutionBwdFilter(static_cast<dnnPrimitive_t>(NULL)),
        bwdb_top_diff(new MKLDiff<Dtype>()),
        bwdb_bias_diff(new MKLDiff<Dtype>()),
        convolutionBwdBias(static_cast<dnnPrimitive_t>(NULL)),
        bwdf_filter_diff_iter(new MKLDiff<Dtype>()),
        bwdb_bias_diff_iter(new MKLDiff<Dtype>()) {
          PERFORMANCE_EVENT_ID_RESET(perf_id_fw_);
          PERFORMANCE_EVENT_ID_RESET(perf_id_bw_);
          PERFORMANCE_EVENT_ID_RESET(perf_id_bw_prop_);
          PERFORMANCE_EVENT_ID_RESET(perf_id_bw_diff_);
          PERFORMANCE_EVENT_ID_RESET(perf_id_bw_bias_);
        }

template <typename Dtype>
void MKLDeconvolutionLayer<Dtype>::compute_output_shape() {
  DeconvolutionLayer<Dtype>::compute_output_shape();
  this->height_out_ = this->stride_h_ * (this->height_ - 1)
      + this->kernel_h_ - 2 * this->pad_h_ ;
  this->width_out_ = this->stride_w_ * (this->width_ - 1)
      + this->kernel_w_ - 2 * this->pad_w_ ;
}

template <typename Dtype>
MKLDeconvolutionLayer<Dtype>::~MKLDeconvolutionLayer() {
    dnnDelete<Dtype>(convolutionFwd);
    dnnDelete<Dtype>(convolutionBwdData);
    dnnDelete<Dtype>(convolutionBwdFilter);
    if (this->bias_term_)
        dnnDelete<Dtype>(convolutionBwdBias);
}

template <typename Dtype>
void MKLDeconvolutionLayer<Dtype>::Init(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

#ifdef _OPENMP
  this->num_of_threads_ = omp_get_max_threads() < bottom[0]->shape(0) ?
                    omp_get_max_threads() : bottom[0]->shape(0);
  if (this->num_of_threads_ < 1) {
     LOG(WARNING) << "DeConv layer: omp_get_max_threads() ="
                  << this->num_of_threads_;
     this->num_of_threads_ = 1;
  }
#endif


  this->width_ = bottom[0]->width();
  this->height_ = bottom[0]->height();
  this->num_ = bottom[0]->num();

  // TODO: clean up this
  kernel_w_ = this->kernel_shape_.cpu_data()[1];
  kernel_h_ = this->kernel_shape_.cpu_data()[0];
  stride_w_ = this->stride_.cpu_data()[1];
  stride_h_ = this->stride_.cpu_data()[0];
  pad_w_ = this->pad_.cpu_data()[1];
  pad_h_ = this->pad_.cpu_data()[0];

  this->bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  int status;
  size_t n, g;
  size_t iw, ih, ic;
  size_t ow, oh, oc;
  size_t kw, kh; /* filter */
  size_t dimension = 4;

  g  = std::max(this->group_, 1);
  n  = this->num_;
  iw = this->width_;
  ih = this->height_;
  ic = this->channels_;

  ow = this->width_out_;
  oh = this->height_out_;
  oc = this->num_output_;

  kw = this->kernel_w_;
  kh = this->kernel_h_;

  size_t bdata_sizes[4] = {iw, ih, ic, n};
  size_t bdata_strides[4] = {1, iw, iw*ih, iw*ih*ic};

  /* starting with MKL 2017 Gold in case of groups filter layout
   * becomes 5D, i.e. groups become a separate dimension */
  size_t g_mkl2017 = g;
  size_t f_dimension = dimension + (g != 1);
  if (getMKLBuildDate() < 20160701) {
      g_mkl2017 = 1;
      f_dimension = dimension;
  }

  size_t fdata_sizes[5] = {kw, kh, oc/g, ic/g_mkl2017, g_mkl2017};
  size_t fdata_strides[5]  = {1, kw, kw*kh, kw*kh*oc/g, kw*kh*ic/g*oc/g};

  size_t bias_sizes[1] = {oc};
  size_t bias_strides[1] = {1};

  size_t tdata_sizes[4] = {ow, oh, oc, n};
  size_t tdata_strides[4]  = {1, ow, ow*oh, ow*oh*oc};

  size_t convolutionStrides[2] = {this->stride_w_, this->stride_h_};
  int    inputOffset[2] = {-this->pad_w_, -this->pad_h_};

  // Names are for debugging purposes only.
  fwd_bottom_data ->name = "fwd_bottom_data   @ " + this->layer_param_.name();
  fwd_top_data    ->name = "fwd_top_data      @ " + this->layer_param_.name();
  fwd_filter_data ->name = "fwd_filter_data   @ " + this->layer_param_.name();
  fwd_bias_data   ->name = "fwd_bias_data     @ " + this->layer_param_.name();
  bwdd_top_diff   ->name = "bwdd_top_diff     @ " + this->layer_param_.name();
  bwdd_bottom_diff->name = "bwdd_bottom_diff  @ " + this->layer_param_.name();
  bwdd_filter_data->name = "bwdd_filter_data  @ " + this->layer_param_.name();
  bwdf_top_diff   ->name = "bwdf_top_diff     @ " + this->layer_param_.name();
  bwdf_bottom_data->name = "bwdf_bottom_data  @ " + this->layer_param_.name();
  bwdf_filter_diff->name = "bwdf_filter_diff  @ " + this->layer_param_.name();
  bwdf2fwd_filter_diff->name =
                       "bwdf2fwd_filter_diff  @ " + this->layer_param_.name();
  bwdb_top_diff   ->name = "bwdb_top_diff     @ " + this->layer_param_.name();
  bwdb_bias_diff  ->name = "bwdb_bias_diff    @ " + this->layer_param_.name();


/*
 * Forward setup, implemented by convolutionBwdData
 */
  dnnDelete<Dtype>(convolutionBwdData);
  status = dnnGroupsConvolutionCreateBackwardData<Dtype>(
    &convolutionBwdData,
    NULL,
    dnnAlgorithmConvolutionDirect,
    g,
    dimension,
    tdata_sizes,
    bdata_sizes,
    fdata_sizes,
    convolutionStrides,
    inputOffset,
    dnnBorderZeros);
  CHECK_EQ(status, 0)
          << "Failed dnnConvolutionCreateBackwardData with status "
          << status << "\n";
  fwd_bottom_data->create_layouts(convolutionBwdData, dnnResourceDiffDst, dimension,
                                  bdata_sizes, bdata_strides);
  fwd_top_data   ->create_layouts(convolutionBwdData, dnnResourceDiffSrc, dimension,
                                  tdata_sizes, tdata_strides);
  fwd_filter_data->create_layouts(convolutionBwdData, dnnResourceFilter,
                                  f_dimension, fdata_sizes, fdata_strides);

/*
 * Backward by Data setup, implemented by  convolutionFwd
 */

  dnnDelete<Dtype>(convolutionFwd);

  status = dnnGroupsConvolutionCreateForward<Dtype>(
          &convolutionFwd,
          NULL,
          dnnAlgorithmConvolutionDirect,
          g,
          dimension,
          tdata_sizes,
          bdata_sizes,
          fdata_sizes,
          convolutionStrides,
          inputOffset,
          dnnBorderZeros);

  CHECK_EQ(status, 0)
          << "Failed dnnCreateConvolution<Dtype>(dnnForward) with status "
          << status << "\n";

  bwdd_bottom_diff->create_layouts(convolutionFwd, dnnResourceDst,
                                   dimension, bdata_sizes, bdata_strides);
  bwdd_top_diff   ->create_layouts(convolutionFwd, dnnResourceSrc,
                                   dimension, tdata_sizes, tdata_strides);
  bwdd_filter_data->create_layouts(convolutionFwd, dnnResourceFilter,
                                   f_dimension, fdata_sizes, fdata_strides);

/*
 * Backward by filter layer setup
 */
  dnnDelete<Dtype>(convolutionBwdFilter);
  status = dnnGroupsConvolutionCreateBackwardFilter<Dtype>(
    &convolutionBwdFilter,
    NULL,
    dnnAlgorithmConvolutionDirect,
    g,
    dimension,
    tdata_sizes,
    bdata_sizes,
    fdata_sizes,
    convolutionStrides,
    inputOffset,
    dnnBorderZeros);
  CHECK_EQ(status, 0)
          << "Failed dnnConvolutionCreateBackwardFilter with status "
          << status << "\n";

  bwdf_bottom_data->create_layouts(convolutionBwdFilter, dnnResourceDiffDst,
                                   dimension, bdata_sizes, bdata_strides);
  bwdf_top_diff   ->create_layouts(convolutionBwdFilter, dnnResourceSrc,
                                   dimension, tdata_sizes, tdata_strides);
  bwdf_filter_diff->create_layouts(convolutionBwdData, dnnResourceFilter,
                                   f_dimension, fdata_sizes, fdata_strides);
  // support for (iter_size > 1) requires additional buffer
  bwdf_filter_diff_iter->create_layouts(convolutionFwd, dnnResourceFilter,
                                   f_dimension, fdata_sizes, fdata_strides);

  // Note: this caused some trouble for older MKL
  if (getMKLBuildDate() > 20160701) {
    // bwdf2fwd_filter_diff:
    // layout_int = internal layout of weight diff
    // layout_usr = internal layout of weight data on forward convolution
    bwdf2fwd_filter_diff->create_internal_layout(convolutionBwdFilter,
        dnnResourceDiffFilter);
    bwdf2fwd_filter_diff->remove_user_layout();
    status = dnnLayoutCreateFromPrimitive<Dtype>(
        &bwdf2fwd_filter_diff->layout_usr, convolutionBwdData, dnnResourceFilter);
    CHECK_EQ(status, 0) << "Failed dnnLayoutCreateFromPrimitive with status "
            << status << "\n";

    bwdf2fwd_filter_diff->create_conversions();
  }

/*
 * Backward by bias layer setup
 */
  if (this->bias_term_) {
    dnnDelete<Dtype>(convolutionBwdBias);
    status = dnnGroupsConvolutionCreateBackwardBias<Dtype>(
      &convolutionBwdBias,
      NULL,
      dnnAlgorithmConvolutionDirect,
      g,
      dimension,
      tdata_sizes);
    CHECK_EQ(status, 0)
            << "Failed dnnConvolutionCreateBackwardBias with status "
            << status << "\n";

    bwdb_top_diff->create_layouts(convolutionBwdBias, dnnResourceDiffDst,
                                  dimension, tdata_sizes, tdata_strides);
    bwdb_bias_diff->create_layouts(convolutionBwdBias, dnnResourceDiffBias,
                                   1, bias_sizes, bias_strides);
    // support for (iter_size > 1) requires additional buffer
    bwdb_bias_diff_iter->create_layouts(convolutionBwdBias, dnnResourceDiffBias,
                                        1, bias_sizes, bias_strides);
  }

}

template <typename Dtype>
void MKLDeconvolutionLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  DeconvolutionLayer<Dtype>::LayerSetUp(bottom, top);

  Init(bottom, top);
}

template <typename Dtype>
void MKLDeconvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  bool reinitialize = (this->width_ == bottom[0]->width() &&
                       this->height_ == bottom[0]->height() &&
                       this->channels_ == bottom[0]->channels() &&
                       this->num_ == bottom[0]->num()) ? false : true;

  BaseConvolutionLayer<Dtype>::ReshapeForMKL(bottom, top);

  if (reinitialize == true) {
    Init(bottom, top);
  }
}

template <typename Dtype>
void MKLDeconvolutionLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int status;
  size_t n, g;
  size_t iw, ih, ic;
  size_t ow, oh, oc;

  g  = this->group_;
  n  = this->num_;
  iw = this->width_;
  ih = this->height_;
  ic = this->channels_/g;

  CHECK(bottom[0]->width()    == iw &&
        bottom[0]->height()   == ih &&
        bottom[0]->channels() == ic*g &&
        bottom[0]->num()      == n)
          << "Inclompatible shape of bottom with layer";

  ow = this->width_out_;
  oh = this->height_out_;
  oc = this->num_output_/g;
  CHECK(top[0]->width()    == ow &&
        top[0]->height()   == oh &&
        top[0]->channels() == oc*g &&
        top[0]->num()      == n) << "Inclompatible shape of bottom with layer";


  void *res_convolutionBwdData[dnnResourceNumber];

  res_convolutionBwdData[dnnResourceDiffDst] =
      fwd_bottom_data->get_converted_prv(bottom[0], false);
  // Currently this conversion adds padding to weights.
  // We don't want that to be stored in the weights prv_ptr_
  res_convolutionBwdData[dnnResourceFilter]  =
      fwd_filter_data->get_converted_prv(this->blobs_[0].get(), true);

  if (fwd_top_data->conversion_needed()) {
      top[0]->set_prv_data_descriptor(fwd_top_data);
      res_convolutionBwdData[dnnResourceDiffSrc] =
          reinterpret_cast<void *>(top[0]->mutable_prv_data());
  } else {
      res_convolutionBwdData[dnnResourceDiffSrc] =
          top[0]->mutable_cpu_data();
  }

  PERFORMANCE_EVENT_ID_INIT(perf_id_fw_, PERFORMANCE_MKL_NAME("FW"));
  PERFORMANCE_MEASUREMENT_BEGIN();
  status = dnnExecute<Dtype>(convolutionBwdData, res_convolutionBwdData);
  PERFORMANCE_MEASUREMENT_END_ID(perf_id_fw_);

  CHECK_EQ(status, 0) << "Forward deconvolution failed with status " << status;

  if (this->bias_term_) {
      const Dtype* bias = this->blobs_[1]->cpu_data();
      Dtype* top_data = top[0]->mutable_cpu_data();

#ifdef _OPENMP
#   pragma omp parallel for num_threads(this->num_of_threads_)
#endif
      for (int n = 0; n < this->num_; ++n) {
          this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
  }
}

template <typename Dtype>
void MKLDeconvolutionLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
  int status;
  size_t n, g;
  size_t iw, ih, ic;
  size_t ow, oh, oc;

  g  = this->group_;
  n  = this->num_;
  iw = this->width_;
  ih = this->height_;
  ic = this->channels_/g;

  CHECK(bottom[0]->width()    == iw &&
        bottom[0]->height()   == ih &&
        bottom[0]->channels() == ic*g &&
        bottom[0]->num()      == n)
          << "Incompatible shape of bottom with layer";

  ow = this->width_out_;
  oh = this->height_out_;
  oc = this->num_output_/g;
  CHECK(top[0]->width()    == ow &&
        top[0]->height()   == oh &&
        top[0]->channels() == oc*g &&
        top[0]->num()      == n) << "Incompatible shape of top with layer";

  if (propagate_down[0]) {

      void *res_convolutionFwd[dnnResourceNumber];
      res_convolutionFwd[dnnResourceSrc] =
          bwdd_top_diff->get_converted_prv(top[0], true);
    // Currently this conversion adds padding to weights.
    // We don't want that to be stored in the weights prv_ptr_
      res_convolutionFwd[dnnResourceFilter] =
          bwdd_filter_data->get_converted_prv(this->blobs_[0].get(), false);

    if (bwdd_bottom_diff->conversion_needed()) {
      bottom[0]->set_prv_diff_descriptor(bwdd_bottom_diff);
      res_convolutionFwd[dnnResourceDst] =
          bottom[0]->mutable_prv_diff();
    } else {
      res_convolutionFwd[dnnResourceDst] =
          bottom[0]->mutable_cpu_diff();
    }
    PERFORMANCE_EVENT_ID_INIT(perf_id_bw_prop_,
        PERFORMANCE_MKL_NAME_DETAILED("BW", "_prop"));
    PERFORMANCE_MEASUREMENT_BEGIN();
    status = dnnExecute<Dtype>(convolutionFwd, res_convolutionFwd);
    PERFORMANCE_MEASUREMENT_END_ID(perf_id_bw_prop_);

    CHECK_EQ(status, 0) << "Backward Data deconv failed with status " << status;
  }

  if (this->param_propagate_down(0)) {
    void *res_convolutionBwdFilter[dnnResourceNumber];
    res_convolutionBwdFilter[dnnResourceDiffDst] =
        bwdf_bottom_data->get_converted_prv(bottom[0], false);

    res_convolutionBwdFilter[dnnResourceSrc] =
            bwdf_top_diff->get_converted_prv(top[0], false);


    if (bwdf_filter_diff->conversion_needed()) {
      this->blobs_[0]->set_prv_diff_descriptor(bwdf_filter_diff);
    }
    if (bwdf2fwd_filter_diff->conversion_needed()) {
      // Different layouts in fwd filters vs bwd diffs
      res_convolutionBwdFilter[dnnResourceDiffFilter] =
              reinterpret_cast<void *>(bwdf2fwd_filter_diff->prv_ptr());
    } else {
      if (Caffe::iter_size() > 1) {
        // if (iter_size > 1) then diffs are accumulated across iterations
        res_convolutionBwdFilter[dnnResourceDiffFilter] =
              bwdf_filter_diff_iter->prv_ptr();
      } else {
        if (bwdf_filter_diff->conversion_needed()) {
          res_convolutionBwdFilter[dnnResourceDiffFilter] =
                this->blobs_[0]->mutable_prv_diff();
        } else {
        res_convolutionBwdFilter[dnnResourceDiffFilter] =
              this->blobs_[0]->mutable_cpu_diff();
        }
      }
    }
    PERFORMANCE_EVENT_ID_INIT(perf_id_bw_, PERFORMANCE_MKL_NAME("BW"));
    PERFORMANCE_MEASUREMENT_BEGIN();
    status = dnnExecute<Dtype>(convolutionBwdFilter, res_convolutionBwdFilter);
    PERFORMANCE_MEASUREMENT_END_ID(perf_id_bw_);

    CHECK_EQ(status, 0) << "Backward Filter conv failed with status " << status;

    if (bwdf2fwd_filter_diff->conversion_needed()) {
      // Different layouts in fwd filters vs bwd diffs
      void *convert_resources[dnnResourceNumber];
      convert_resources[dnnResourceFrom] = bwdf2fwd_filter_diff->prv_ptr();

      if (Caffe::iter_size() > 1) {
        // if (iter_size > 1) then diffs are accumulated across iterations
        convert_resources[dnnResourceTo] =
              bwdf_filter_diff_iter->prv_ptr();
        if (bwdf_filter_diff->conversion_needed())
          DLOG(INFO) << "convert priv => priv  " << bwdf2fwd_filter_diff->name
                     << " => " << bwdf_filter_diff->name;
        else
          DLOG(INFO) << "convert priv =>       " << bwdf2fwd_filter_diff->name
                     << " =>";
      } else {
        if (bwdf_filter_diff->conversion_needed()) {
          convert_resources[dnnResourceTo] =
                this->blobs_[0]->mutable_prv_diff();
          DLOG(INFO) << "convert priv => priv  " << bwdf2fwd_filter_diff->name
                     << " => " << bwdf_filter_diff->name;
        } else {
          convert_resources[dnnResourceTo] =
                this->blobs_[0]->mutable_cpu_diff();
          DLOG(INFO) << "convert priv =>       " << bwdf2fwd_filter_diff->name
                     << " =>";
        }
      }

      PERFORMANCE_EVENT_ID_INIT(perf_id_bw_diff_,
          PERFORMANCE_MKL_NAME_DETAILED("BW", "_diff"));
      PERFORMANCE_MEASUREMENT_BEGIN();
      status = dnnExecute<Dtype>(bwdf2fwd_filter_diff->convert_from_int,
              convert_resources);
      PERFORMANCE_MEASUREMENT_END_ID(perf_id_bw_diff_);

      CHECK_EQ(status, 0) << "Conversion failed with status " << status;
    }

    if (Caffe::iter_size() > 1) {
      // if (iter_size > 1) then diffs are accumulated across iterations
      if (bwdf_filter_diff->conversion_needed()) {
        caffe_axpy<Dtype>((const int)this->blobs_[0]->prv_diff_count(), 1,
              reinterpret_cast<Dtype*>(bwdf_filter_diff_iter->prv_ptr()),
              this->blobs_[0]->mutable_prv_diff());
      } else {
        caffe_axpy<Dtype>((const int)this->blobs_[0]->count(), 1,
              reinterpret_cast<Dtype*>(bwdf_filter_diff_iter->prv_ptr()),
              this->blobs_[0]->mutable_cpu_diff());
      }
    }
  }

  if (this->param_propagate_down(1)) {
    void *res_convolutionBwdBias[dnnResourceNumber];

    res_convolutionBwdBias[dnnResourceDiffDst] =
            bwdb_top_diff->get_converted_prv(top[0], true);
    if (Caffe::iter_size() > 1) {
      // if (iter_size > 1) then diffs are accumulated across iterations
      res_convolutionBwdBias[dnnResourceDiffBias] =
            bwdb_bias_diff_iter->prv_ptr();
    } else {
      if (bwdb_bias_diff->conversion_needed()) {
        this->blobs_[1]->set_prv_diff_descriptor(bwdb_bias_diff);
          res_convolutionBwdBias[dnnResourceDiffBias] =
              reinterpret_cast<void *>(this->blobs_[1]->mutable_prv_diff());

      } else {
        res_convolutionBwdBias[dnnResourceDiffBias] =
            reinterpret_cast<void *>(this->blobs_[1]->mutable_cpu_diff());
      }
    }

    PERFORMANCE_EVENT_ID_INIT(perf_id_bw_bias_,
        PERFORMANCE_MKL_NAME_DETAILED("BW", "_bias"));
    PERFORMANCE_MEASUREMENT_BEGIN();
    status = dnnExecute<Dtype>(convolutionBwdBias, res_convolutionBwdBias);
    PERFORMANCE_MEASUREMENT_END_ID(perf_id_bw_bias_);

    CHECK_EQ(status, 0) << "Backward Bias failed with status " << status;

    if (Caffe::iter_size() > 1) {
      // if (iter_size > 1) then diffs are accumulated across iterations
      if (bwdb_bias_diff->conversion_needed()) {
        caffe_axpy<Dtype>((const int)this->blobs_[1]->prv_diff_count(), 1,
              reinterpret_cast<Dtype*>(bwdb_bias_diff_iter->prv_ptr()),
              this->blobs_[1]->mutable_prv_diff());
      } else {
        caffe_axpy<Dtype>((const int)this->blobs_[1]->count(), 1,
              reinterpret_cast<Dtype*>(bwdb_bias_diff_iter->prv_ptr()),
              this->blobs_[1]->mutable_cpu_diff());
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MKLDeconvolutionLayer);
#else
template <typename Dtype>
void MKLDeconvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {NOT_IMPLEMENTED;}
template <typename Dtype>
void MKLDeconvolutionLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom)
  {NOT_IMPLEMENTED;}
#endif

INSTANTIATE_CLASS(MKLDeconvolutionLayer);
}  // namespace caffe
#endif  // #ifdef MKL2017_SUPPORTED
