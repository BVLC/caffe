#ifdef MKL2017_SUPPORTED
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/mkl_layers.hpp"

// Uncomment to see where the layout conversions are done
// #undef DLOG
#ifndef DLOG
#define DLOG LOG
#endif

namespace caffe {
template <typename Dtype>
MKLConvolutionLayer<Dtype>::MKLConvolutionLayer(
  const LayerParameter& param)
      : ConvolutionLayer<Dtype>(param),
        fwd_bottom_data   (new MKLData<Dtype>()),
        fwd_top_data      (new MKLData<Dtype>()),
        fwd_filter_data   (new MKLData<Dtype>()),
        fwd_bias_data     (new MKLData<Dtype>()),
        bwdd_top_diff     (new MKLDiff<Dtype>()),
        bwdd_bottom_diff  (new MKLDiff<Dtype>()),
        bwdd_filter_data  (new MKLData<Dtype>()),
        bwdf_top_diff     (new MKLDiff<Dtype>()),
        bwdf_filter_diff  (new MKLDiff<Dtype>()),
        bwdf_bottom_data  (new MKLData<Dtype>()),
        bwdb_top_diff     (new MKLDiff<Dtype>()),
        bwdb_bias_diff    (new MKLDiff<Dtype>()) {}

template <typename Dtype>
void MKLConvolutionLayer<Dtype>::compute_output_shape() {
  ConvolutionLayer<Dtype>::compute_output_shape();
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
      / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
      / this->stride_w_ + 1;
}

template <typename Dtype>
MKLConvolutionLayer<Dtype>::~MKLConvolutionLayer() {
    dnnDelete<Dtype>(convolutionFwd);
    dnnDelete<Dtype>(convolutionBwdData);
    dnnDelete<Dtype>(convolutionBwdFilter);
    if (this->bias_term_)
        dnnDelete<Dtype>(convolutionBwdBias);
}

template <typename Dtype>
void MKLConvolutionLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);

  this->width_ = bottom[0]->width();
  this->height_ = bottom[0]->height();
  this->num_ = bottom[0]->num();

  // TODO: clean up this
  kernel_w_ = this->kernel_shape_.cpu_data()[0];
  kernel_h_ = this->kernel_shape_.cpu_data()[1];
  stride_w_ = this->stride_.cpu_data()[0];
  stride_h_ = this->stride_.cpu_data()[1];
  pad_w_ = this->pad_.cpu_data()[0];
  pad_h_ = this->pad_.cpu_data()[1];

  this->bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  int status;
  size_t n, g;
  size_t iw, ih, ic;
  size_t ow, oh, oc;
  size_t kw, kh; /* filter */
  size_t dimension = 4;

  g  = this->group_;
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

  size_t fdata_sizes[4] = {kw, kh, ic/g, oc};
  size_t fdata_strides[4]  = {1, kw, kw*kh, kw*kh*ic/g};

  size_t bias_sizes[1] = {oc};
  size_t bias_strides[1] = {1};

  size_t tdata_sizes[4] = {ow, oh, oc, n};
  size_t tdata_strides[4]  = {1, ow, ow*oh, ow*oh*oc};

  size_t convolutionStrides[2] = {this->stride_w_, this->stride_h_};
  int inputOffset[2] = {-this->pad_w_, -this->pad_h_};
  // LOG(INFO) << "mkl conv layer parameters: \n";
  // LOG(INFO) << g << " " << n << " " << iw << " " << ih << " "
  // <<  ic << " " << ow << " " << oh << " " << oc << " " << kw << " " << kh;

  if (this->bias_term_) {
    status = dnnGroupsConvolutionCreateForwardBias<Dtype>(
      &convolutionFwd,
      NULL,
      dnnAlgorithmConvolutionDirect,
      // dnnAlgorithmConvolutionGemm,
      g,
      dimension,
      bdata_sizes,
      tdata_sizes,
      fdata_sizes,
      convolutionStrides,
      inputOffset,
      dnnBorderZeros);
  } else {
    status = dnnGroupsConvolutionCreateForward<Dtype>(
      &convolutionFwd,
      NULL,
      dnnAlgorithmConvolutionDirect,
      // dnnAlgorithmConvolutionGemm,
      g,
      dimension,
      bdata_sizes,
      tdata_sizes,
      fdata_sizes,
      convolutionStrides,
      inputOffset,
      dnnBorderZeros);
  }

  // LOG(INFO) << "MKL conv layer creating forward(bias) done";
  CHECK_EQ(status, 0)
          << "Failed dnnCreateConvolution<Dtype>(dnnForward) with status "
          << status << "\n";

  status = dnnLayoutCreateFromPrimitive<Dtype>(
          &fwd_bottom_data->layout_int, convolutionFwd, dnnResourceSrc);
  CHECK_EQ(status, 0) << "Failed dnnLayoutCreateFromPrimitive with status "
          << status << "\n";
  status = dnnLayoutCreateFromPrimitive<Dtype>(
          &fwd_top_data->layout_int, convolutionFwd, dnnResourceDst);
  CHECK_EQ(status, 0) << "Failed dnnLayoutCreateFromPrimitive with status "
          << status << "\n";
  status = dnnLayoutCreateFromPrimitive<Dtype>(
          &fwd_filter_data->layout_int, convolutionFwd, dnnResourceFilter);
  CHECK_EQ(status, 0) <<
          "Failed dnnLayoutCreateFromPrimitive with status " << status << "\n";
  status = dnnLayoutCreate<Dtype>(
          &fwd_bottom_data->layout_usr, dimension, bdata_sizes, bdata_strides);
  CHECK_EQ(status, 0)
          << "Failed creation of l_fwd_bottom_data_usr layout with status "
          << status << "\n";
  status = dnnLayoutCreate<Dtype>(
          &fwd_top_data->layout_usr   , dimension, tdata_sizes, tdata_strides);
  CHECK_EQ(status, 0) <<
          "Failed creation of l_fwd_top_data_usr layout with status "
          << status << "\n";
  status = dnnLayoutCreate<Dtype>(
          &fwd_filter_data->layout_usr, dimension, fdata_sizes, fdata_strides);
  CHECK_EQ(status, 0)
          << "Failed creation of l_fwd_filter_data_usr layout with status "
          << status << "\n";

  fwd_bottom_data->create_conversions();
  // std::cout << "creating fwd_top_data conversions.\n";
  fwd_top_data   ->create_conversions();
  // std::cout << "creating fwd_top_data conversions done.\n";
  fwd_filter_data->create_conversions();

  if (this->bias_term_) {
    status = dnnLayoutCreateFromPrimitive<Dtype>(
            &fwd_bias_data->layout_int, convolutionFwd, dnnResourceBias);
    CHECK_EQ(status, 0) << "Failed dnnLayoutCreateFromPrimitive with status "
            << status << "\n";
    status = dnnLayoutCreate<Dtype>(
            &fwd_bias_data->layout_usr, 1, bias_sizes, bias_strides);
    CHECK_EQ(status, 0)
            << "Failed creation of l_fwd_bias_data_usr layout with status "
            << status << "\n";
    fwd_bias_data  ->create_conversions();
  }

/*
 * Backward by data layer setup
 */
  status = dnnGroupsConvolutionCreateBackwardData<Dtype>(
    &convolutionBwdData,
    NULL,
    dnnAlgorithmConvolutionDirect,
    g,
    dimension,
    bdata_sizes,
    tdata_sizes,
    fdata_sizes,
    convolutionStrides,
    inputOffset,
    dnnBorderZeros);
  CHECK_EQ(status, 0)
          << "Failed dnnConvolutionCreateBackwardData with status "
          << status << "\n";

  status = dnnLayoutCreateFromPrimitive<Dtype>(&bwdd_bottom_diff->layout_int,
          convolutionBwdData, dnnResourceDiffSrc);
  CHECK_EQ(status, 0)
          << "Failed dnnLayoutCreateFromPrimitive with status "
          << status << "\n";
  status = dnnLayoutCreateFromPrimitive<Dtype>(
          &bwdd_top_diff->layout_int, convolutionBwdData, dnnResourceDiffDst);
  CHECK_EQ(status, 0) << "Failed dnnLayoutCreateFromPrimitive with status "
          << status << "\n";
  status = dnnLayoutCreateFromPrimitive<Dtype>(&bwdd_filter_data->layout_int,
          convolutionBwdData, dnnResourceFilter);
  CHECK_EQ(status, 0) << "Failed dnnLayoutCreateFromPrimitive with status "
          << status << "\n";
  status = dnnLayoutCreate<Dtype>(&bwdd_bottom_diff->layout_usr, dimension,
          bdata_sizes, bdata_strides);
  CHECK_EQ(status, 0)
          << "Failed creation of bwdd_bottom_diff->layout_usr with status "
          << status << "\n";
  status = dnnLayoutCreate<Dtype>(&bwdd_top_diff->layout_usr, dimension,
          tdata_sizes, tdata_strides);
  CHECK_EQ(status, 0)
          << "Failed creation of bwdd_top_diff->layout_usr with status "
          << status << "\n";
  status = dnnLayoutCreate<Dtype>(&bwdd_filter_data->layout_usr, dimension,
          fdata_sizes, fdata_strides);
  CHECK_EQ(status, 0)
          << "Failed creation of bwdd_filter_data->layout_usr with status "
          << status << "\n";

  bwdd_bottom_diff->create_conversions();
  bwdd_top_diff->create_conversions();
  bwdd_filter_data->create_conversions();

/*
 * Backward by filter layer setup
 */
  status = dnnGroupsConvolutionCreateBackwardFilter<Dtype>(
    &convolutionBwdFilter,
    NULL,
    dnnAlgorithmConvolutionDirect,
    g,
    dimension,
    bdata_sizes,
    tdata_sizes,
    fdata_sizes,
    convolutionStrides,
    inputOffset,
    dnnBorderZeros);
  CHECK_EQ(status, 0)
          << "Failed dnnConvolutionCreateBackwardFilter with status "
          << status << "\n";

  status = dnnLayoutCreateFromPrimitive<Dtype>(&bwdf_bottom_data->layout_int,
          convolutionBwdFilter, dnnResourceSrc);
  CHECK_EQ(status, 0) << "Failed dnnLayoutCreateFromPrimitive with status "
          << status << "\n";
  status = dnnLayoutCreateFromPrimitive<Dtype>(&bwdf_top_diff->layout_int,
          convolutionBwdFilter, dnnResourceDiffDst);
  CHECK_EQ(status, 0) << "Failed dnnLayoutCreateFromPrimitive with status "
          << status << "\n";
  status = dnnLayoutCreateFromPrimitive<Dtype>(&bwdf_filter_diff->layout_int,
          convolutionBwdFilter, dnnResourceDiffFilter);
  CHECK_EQ(status, 0) << "Failed dnnLayoutCreateFromPrimitive with status "
          << status << "\n";

  status = dnnLayoutCreate<Dtype>(&bwdf_bottom_data->layout_usr, dimension,
          bdata_sizes, bdata_strides);
  CHECK_EQ(status, 0)
          << "Failed creation of bwdf_bottom_data->layout_usr with status "
          << status << "\n";
  status = dnnLayoutCreate<Dtype>(&bwdf_top_diff->layout_usr, dimension,
          tdata_sizes, tdata_strides);
  CHECK_EQ(status, 0)
          << "Failed creation of bwdf_top_diff->layout_usr with status "
          << status << "\n";
  status = dnnLayoutCreate<Dtype>(&bwdf_filter_diff->layout_usr, dimension,
          fdata_sizes, fdata_strides);
  CHECK_EQ(status, 0)
          << "Failed creation of bwdf_filter_diff->layout_usr with status "
          << status << "\n";

  bwdf_bottom_data->create_conversions();
  bwdf_top_diff->create_conversions();
  bwdf_filter_diff->create_conversions();

/*
 * Backward by bias layer setup
 */
  if (this->bias_term_) {
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

    status = dnnLayoutCreateFromPrimitive<Dtype>(&bwdb_top_diff->layout_int,
            convolutionBwdBias, dnnResourceDiffDst);
    CHECK_EQ(status, 0) << "Failed dnnLayoutCreateFromPrimitive with status "
            << status << "\n";
    status = dnnLayoutCreateFromPrimitive<Dtype>(&bwdb_bias_diff->layout_int,
            convolutionBwdBias, dnnResourceDiffBias);
    CHECK_EQ(status, 0) << "Failed dnnLayoutCreateFromPrimitive with status "
            << status << "\n";

    status = dnnLayoutCreate<Dtype>(&bwdb_top_diff->layout_usr , dimension,
            tdata_sizes, tdata_strides);
    CHECK_EQ(status, 0)
            << "Failed creation of bwdb_top_diff->layout_usr with status "
            << status << "\n";
    status = dnnLayoutCreate<Dtype>(&bwdb_bias_diff->layout_usr, 1,
            bias_sizes, bias_strides);
    CHECK_EQ(status, 0)
            << "Failed creation of bwdb_bias_diff->layout_usr with status "
            << status << "\n";

    bwdb_top_diff->create_conversions();
    bwdb_bias_diff->create_conversions();
  }

  // Names are for debugging purposes only. TODO: Consider removing this.
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
  bwdb_top_diff   ->name = "bwdb_top_diff     @ " + this->layer_param_.name();
  bwdb_bias_diff  ->name = "bwdb_bias_diff    @ " + this->layer_param_.name();
}

template <typename Dtype, bool is_diff>
void MKLMemoryDescriptor<Dtype, is_diff>::convert_from_prv(void* prv_ptr,
        void* cpu_ptr) {
  CHECK(prv_ptr);
  CHECK(cpu_ptr);
  CHECK(this->convert_from_int);
  int status;
  void *convert_resources[dnnResourceNumber];

  DLOG(INFO) << "convert priv =>           "  << this->name << " =>";

  convert_resources[dnnResourceFrom] = prv_ptr;
  convert_resources[dnnResourceTo]   = cpu_ptr;
  status = dnnExecute<Dtype>(this->convert_from_int, convert_resources);
  CHECK_EQ(status, 0) << "Conversion from prv failed with status " << status;
}

template <typename Dtype, bool is_diff>
Dtype* MKLMemoryDescriptor<Dtype, is_diff>::get_converted_prv(
  Blob<Dtype>* blob, bool set_prv_ptr,
  MKLMemoryDescriptor<Dtype, is_diff>* converted_in_fwd) {
  if (this->convert_to_int) {
    int status;
    void *convert_resources[dnnResourceNumber];
    const Dtype* prv_ptr = is_diff ?  blob->prv_diff() : blob->prv_data();
    if (prv_ptr == NULL) {
      if (converted_in_fwd) {
        // hack for reusing previously done conversion
        // if(dnnLayoutCompare(converted_in_fwd->layout_int , this->layout_int))
        if (1) {
          DLOG(INFO) << "reusing fwd               "
                  << converted_in_fwd->name << " == " << this->name;
          return converted_in_fwd->internal_ptr;
        } else {
          DLOG(INFO) << "layout doesn't match      "
                  << converted_in_fwd->name << " != " << this->name;
        }
      }

      DLOG(INFO) << "convert      => priv                                => "
                 << this->name;

      convert_resources[dnnResourceFrom] =
              is_diff ?
                reinterpret_cast<void *>(const_cast<Dtype*>(blob->cpu_diff()))
              : reinterpret_cast<void *>(const_cast<Dtype*>(blob->cpu_data()));
      convert_resources[dnnResourceTo] =
              reinterpret_cast<void *>(this->internal_ptr);

      status = dnnExecute<Dtype>(this->convert_to_int, convert_resources);
      CHECK_EQ(status, 0) << "Conversion failed with status " << status;

      if (set_prv_ptr) {
        if (is_diff)
          blob->set_prv_diff(this->internal_ptr, get_shared_ptr(), true);
        else
          blob->set_prv_data(this->internal_ptr, get_shared_ptr(), true);
      }
      return this->internal_ptr;
    } else {
      // This section helps if padding needs to be added (or removed...)
      // TODO: consider removing when no longer needed.
      shared_ptr<PrvMemDescr> prv_mem_descriptor =
          is_diff ? (blob->get_prv_descriptor_diff()) :
            (blob->get_prv_descriptor_data());

      CHECK_EQ(prv_mem_descriptor->get_descr_type(),
              PrvMemDescr::PRV_DESCR_MKL2017);

      shared_ptr<MKLMemoryDescriptor<Dtype, is_diff> > current_descr =
        boost::static_pointer_cast<MKLMemoryDescriptor<Dtype, is_diff> >
              (prv_mem_descriptor);

      if (!dnnLayoutCompare<Dtype>(current_descr->layout_int,
              this->layout_int)) {
        if (converted_in_fwd) {
          // hack for reusing previously done conversion
          // if(dnnLayoutCompare(converted_in_fwd->layout_int,this->layout_int))
          if (1) {
            DLOG(INFO) << "reusing fwd               "
                    << converted_in_fwd->name << " == " << this->name;
            return converted_in_fwd->internal_ptr;
          } else {
            DLOG(INFO) << "layout doesn't match      "
                    << converted_in_fwd->name << " != " << this->name;
          }
        }
        DLOG(INFO) << "convert priv => priv      "
                << current_descr->name << " => " << this->name;

        dnnPrimitive_t convert_padding;
        status = dnnConversionCreate<Dtype>(&convert_padding,
                current_descr->layout_int , this->layout_int);
        // CHECK_EQ(status, 0)
        // << "Failed creation convert_padding with status " << status << "\n";
        if (status != 0) {
          // TODO: Very weird that we end up here for conv1. No idea why....
          DLOG(INFO) << "!!!! Failed creation convert_padding with status "
                  << status << "\n";

          convert_resources[dnnResourceFrom] = is_diff ?
            reinterpret_cast<void *>(const_cast<Dtype*>(blob->cpu_diff())) :
            reinterpret_cast<void *>(const_cast<Dtype*>(blob->cpu_data()));
          convert_resources[dnnResourceTo] =
            reinterpret_cast<void*>(this->internal_ptr);

          status = dnnExecute<Dtype>(this->convert_to_int, convert_resources);
          CHECK_EQ(status, 0) << "Conversion failed with status " << status;

        } else {
          convert_resources[dnnResourceFrom] = is_diff ?
            reinterpret_cast<void *>(const_cast<Dtype *>(blob->prv_diff())) :
            reinterpret_cast<void *>(const_cast<Dtype *>(blob->prv_data()));
          convert_resources[dnnResourceTo] =
                  reinterpret_cast<void *>(this->internal_ptr);
          status = dnnExecute<Dtype>(convert_padding, convert_resources);
          CHECK_EQ(status, 0) << "Conversion failed with status " << status;
          dnnDelete<Dtype>(convert_padding);
        }

        if (set_prv_ptr) {
          if (is_diff)
            blob->set_prv_diff(this->internal_ptr, get_shared_ptr(), true);
          else
            blob->set_prv_data(this->internal_ptr, get_shared_ptr(), true);
        }
        return this->internal_ptr;
      } else if (current_descr.get() != this) {
        DLOG(INFO) << "layout OK                 "
                << current_descr->name << " == " << this->name;
      }
    }

    return const_cast<Dtype *>(prv_ptr);
  }

  return (is_diff ? const_cast<Dtype *>(blob->cpu_diff()) :
                    const_cast<Dtype *>(blob->cpu_data()));
}


template <typename Dtype>
void MKLConvolutionLayer<Dtype>::Forward_cpu(
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


  void *res_convolutionFwd[dnnResourceNumber];
  res_convolutionFwd[dnnResourceSrc] =
    fwd_bottom_data->get_converted_prv(bottom[0], false);
  res_convolutionFwd[dnnResourceFilter] =
    fwd_filter_data->get_converted_prv(this->blobs_[0].get(), true);
  if (this->bias_term_) {
    res_convolutionFwd[dnnResourceBias] =
      fwd_bias_data  ->get_converted_prv(this->blobs_[1].get(), true);
  }

  if (fwd_top_data->convert_from_int) {
    top[0]->set_prv_data(fwd_top_data->internal_ptr, fwd_top_data, false);
    res_convolutionFwd[dnnResourceDst] =
            reinterpret_cast<void *>(fwd_top_data->internal_ptr);
  } else {
    res_convolutionFwd[dnnResourceDst] = top[0]->mutable_cpu_data();
  }
  status = dnnExecute<Dtype>(convolutionFwd, res_convolutionFwd);
  // LOG(INFO) << "MKL forward done." ;

  CHECK_EQ(status, 0) << "Forward convolution failed with status " << status;
}

template <typename Dtype>
void MKLConvolutionLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
  // LOG(INFO) << "MKL conv -- " << this->layer_param().name()
  // << " backward start";
  // LOG(INFO) << "Propagate down: " << propagate_down[0];
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
        top[0]->num()      == n) << "Incompatible shape of bottom with layer";


  if (propagate_down[0]) {
    // LOG(INFO) << "propagate_down";
    void *res_convolutionBwdData[dnnResourceNumber];

    res_convolutionBwdData[dnnResourceDiffDst] =
      bwdd_top_diff->get_converted_prv(top[0], true);

    // LOG(INFO) << "set diff dst done";

    // Currently this conversion adds padding to weights.
    // We don't want that to be stored in the weights prv_ptr_
    res_convolutionBwdData[dnnResourceFilter]  =
      bwdd_filter_data->get_converted_prv(this->blobs_[0].get(), false);

    // LOG(INFO) << "set filter done";

    if (bwdd_bottom_diff->convert_from_int) {
      bottom[0]->set_prv_diff(bwdd_bottom_diff->internal_ptr, bwdd_bottom_diff,
              false);
      res_convolutionBwdData[dnnResourceDiffSrc] =
              reinterpret_cast<void *>(bwdd_bottom_diff->internal_ptr);

      // LOG(INFO) << "set diff src done -- 1.";

    } else {
      res_convolutionBwdData[dnnResourceDiffSrc] =
              bottom[0]->mutable_cpu_diff();

      // LOG(INFO) << "set diff src done -- 2.";
    }
    status = dnnExecute<Dtype>(convolutionBwdData, res_convolutionBwdData);
    CHECK_EQ(status, 0) << "Backward Data conv failed with status " << status;

    // LOG(INFO) << "backward bottom done.";
  }

  if (this->param_propagate_down(0)) {
    // LOG(INFO) << "param_propagate_down";

    void *res_convolutionBwdFilter[dnnResourceNumber];

    res_convolutionBwdFilter[dnnResourceDiffDst] =
            bwdf_top_diff->get_converted_prv(top[0], true);
    // The last get_converted_prv() argument is a hack for
    // reusing conversion done already in the forward direction.
    res_convolutionBwdFilter[dnnResourceSrc] =
            bwdf_bottom_data->get_converted_prv(bottom[0], false,
            fwd_bottom_data.get());

    if (bwdf_filter_diff->convert_from_int) {
      this->blobs_[0]->set_prv_diff(bwdf_filter_diff->internal_ptr,
              bwdf_filter_diff, false);
      res_convolutionBwdFilter[dnnResourceDiffFilter] =
              reinterpret_cast<void *>(bwdf_filter_diff->internal_ptr);
    } else {
      res_convolutionBwdFilter[dnnResourceDiffFilter] =
              this->blobs_[0]->mutable_cpu_diff();
    }
    status = dnnExecute<Dtype>(convolutionBwdFilter, res_convolutionBwdFilter);
    CHECK_EQ(status, 0) << "Backward Filter conv failed with status " << status;
  }

  if (this->param_propagate_down(1)) {
    // LOG(INFO) << "param_propagate_down(1)" << std::endl;
    void *res_convolutionBwdBias[dnnResourceNumber];

    res_convolutionBwdBias[dnnResourceDiffDst] =
            bwdb_top_diff->get_converted_prv(top[0], true);

    if (bwdb_bias_diff->convert_from_int) {
      this->blobs_[1]->set_prv_diff(bwdb_bias_diff->internal_ptr,
              bwdb_bias_diff, false);
      res_convolutionBwdBias[dnnResourceDiffBias] =
              bwdb_bias_diff->internal_ptr;
    } else {
      res_convolutionBwdBias[dnnResourceDiffBias] =
              reinterpret_cast<void *>(this->blobs_[1]->mutable_cpu_diff());
    }
    status = dnnExecute<Dtype>(convolutionBwdBias, res_convolutionBwdBias);
    CHECK_EQ(status, 0) << "Backward Bias failed with status " << status;
  }
  // LOG(INFO) << "MKL conv backward done.";
}

#ifdef CPU_ONLY
STUB_GPU(MKLConvolutionLayer);
#else
template <typename Dtype>
void MKLConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {NOT_IMPLEMENTED;}
template <typename Dtype>
void MKLConvolutionLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom)
  {NOT_IMPLEMENTED;}
#endif

INSTANTIATE_CLASS(MKLConvolutionLayer);
}  // namespace caffe
#endif  // #ifdef MKL2017_SUPPORTED
