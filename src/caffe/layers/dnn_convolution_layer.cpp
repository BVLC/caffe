#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "dnn.h"

namespace caffe {
static const int print_conversion = 1;

template <typename Dtype>
DnnConvolutionLayer<Dtype>::DnnConvolutionLayer(const LayerParameter& param)
      : ConvolutionLayer<Dtype>(param)
{
}

template <typename Dtype>
void DnnConvolutionLayer<Dtype>::compute_output_shape() {
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
      / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
      / this->stride_w_ + 1;
}

template <typename Dtype>
DnnConvolutionLayer<Dtype>::~DnnConvolutionLayer()
{
    dnnDelete<Dtype>(convolutionFwd);

    dnnDelete<Dtype>(convolutionBwdData);
    dnnLayoutDelete<Dtype>(l_bwdd_top_diff_int);
    dnnLayoutDelete<Dtype>(l_bwdd_bottom_diff_int);
    dnnLayoutDelete<Dtype>(l_bwdd_filter_data_int);
    dnnLayoutDelete<Dtype>(l_bwdd_top_diff_usr);
    dnnLayoutDelete<Dtype>(l_bwdd_bottom_diff_usr);
    dnnLayoutDelete<Dtype>(l_bwdd_filter_data_usr);

    dnnReleaseBuffer<Dtype>(bwdd_top_diff_int);
    dnnReleaseBuffer<Dtype>(bwdd_bottom_diff_int);
    dnnReleaseBuffer<Dtype>(bwdd_filter_data_int);

    dnnDelete<Dtype>(convertBwdData_top);
    dnnDelete<Dtype>(convertBwdData_bottom);
    dnnDelete<Dtype>(convertBwdData_filter);

    dnnDelete<Dtype>(convolutionBwdFilter);
    dnnLayoutDelete<Dtype>(l_bwdf_top_diff_int);
    dnnLayoutDelete<Dtype>(l_bwdf_bottom_data_int);
    dnnLayoutDelete<Dtype>(l_bwdf_filter_diff_int);
    dnnLayoutDelete<Dtype>(l_bwdf_top_diff_usr);
    dnnLayoutDelete<Dtype>(l_bwdf_bottom_data_usr);
    dnnLayoutDelete<Dtype>(l_bwdf_filter_diff_usr);

    dnnReleaseBuffer<Dtype>(bwdf_top_diff_int);
    dnnReleaseBuffer<Dtype>(bwdf_bottom_data_int);
    dnnReleaseBuffer<Dtype>(bwdf_filter_diff_int);

    dnnDelete<Dtype>(convertBwdFilter_top);
    dnnDelete<Dtype>(convertBwdFilter_bottom);
    dnnDelete<Dtype>(convertBwdFilter_filter);

    dnnDelete<Dtype>(convolutionBwdBias);
    dnnLayoutDelete<Dtype>(l_bwdb_top_diff_int);
    dnnLayoutDelete<Dtype>(l_bwdb_bias_diff_int);
    dnnLayoutDelete<Dtype>(l_bwdb_top_diff_usr);
    dnnLayoutDelete<Dtype>(l_bwdb_bias_diff_usr);

    dnnReleaseBuffer<Dtype>(bwdb_top_diff_int);
    dnnReleaseBuffer<Dtype>(bwdb_bias_diff_int);

    dnnDelete<Dtype>(convertBwdBias_top);
    dnnDelete<Dtype>(convertBwdBias_bias);
}

template <typename Dtype>
void DnnConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  this->width_ = bottom[0]->width(); this->height_ = bottom[0]->height(); this->num_ = bottom[0]->num();
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
  int    inputOffset[2] = {-this->pad_w_, -this->pad_h_};

  int is_Alexnet_pcl = 0;
  is_Alexnet_pcl =
   (   (ic ==   3 && iw == 227 && ih == 227 && oc ==   96 && kw == 11 && kh == 11
        && convolutionStrides[0] == 4 && convolutionStrides[1] == 4 &&  this->pad_w_ ==  0 && this->pad_h_ == 0 && g == 1)
    || (ic ==  96 && iw ==  27 && ih ==  27 && oc ==  256 && kw ==  5 && kh ==  5
        && convolutionStrides[0] == 1 && convolutionStrides[1] == 1 &&  this->pad_w_ ==  2 && this->pad_h_ == 2 && g == 2)
    || (ic == 256 && iw ==  13 && ih ==  13 && oc ==  384 && kw ==  3 && kh ==  3
        && convolutionStrides[0] == 1 && convolutionStrides[1] == 1 &&  this->pad_w_ ==  1 && this->pad_h_ == 1 && g == 1)
    || (ic == 384 && iw ==  13 && ih ==  13 && oc ==  384 && kw ==  3 && kh ==  3
        && convolutionStrides[0] == 1 && convolutionStrides[1] == 1 &&  this->pad_w_ ==  1 && this->pad_h_ == 1 && g == 2)
    || (ic == 384 && iw ==  13 && ih ==  13 && oc ==  256 && kw ==  3 && kh ==  3
        && convolutionStrides[0] == 1 && convolutionStrides[1] == 1 &&  this->pad_w_ ==  1 && this->pad_h_ == 1 && g == 2)
   );

  if (g > 1)
  {
    status = dnnGroupsConvolutionCreateForwardBias<Dtype>(
      &convolutionFwd,
      dnnAlgorithmConvolutionDirect,
      g,
      dimension,
      bdata_sizes,
      tdata_sizes,
      fdata_sizes,
      convolutionStrides,
      inputOffset,
      dnnBorderZeros);
  } else
  {
    status = dnnConvolutionCreateForwardBias<Dtype>(
      &convolutionFwd,
      dnnAlgorithmConvolutionDirect,
      dimension,
      bdata_sizes,
      tdata_sizes,
      fdata_sizes,
      convolutionStrides,
      inputOffset,
      dnnBorderZeros);
  }
  CHECK(status == 0) << "Failed dnnCreateConvolution<Dtype>(dnnForward) with status " << status << "\n"  ;

  // TODO: fix error messages!
  status = dnnLayoutCreateFromPrimitive<Dtype>(&fwd_bottom_data.layout_int, convolutionFwd, dnnResourceSrc);
  CHECK(status == 0) << "Failed dnnLayoutCreateFromPrimitive<Dtype>(l_fwd_bottom_data_int, ...) with status " << status << "\n";
  status = dnnLayoutCreateFromPrimitive<Dtype>(&fwd_top_data.layout_int, convolutionFwd, dnnResourceDst);
  CHECK(status == 0) << "Failed dnnLayoutCreateFromPrimitive<Dtype>(l_fwd_top_data_int, ...) with status " << status << "\n";
  status = dnnLayoutCreateFromPrimitive<Dtype>(&fwd_filter_data.layout_int, convolutionFwd, dnnResourceFilter);
  CHECK(status == 0) << "Failed dnnLayoutCreateFromPrimitive<Dtype>(l_fwd_filter_data_int, ...) with status " << status << "\n";
  status = dnnLayoutCreateFromPrimitive<Dtype>(&fwd_bias_data.layout_int, convolutionFwd, dnnResourceBias);
  CHECK(status == 0) << "Failed dnnLayoutCreateFromPrimitive<Dtype>(l_fwd_bias_data_int, ...) with status " << status << "\n";

  status = dnnLayoutCreate<Dtype>(&fwd_bottom_data.layout_usr, dimension, bdata_sizes, bdata_strides);
  CHECK(status == 0) << "Failed creation of l_fwd_bottom_data_usr layout with status " << status << "\n";
  status = dnnLayoutCreate<Dtype>(&fwd_top_data.layout_usr   , dimension, tdata_sizes, tdata_strides);
  CHECK(status == 0) << "Failed creation of l_fwd_top_data_usr layout with status " << status << "\n";
  status = dnnLayoutCreate<Dtype>(&fwd_filter_data.layout_usr, dimension, fdata_sizes, fdata_strides);
  CHECK(status == 0) << "Failed creation of l_fwd_filter_data_usr layout with status " << status << "\n";
  status = dnnLayoutCreate<Dtype>(&fwd_bias_data.layout_usr  ,         1, bias_sizes , bias_strides );
  CHECK(status == 0) << "Failed creation of l_fwd_bias_data_usr layout with status " << status << "\n";

  fwd_bottom_data.create_conversions();
  fwd_top_data.create_conversions();
  fwd_filter_data.create_conversions();
  fwd_bias_data.create_conversions();

/*
 * Backward by data layer setup
 */
  convertBwdData_bottom = NULL;
  convertBwdData_filter = NULL;
  convertBwdData_top    = NULL;
  bwdd_bottom_diff_int = bwdd_top_diff_int = bwdd_filter_data_int = NULL;
  l_bwdd_bottom_diff_int = l_bwdd_top_diff_int = l_bwdd_filter_data_int = NULL;
  l_bwdd_bottom_diff_usr = l_bwdd_top_diff_usr = l_bwdd_filter_data_usr = NULL;

  if (g > 1)
  {
    status = dnnGroupsConvolutionCreateBackwardData<Dtype>(
      &convolutionBwdData,
      dnnAlgorithmConvolutionDirect,
      g,
      dimension,
      bdata_sizes,
      tdata_sizes,
      fdata_sizes,
      convolutionStrides,
      inputOffset,
      dnnBorderZeros);
  } else
  {
    status = dnnConvolutionCreateBackwardData<Dtype>(
      &convolutionBwdData,
      dnnAlgorithmConvolutionDirect,
      dimension,
      bdata_sizes,
      tdata_sizes,
      fdata_sizes,
      convolutionStrides,
      inputOffset,
      dnnBorderZeros);
  }
  CHECK(status == 0) << "Failed dnnCreateConvolution<Dtype>(dnnBackwardData) with status " << status << "\n";

  status = dnnLayoutCreateFromPrimitive<Dtype>(&l_bwdd_bottom_diff_int, convolutionBwdData, dnnResourceDiffSrc);
  CHECK(status == 0) << "Failed dnnLayoutCreateFromPrimitive<Dtype>(l_bwdd_bottom_diff_int, ...) with status " << status << "\n";
  status = dnnLayoutCreateFromPrimitive<Dtype>(&l_bwdd_top_diff_int   , convolutionBwdData, dnnResourceDiffDst);
  CHECK(status == 0) << "Failed dnnLayoutCreateFromPrimitive<Dtype>(l_bwdd_top_diff_int, ...) with status " << status << "\n";
  status = dnnLayoutCreateFromPrimitive<Dtype>(&l_bwdd_filter_data_int, convolutionBwdData, dnnResourceFilter);
  CHECK(status == 0) << "Failed dnnLayoutCreateFromPrimitive<Dtype>(fl_bwdd_filter_data_int, ...) with status " << status << "\n";
  if (!is_Alexnet_pcl)
  {
    status = dnnLayoutCreate<Dtype>(&l_bwdd_bottom_diff_usr, dimension, bdata_sizes, bdata_strides);
    CHECK(status == 0) << "Failed creation of l_bwdd_bottom_diff_usr with status " << status << "\n";
    status = dnnLayoutCreate<Dtype>(&l_bwdd_top_diff_usr   , dimension, tdata_sizes, tdata_strides);
    CHECK(status == 0) << "Failed creation of l_bwdd_top_diff_usr with status " << status << "\n";
    status = dnnLayoutCreate<Dtype>(&l_bwdd_filter_data_usr, dimension, fdata_sizes, fdata_strides);
    CHECK(status == 0) << "Failed creation of l_bwdd_filter_data_usr with status " << status << "\n";
  } else
  {
    status = dnnLayoutPCLCreate<Dtype>(&l_bwdd_bottom_diff_usr, dimension, bdata_sizes);
    CHECK(status == 0) << "Failed creation of l_bwdd_bottom_diff_usr with status " << status << "\n";
  }

  if (!dnnLayoutCompare<Dtype>(l_bwdd_bottom_diff_usr , l_bwdd_bottom_diff_int))
  {
      status = dnnConversionCreate<Dtype>(&convertBwdData_bottom, l_bwdd_bottom_diff_int, l_bwdd_bottom_diff_usr);
      CHECK(status == 0) << "Failed creation convertBwdData_bottom with status " << status << "\n";
      status = dnnAllocateBuffer<Dtype>((void **)&bwdd_bottom_diff_int, l_bwdd_bottom_diff_int);
      CHECK(status == 0) << "Failed bwdd_bottom_diff_int memory allocation with status " << status << "\n";
  }

  if (!is_Alexnet_pcl  && !dnnLayoutCompare<Dtype>(l_bwdd_top_diff_usr , l_bwdd_top_diff_int))
  {
      status = dnnConversionCreate<Dtype>(&convertBwdData_top, l_bwdd_top_diff_usr , l_bwdd_top_diff_int);
      CHECK(status == 0) << "Failed creation convertBwdData_top with status " << status << "\n";
      status = dnnAllocateBuffer<Dtype>((void **)&bwdd_top_diff_int, l_bwdd_top_diff_int);
      CHECK(status == 0) << "Failed bwdd_top_diff_int memory allocation with status " << status << "\n";
  }

  if (!dnnLayoutCompare<Dtype>((!is_Alexnet_pcl)? l_bwdd_filter_data_usr : fwd_filter_data.layout_int, l_bwdd_filter_data_int))
  {
      status = dnnConversionCreate<Dtype>(&convertBwdData_filter, (!is_Alexnet_pcl) ? l_bwdd_filter_data_usr : fwd_filter_data.layout_int , l_bwdd_filter_data_int);
      CHECK(status == 0) << "Failed creation convertBwdData_filter with status " << status << "\n";
      status = dnnAllocateBuffer<Dtype>((void **)&bwdd_filter_data_int, l_bwdd_filter_data_int);
      CHECK(status == 0) << "Failed l_bwdd_filter_data_int memory allocation with status " << status << "\n";
  }

/*
 * Backward by filter layer setup
 */
  convertBwdFilter_bottom = NULL;
  convertBwdFilter_filter = NULL;
  convertBwdFilter_top    = NULL;
  bwdf_top_diff_int = bwdf_bottom_data_int = bwdf_filter_diff_int = NULL;
  l_bwdf_top_diff_int = l_bwdf_bottom_data_int = l_bwdf_filter_diff_int = NULL;
  l_bwdf_top_diff_usr = l_bwdf_bottom_data_usr = l_bwdf_filter_diff_usr = NULL;

  if (g > 1)
  {
    status = dnnGroupsConvolutionCreateBackwardFilter<Dtype>(
      &convolutionBwdFilter,
      dnnAlgorithmConvolutionDirect,
      g,
      dimension,
      bdata_sizes,
      tdata_sizes,
      fdata_sizes,
      convolutionStrides,
      inputOffset,
      dnnBorderZeros);
  } else
  {
    status = dnnConvolutionCreateBackwardFilter<Dtype>(
      &convolutionBwdFilter,
      dnnAlgorithmConvolutionDirect,
      dimension,
      bdata_sizes,
      tdata_sizes,
      fdata_sizes,
      convolutionStrides,
      inputOffset,
      dnnBorderZeros);
  }
  CHECK(status == 0) << "Failed dnnCreateConvolution<Dtype>(dnnBackwardFilter) with status " << status << "\n";

  status = dnnLayoutCreateFromPrimitive<Dtype>(&l_bwdf_bottom_data_int, convolutionBwdFilter, dnnResourceSrc);
  CHECK(status == 0) << "Failed dnnLayoutCreateFromPrimitive<Dtype>(l_bwdf_bottom_data_int, ...) with status " << status << "\n";
  status = dnnLayoutCreateFromPrimitive<Dtype>(&l_bwdf_top_diff_int   , convolutionBwdFilter, dnnResourceDiffDst);
  CHECK(status == 0) << "Failed dnnLayoutCreateFromPrimitive<Dtype>(l_bwdf_top_diff_int, ...) with status " << status << "\n";
  status = dnnLayoutCreateFromPrimitive<Dtype>(&l_bwdf_filter_diff_int, convolutionBwdFilter, dnnResourceDiffFilter);
  CHECK(status == 0) << "Failed dnnLayoutCreateFromPrimitive<Dtype>(l_bwdf_filter_diff_int, ...) with status " << status << "\n";

  if (!is_Alexnet_pcl)
  {
    status = dnnLayoutCreate<Dtype>(&l_bwdf_bottom_data_usr, dimension, bdata_sizes, bdata_strides);
    CHECK(status == 0) << "Failed creation of l_bwdf_bottom_data_usr with status " << status << "\n";
    status = dnnLayoutCreate<Dtype>(&l_bwdf_top_diff_usr   , dimension, tdata_sizes, tdata_strides);
    CHECK(status == 0) << "Failed creation of l_bwdf_top_diff_usr with status " << status << "\n";
    status = dnnLayoutCreate<Dtype>(&l_bwdf_filter_diff_usr, dimension, fdata_sizes, fdata_strides);
    CHECK(status == 0) << "Failed creation of l_bwdf_filter_diff_usr with status " << status << "\n";
  } else
  {
    if (ic != 3)
      status = dnnLayoutPCLCreate<Dtype>(&l_bwdf_bottom_data_usr, dimension, bdata_sizes);
    else
      status = dnnLayoutCreate<Dtype>(&l_bwdf_bottom_data_usr, dimension, bdata_sizes, bdata_strides);
    CHECK(status == 0) << "Failed creation of l_bwdf_bottom_data_usr with status " << status << "\n";
  }

  if (!dnnLayoutCompare<Dtype>(l_bwdf_bottom_data_usr , l_bwdf_bottom_data_int))
  {
      status = dnnConversionCreate<Dtype>(&convertBwdFilter_bottom, l_bwdf_bottom_data_usr, l_bwdf_bottom_data_int);
      CHECK(status == 0) << "Failed creation convertBwdFilter_bottom with status " << status << "\n";
      status = dnnAllocateBuffer<Dtype>((void **)&bwdf_bottom_data_int, l_bwdf_bottom_data_int);
      CHECK(status == 0) << "Failed bwdf_bottom_data_int memory allocation with status " << status << "\n";
  }

  if (!is_Alexnet_pcl && !dnnLayoutCompare<Dtype>(l_bwdf_top_diff_usr , l_bwdf_top_diff_int))
  {
      status = dnnConversionCreate<Dtype>(&convertBwdFilter_top, l_bwdf_top_diff_usr , l_bwdf_top_diff_int);
      CHECK(status == 0) << "Failed creation convertBwdFilter_top with status " << status << "\n";
      status = dnnAllocateBuffer<Dtype>((void **)&bwdf_top_diff_int, l_bwdf_top_diff_int);
      CHECK(status == 0) << "Failed bwdf_top_diff_int memory allocation with status " << status << "\n";
  }

  if (!is_Alexnet_pcl && !dnnLayoutCompare<Dtype>(l_bwdf_filter_diff_usr , l_bwdf_filter_diff_int))
  {
      status = dnnConversionCreate<Dtype>(&convertBwdFilter_filter, l_bwdf_filter_diff_int, l_bwdf_filter_diff_usr);
      CHECK(status == 0) << "Failed creation convertBwdFilter_filter with status " << status << "\n";
      status = dnnAllocateBuffer<Dtype>((void **)&bwdf_filter_diff_int, l_bwdf_filter_diff_int);
      CHECK(status == 0) << "Failed bwdf_filter_diff_int memory allocation with status " << status << "\n";
  }

/*
 * Backward by bias layer setup
 */
  convertBwdBias_bias = NULL;
  convertBwdBias_top  = NULL;
  bwdb_top_diff_int = bwdb_bias_diff_int = NULL;
  l_bwdb_top_diff_int = l_bwdb_bias_diff_int = NULL;
  l_bwdb_top_diff_usr = l_bwdb_bias_diff_usr = NULL;

  if (g > 1)
  {
    status = dnnGroupsConvolutionCreateBackwardBias<Dtype>(
      &convolutionBwdBias,
      dnnAlgorithmConvolutionDirect,
      g,
      dimension,
      tdata_sizes);
  } else
  {
    status = dnnConvolutionCreateBackwardBias<Dtype>(
      &convolutionBwdBias,
      dnnAlgorithmConvolutionDirect,
      dimension,
      tdata_sizes);
  }
  CHECK(status == 0) << "Failed dnnCreateConvolution<Dtype>(dnnBackwardBias) with status " << status << "\n"  ;

  status = dnnLayoutCreateFromPrimitive<Dtype>(&l_bwdb_top_diff_int , convolutionBwdBias, dnnResourceDiffDst);
  CHECK(status == 0) << "Failed dnnLayoutCreateFromPrimitive<Dtype>(l_bwdb_top_diff_int, ...) with status " << status << "\n";
  status = dnnLayoutCreateFromPrimitive<Dtype>(&l_bwdb_bias_diff_int, convolutionBwdBias, dnnResourceDiffBias);
  CHECK(status == 0) << "Failed dnnLayoutCreateFromPrimitive<Dtype>(l_bwdb_bias_diff_int, ...) with status " << status << "\n";

  if (!is_Alexnet_pcl)
  {
    status = dnnLayoutCreate<Dtype>(&l_bwdb_top_diff_usr , dimension, tdata_sizes, tdata_strides);
    CHECK(status == 0) << "Failed creation of l_bwdb_top_diff_usr with status " << status << "\n";
    status = dnnLayoutCreate<Dtype>(&l_bwdb_bias_diff_usr,         1, bias_sizes , bias_strides );
    CHECK(status == 0) << "Failed creation of l_bwdb_filter_diff_usr with status " << status << "\n";
  }

  if (!is_Alexnet_pcl && !dnnLayoutCompare<Dtype>(l_bwdb_top_diff_usr , l_bwdb_top_diff_int))
  {
      status = dnnConversionCreate<Dtype>(&convertBwdBias_top, l_bwdb_top_diff_usr , l_bwdb_top_diff_int);
      CHECK(status == 0) << "Failed creation convertBwdBias_top with status " << status << "\n";
      status = dnnAllocateBuffer<Dtype>((void **)&bwdb_top_diff_int, l_bwdb_top_diff_int);
      CHECK(status == 0) << "Failed bwdb_top_diff_int memory allocation with status " << status << "\n";
  }

  if (!is_Alexnet_pcl && !dnnLayoutCompare<Dtype>(l_bwdb_bias_diff_usr , l_bwdb_bias_diff_int))
  {
      status = dnnConversionCreate<Dtype>(&convertBwdBias_bias, l_bwdb_bias_diff_int , l_bwdb_bias_diff_usr);
      CHECK(status == 0) << "Failed creation convertBwdBias_bias with status " << status << "\n";
      status = dnnAllocateBuffer<Dtype>((void **)&bwdb_bias_diff_int, l_bwdb_bias_diff_int);
      CHECK(status == 0) << "Failed bwdb_bias_diff_int memory allocation with status " << status << "\n";
  }
}

template <typename Dtype>
void MklDnnMemoryDescriptor<Dtype>::convert_from_prv(void* prv_ptr, void* cpu_ptr, void* prv_descriptor)
{
  CHECK(prv_ptr);
  CHECK(cpu_ptr);
  CHECK(prv_descriptor);

  int status;
  void *convert_resources[dnnResourceNumber];
  MklDnnMemoryDescriptor<Dtype>* mem_descr = (MklDnnMemoryDescriptor<Dtype>*) prv_descriptor;

  if(print_conversion) LOG(INFO) << "== convert_from_prv () ==";

  convert_resources[dnnResourceFrom] = (void *)prv_ptr;
  convert_resources[dnnResourceTo]   = (void *)cpu_ptr;
  status = dnnExecute<Dtype>(mem_descr->convert_from_int, convert_resources);
  CHECK(status == 0) << "[8] | Forward conv failed with status " << status;
}

template <typename Dtype>
void DnnConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
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
        bottom[0]->num()      == n) << "Inclompatible shape of bottom with layer";

  ow = this->width_out_;
  oh = this->height_out_;
  oc = this->num_output_/g;
  CHECK(top[0]->width()    == ow &&
        top[0]->height()   == oh &&
        top[0]->channels() == oc*g &&
        top[0]->num()      == n) << "Inclompatible shape of bottom with layer";

  void *convert_resources[dnnResourceNumber];
  void *res_convolutionFwd[dnnResourceNumber];

  if (fwd_bottom_data.convert_to_int)
  {
    const Dtype* prv_bottom = bottom[0]->prv_data();
    if(prv_bottom == NULL)
    {
      if(print_conversion) LOG(INFO) << "convertFwd_bottom for " << this->layer_param_.name();
      convert_resources[dnnResourceFrom] = (void *)bottom[0]->cpu_data();
      convert_resources[dnnResourceTo]   = (void *)fwd_bottom_data.internal_ptr;
      status = dnnExecute<Dtype>(fwd_bottom_data.convert_to_int, convert_resources);
      CHECK(status == 0) << "[4] | Forward conv failed with status " << status;
      res_convolutionFwd[dnnResourceSrc] = (void *)fwd_bottom_data.internal_ptr;

      bottom[0]->set_prv_data(fwd_bottom_data.internal_ptr, true);
    }
    else
    {
      MklDnnMemoryDescriptor<Dtype>* desc = (MklDnnMemoryDescriptor<Dtype>* ) bottom[0]->get_prv_descriptor_data();

      if(!dnnLayoutCompare<Dtype>(desc->layout_int , fwd_bottom_data.layout_int))
      {
        if(print_conversion) LOG(INFO) << " convertFwd_bottom (PRV but different layout) for " << this->layer_param_.name();

        dnnPrimitive_t convert_padding;
        status = dnnConversionCreate<Dtype>(&convert_padding, desc->layout_int , fwd_bottom_data.layout_int);

        convert_resources[dnnResourceFrom] = (void *)bottom[0]->prv_data();
        convert_resources[dnnResourceTo]   = (void *)fwd_bottom_data.internal_ptr;
        status = dnnExecute<Dtype>(convert_padding, convert_resources);
        CHECK(status == 0) << "[4] | Forward conv failed with status " << status;
        dnnDelete<Dtype>(convert_padding);
        res_convolutionFwd[dnnResourceSrc] = (void *)fwd_bottom_data.internal_ptr;

        bottom[0]->set_prv_data(fwd_bottom_data.internal_ptr, true);
        top[0]->set_prv_converter_data(&fwd_bottom_data, &MklDnnMemoryDescriptor<Dtype>::convert_from_prv);
      }
    }
  } else
    res_convolutionFwd[dnnResourceSrc] = (void *)bottom[0]->cpu_data();

  if (fwd_filter_data.convert_to_int)
  {
    const Dtype* prv_filter = this->blobs_[0]->prv_data();

    if (prv_filter == NULL)
    {
     // if (print_conversion)
      LOG(INFO) << "converFwd_filter for " << this->layer_param_.name();

      convert_resources[dnnResourceFrom] = (void *)this->blobs_[0]->cpu_data();
      convert_resources[dnnResourceTo]   = (void *)fwd_filter_data.internal_ptr;
      status = dnnExecute<Dtype>(fwd_filter_data.convert_to_int, convert_resources);
      CHECK(status == 0) << "[5] | Forward conv failed with status " << status;
      res_convolutionFwd[dnnResourceFilter] = (void *)fwd_filter_data.internal_ptr;

      this->blobs_[0]->set_prv_data(fwd_filter_data.internal_ptr, true);
    }
    else
    {
      res_convolutionFwd[dnnResourceFilter] = (void *)prv_filter;
    }
  } else {
    res_convolutionFwd[dnnResourceFilter] = (void *)this->blobs_[0]->cpu_data();;
  }

  if (fwd_bias_data.convert_to_int)
  {
    const Dtype* prv_bias = this->blobs_[1]->prv_data();

    if (prv_bias == NULL)
    {
      if(print_conversion) LOG(INFO) << "convertFwd_bias";
      convert_resources[dnnResourceFrom] = (void *) this->blobs_[1]->cpu_data();
      convert_resources[dnnResourceTo]   = (void *)fwd_bias_data.internal_ptr;
      status = dnnExecute<Dtype>(fwd_bias_data.convert_to_int, convert_resources);
      CHECK(status == 0) << "[6] | Forward conv failed with status " << status;
      res_convolutionFwd[dnnResourceBias] = (void *)fwd_bias_data.internal_ptr;

      this->blobs_[1]->set_prv_data(fwd_bias_data.internal_ptr, true);
    }
    else
    {
      res_convolutionFwd[dnnResourceBias] = (void *)prv_bias;
    }
  } else
    res_convolutionFwd[dnnResourceBias] = (void *) this->blobs_[1]->cpu_data();

  if (fwd_top_data.convert_from_int)
  {
    top[0]->set_prv_data(fwd_top_data.internal_ptr, false);
    top[0]->set_prv_converter_data(&fwd_top_data, &MklDnnMemoryDescriptor<Dtype>::convert_from_prv);
    res_convolutionFwd[dnnResourceDst] = (void *)fwd_top_data.internal_ptr;
  }
  else
    res_convolutionFwd[dnnResourceDst] = top[0]->mutable_cpu_data();

  status = dnnExecute<Dtype>( convolutionFwd, res_convolutionFwd);
  CHECK(status == 0) << "[7] | Forward conv failed with status " << status;

}

template <typename Dtype>
void DnnConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
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
        bottom[0]->num()      == n) << "Incompatible shape of bottom with layer";

  ow = this->width_out_;
  oh = this->height_out_;
  oc = this->num_output_/g;
  CHECK(top[0]->width()    == ow &&
        top[0]->height()   == oh &&
        top[0]->channels() == oc*g &&
        top[0]->num()      == n) << "Incompatible shape of bottom with layer";

  if (propagate_down[0])
  {
    void *convert_resources[dnnResourceNumber];
    void *res_convolutionBwdData[dnnResourceNumber];
    Dtype *bwdd_bottom_diff_usr = bottom[0]->mutable_cpu_diff();
    const Dtype *bwdd_top_diff_usr    = top[0]->cpu_diff();
    const Dtype *bwdd_filter_data_usr = this->blobs_[0]->cpu_data();

    if (convertBwdData_top)
    {
if(print_conversion)      LOG(INFO) << "convertBwdData_top for " << this->layer_param_.name();
      convert_resources[dnnResourceFrom] = (void *)bwdd_top_diff_usr;
      convert_resources[dnnResourceTo]   = (void *)bwdd_top_diff_int;
      status = dnnExecute<Dtype>(convertBwdData_top, convert_resources);
      CHECK(status == 0) << "[4] | Backward Data conv failed with status " << status;
      res_convolutionBwdData[dnnResourceDiffDst] = (void *)bwdd_top_diff_int;
    } else
    {
      res_convolutionBwdData[dnnResourceDiffDst] = (void *)bwdd_top_diff_usr;
    }

    if (convertBwdData_filter)
    {
if(print_conversion)      LOG(INFO) << "convertBwdData_filter for " << this->layer_param_.name();
      convert_resources[dnnResourceFrom] = (void *)bwdd_filter_data_usr;
      convert_resources[dnnResourceTo]   = (void *)bwdd_filter_data_int;
      status = dnnExecute<Dtype>(convertBwdData_filter, convert_resources);
      CHECK(status == 0) << "[5] | Backward Data conv failed with status " << status;
      res_convolutionBwdData[dnnResourceFilter] = (void *)bwdd_filter_data_int;
    } else
    {
      res_convolutionBwdData[dnnResourceFilter] = (void *)bwdd_filter_data_usr;
    }

    res_convolutionBwdData[dnnResourceDiffSrc] = convertBwdData_bottom ? (void *)bwdd_bottom_diff_int : (void *)bwdd_bottom_diff_usr;
    status = dnnExecute<Dtype>( convolutionBwdData, res_convolutionBwdData);
    CHECK(status == 0) << "[6] | Backward Data conv failed with status " << status;

    if (convertBwdData_bottom)
    {
      if(print_conversion)      LOG(INFO) << "convertBwdData_bottom for " << this->layer_param_.name();
      convert_resources[dnnResourceFrom] = (void *)bwdd_bottom_diff_int;
      convert_resources[dnnResourceTo]   = (void *)bwdd_bottom_diff_usr;
      status = dnnExecute<Dtype>(convertBwdData_bottom, convert_resources);
      CHECK(status == 0) << "[7] | Backward Data conv failed with status " << status;
    }
  }
  if (this->param_propagate_down(0))
  {
    void *convert_resources[dnnResourceNumber];
    void *res_convolutionBwdFilter[dnnResourceNumber];
    const Dtype *bwdf_bottom_data_usr = bottom[0]->cpu_data();
    const Dtype *bwdf_top_diff_usr    = top[0]->cpu_diff();
    Dtype *bwdf_filter_diff_usr = this->blobs_[0]->mutable_cpu_diff();

    if (convertBwdFilter_top)
    {
if(print_conversion)      LOG(INFO) << "convertBwdData_top for " << this->layer_param_.name();
      convert_resources[dnnResourceFrom] = (void *)bwdf_top_diff_usr;
      convert_resources[dnnResourceTo]   = (void *)bwdf_top_diff_int;
      status = dnnExecute<Dtype>(convertBwdFilter_top, convert_resources);
      CHECK(status == 0) << "[4] | Backward Filter conv failed with status " << status;
      res_convolutionBwdFilter[dnnResourceDiffDst] = (void *)bwdf_top_diff_int;
    } else
    {
      res_convolutionBwdFilter[dnnResourceDiffDst] = (void *)bwdf_top_diff_usr;

    }

    if (convertBwdFilter_bottom)
    {
if(print_conversion)      LOG(INFO) << "convertBwdFilter_bottom for " << this->layer_param_.name();
      convert_resources[dnnResourceFrom] = (void *)bwdf_bottom_data_usr;
      convert_resources[dnnResourceTo]   = (void *)bwdf_bottom_data_int;
      status = dnnExecute<Dtype>(convertBwdFilter_bottom, convert_resources);
      CHECK(status == 0) << "[5] | Backward Filter conv failed with status " << status;
      res_convolutionBwdFilter[dnnResourceSrc] = (void *)bwdf_bottom_data_int;
    } else
    {
      res_convolutionBwdFilter[dnnResourceSrc] = (void *)bwdf_bottom_data_usr;
    }

    res_convolutionBwdFilter[dnnResourceDiffFilter] = convertBwdFilter_filter ? (void *)bwdf_filter_diff_int : (void *)bwdf_filter_diff_usr;
    status = dnnExecute<Dtype>( convolutionBwdFilter, res_convolutionBwdFilter);
    CHECK(status == 0) << "[6] | Backward Filter conv failed with status " << status;

    if (convertBwdFilter_filter)
    {
if(print_conversion)      LOG(INFO) << "convertBwdFilter_filter for " << this->layer_param_.name();
      convert_resources[dnnResourceFrom] = (void *)bwdf_filter_diff_int;
      convert_resources[dnnResourceTo]   = (void *)bwdf_filter_diff_usr;
      status = dnnExecute<Dtype>(convertBwdFilter_filter, convert_resources);
      CHECK(status == 0) << "[7] | Backward Filter conv failed with status " << status;
    }
  }
  if (this->param_propagate_down(1))
  {
    void *convert_resources[dnnResourceNumber];
    void *res_convolutionBwdBias[dnnResourceNumber];
    const Dtype *bwdb_top_diff_usr = top[0]->cpu_diff();
    Dtype *bwdb_bias_diff_usr= this->blobs_[1]->mutable_cpu_diff();

    if (convertBwdBias_top)
    {
if(print_conversion)      LOG(INFO) << "convertBwdBias_top for " << this->layer_param_.name();
      convert_resources[dnnResourceFrom] = (void *)bwdb_top_diff_usr;
      convert_resources[dnnResourceTo]   = (void *)bwdb_top_diff_int;
      status = dnnExecute<Dtype>(convertBwdBias_top, convert_resources);
      CHECK(status == 0) << "[3] | Backward Bias conv failed with status " << status;
      res_convolutionBwdBias[dnnResourceDiffDst] = (void *)bwdb_top_diff_int;
    } else
    {
      res_convolutionBwdBias[dnnResourceDiffDst] = (void *)bwdb_top_diff_usr;
    }

    res_convolutionBwdBias[dnnResourceDiffBias] = convertBwdBias_bias ? (void *)bwdb_bias_diff_int : (void *)bwdb_bias_diff_usr;
    status = dnnExecute<Dtype>( convolutionBwdBias, res_convolutionBwdBias);
    CHECK(status == 0) << "[4] | Backward Bias conv failed with status " << status;

    if (convertBwdBias_bias)
    {
if(print_conversion)      LOG(INFO) << "convertBwdBias_bias for " << this->layer_param_.name();
      convert_resources[dnnResourceFrom] = (void *)bwdb_bias_diff_int;
      convert_resources[dnnResourceTo]   = (void *)bwdb_bias_diff_usr;
      status = dnnExecute<Dtype>(convertBwdBias_bias, convert_resources);
      CHECK(status == 0) << "[5] | Backward Bias conv failed with status " << status;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(DnnConvolutionLayer);
#endif

INSTANTIATE_CLASS(DnnConvolutionLayer);
REGISTER_LAYER_CLASS(DnnConvolution);
}  // namespace caffe
