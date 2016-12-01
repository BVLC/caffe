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

  #ifndef _MKL_DNN_CPPWRAPPER_H
  #define _MKL_DNN_CPPWRAPPER_H

  #include <stdarg.h>
  #include <stddef.h>

  #include "mkl_dnn_types.h"
  #include "mkl_dnn.h"
  #include "mkl_version.h"

  #define TEMPLATE_PREFIX template <typename Dtype> inline
  #define SPEC_PREFIX template <> inline

  #if (__INTEL_MKL__ < 2017) || (__INTEL_MKL_BUILD_DATE <= 20160311)
  #error: To use the new MKL DNN API, you must install Intel(R) MKL 2017 Beta Update 1 or higher.
  #endif



  TEMPLATE_PREFIX dnnError_t dnnLayoutCreate(
          dnnLayout_t *pLayout, size_t dimension, const size_t size[], const size_t strides[]);
  SPEC_PREFIX dnnError_t dnnLayoutCreate<float>(
          dnnLayout_t *pLayout, size_t dimension, const size_t size[], const size_t strides[])
          {return dnnLayoutCreate_F32(pLayout, dimension, size, strides);}
  SPEC_PREFIX dnnError_t dnnLayoutCreate<double>(
          dnnLayout_t *pLayout, size_t dimension, const size_t size[], const size_t strides[])
          {return dnnLayoutCreate_F64(pLayout, dimension, size, strides);}

  TEMPLATE_PREFIX dnnError_t dnnLayoutCreateFromPrimitive(
          dnnLayout_t *pLayout, const dnnPrimitive_t primitive, dnnResourceType_t type);
  SPEC_PREFIX dnnError_t dnnLayoutCreateFromPrimitive<float>(
          dnnLayout_t *pLayout, const dnnPrimitive_t primitive, dnnResourceType_t type)
          {return dnnLayoutCreateFromPrimitive_F32(pLayout, primitive, type);}
  SPEC_PREFIX dnnError_t dnnLayoutCreateFromPrimitive<double>(
          dnnLayout_t *pLayout, const dnnPrimitive_t primitive, dnnResourceType_t type)
          {return dnnLayoutCreateFromPrimitive_F64(pLayout, primitive, type);}

  TEMPLATE_PREFIX size_t dnnLayoutGetMemorySize(
          const dnnLayout_t layout);
  SPEC_PREFIX size_t dnnLayoutGetMemorySize<float>(
          const dnnLayout_t layout)
          {return dnnLayoutGetMemorySize_F32(layout);}
  SPEC_PREFIX size_t dnnLayoutGetMemorySize<double>(
          const dnnLayout_t layout)
          {return dnnLayoutGetMemorySize_F64(layout);}

  TEMPLATE_PREFIX int dnnLayoutCompare(
          const dnnLayout_t l1, const dnnLayout_t l2);
  SPEC_PREFIX int dnnLayoutCompare<float>(
          const dnnLayout_t l1, const dnnLayout_t l2)
          {return dnnLayoutCompare_F32(l1, l2);}
  SPEC_PREFIX int dnnLayoutCompare<double>(
          const dnnLayout_t l1, const dnnLayout_t l2)
          {return dnnLayoutCompare_F64(l1, l2);}


  TEMPLATE_PREFIX dnnError_t dnnAllocateBuffer(
          void **pPtr, dnnLayout_t layout);
  SPEC_PREFIX dnnError_t dnnAllocateBuffer<float>(
          void **pPtr, dnnLayout_t layout)
      {return dnnAllocateBuffer_F32(pPtr, layout);}
  SPEC_PREFIX dnnError_t dnnAllocateBuffer<double>(
          void **pPtr, dnnLayout_t layout)
      {return dnnAllocateBuffer_F64(pPtr, layout);}

  TEMPLATE_PREFIX dnnError_t dnnReleaseBuffer(
          void* ptr);
  SPEC_PREFIX dnnError_t dnnReleaseBuffer<float>(
          void* ptr) {
    dnnError_t status = E_SUCCESS;
    if( ptr != NULL) {
      status = dnnReleaseBuffer_F32(ptr);
    }
    return status; 
  }
  SPEC_PREFIX dnnError_t dnnReleaseBuffer<double>(
          void* ptr) {
    dnnError_t status = E_SUCCESS;
    if( ptr != NULL) {
      status = dnnReleaseBuffer_F64(ptr);
    }
    return status; 
  }

  TEMPLATE_PREFIX dnnError_t dnnLayoutDelete(
          dnnLayout_t& layout);
  SPEC_PREFIX dnnError_t dnnLayoutDelete<float>(
          dnnLayout_t& layout) {
    dnnError_t status = E_SUCCESS;
    if( layout != NULL) {
      status = dnnLayoutDelete_F32(layout);
      layout = NULL;
    }
    return status;
  }
  SPEC_PREFIX dnnError_t dnnLayoutDelete<double>(
          dnnLayout_t& layout) {
    dnnError_t status = E_SUCCESS;
    if( layout != NULL) {
      status = dnnLayoutDelete_F64(layout);
      layout = NULL;
    }
    return status;
  }

TEMPLATE_PREFIX dnnError_t dnnPrimitiveAttributesCreate(
        dnnPrimitiveAttributes_t *attributes);
SPEC_PREFIX dnnError_t dnnPrimitiveAttributesCreate<float>(
        dnnPrimitiveAttributes_t *attributes)
    {return dnnPrimitiveAttributesCreate_F32 (attributes);}
SPEC_PREFIX dnnError_t dnnPrimitiveAttributesCreate<double>(
        dnnPrimitiveAttributes_t *attributes)
    {return dnnPrimitiveAttributesCreate_F64 (attributes);}


TEMPLATE_PREFIX dnnError_t dnnPrimitiveAttributesDestroy(
        dnnPrimitiveAttributes_t attributes);
SPEC_PREFIX dnnError_t dnnPrimitiveAttributesDestroy<float>(
        dnnPrimitiveAttributes_t attributes)
        {return dnnPrimitiveAttributesDestroy_F32(attributes);}
SPEC_PREFIX dnnError_t dnnPrimitiveAttributesDestroy<double>(
        dnnPrimitiveAttributes_t attributes)
        {return dnnPrimitiveAttributesDestroy_F64(attributes);}

TEMPLATE_PREFIX dnnError_t dnnPrimitiveGetAttributes(
        dnnPrimitive_t primitive,
        dnnPrimitiveAttributes_t *attributes);
SPEC_PREFIX dnnError_t dnnPrimitiveGetAttributes<float>(
        dnnPrimitive_t primitive,
        dnnPrimitiveAttributes_t *attributes)
        {return dnnPrimitiveGetAttributes_F32(primitive, attributes);}
SPEC_PREFIX dnnError_t dnnPrimitiveGetAttributes<double>(
        dnnPrimitive_t primitive,
        dnnPrimitiveAttributes_t *attributes)
        {return dnnPrimitiveGetAttributes_F64(primitive, attributes);}

TEMPLATE_PREFIX dnnError_t dnnExecute(
        dnnPrimitive_t primitive, void *resources[]);
SPEC_PREFIX dnnError_t dnnExecute<float>(
        dnnPrimitive_t primitive, void *resources[])
        {return dnnExecute_F32(primitive, resources);}
SPEC_PREFIX dnnError_t dnnExecute<double>(
        dnnPrimitive_t primitive, void *resources[])
        {return dnnExecute_F64(primitive, resources);}

TEMPLATE_PREFIX dnnError_t dnnExecuteAsync(
        dnnPrimitive_t primitive, void *resources[]);
SPEC_PREFIX dnnError_t dnnExecuteAsync<float>(
        dnnPrimitive_t primitive, void *resources[])
        {return dnnExecuteAsync_F32(primitive, resources);}
SPEC_PREFIX dnnError_t dnnExecuteAsync<double>(
        dnnPrimitive_t primitive, void *resources[])
        {return dnnExecuteAsync_F64(primitive, resources);}

TEMPLATE_PREFIX dnnError_t dnnWaitFor(
        dnnPrimitive_t primitive);
SPEC_PREFIX dnnError_t dnnWaitFor<float>(
        dnnPrimitive_t primitive)
        {return dnnWaitFor_F32(primitive);}
SPEC_PREFIX dnnError_t dnnWaitFor<double>(
        dnnPrimitive_t primitive)
        {return dnnWaitFor_F64(primitive);}

TEMPLATE_PREFIX dnnError_t dnnDelete(
        dnnPrimitive_t& primitive);
SPEC_PREFIX dnnError_t dnnDelete<float>(
        dnnPrimitive_t& primitive) {
  dnnError_t status = E_SUCCESS;
  if (primitive != NULL) {
    status = dnnDelete_F32(primitive); 
    primitive = NULL;
  }
  return status;
}
SPEC_PREFIX dnnError_t dnnDelete<double>(
        dnnPrimitive_t& primitive) {
  dnnError_t status = E_SUCCESS;
  if (primitive != NULL) {
    status = dnnDelete_F64(primitive); 
    primitive = NULL;
  }
  return status;
}

TEMPLATE_PREFIX dnnError_t dnnConversionCreate(
        dnnPrimitive_t* pConversion, const dnnLayout_t from, const dnnLayout_t to);
SPEC_PREFIX dnnError_t dnnConversionCreate<float>(
        dnnPrimitive_t* pConversion, const dnnLayout_t from, const dnnLayout_t to)
        {return dnnConversionCreate_F32(pConversion, from, to);}
SPEC_PREFIX dnnError_t dnnConversionCreate<double>(
        dnnPrimitive_t* pConversion, const dnnLayout_t from, const dnnLayout_t to)
        {return dnnConversionCreate_F64(pConversion, from, to);}


TEMPLATE_PREFIX dnnError_t dnnConversionExecute(
        dnnPrimitive_t conversion, void *from, void *to);
SPEC_PREFIX dnnError_t dnnConversionExecute<float>(
        dnnPrimitive_t conversion, void *from, void *to)
        {return dnnConversionExecute_F32(conversion, from, to);}
SPEC_PREFIX dnnError_t dnnConversionExecute<double>(
        dnnPrimitive_t conversion, void *from, void *to)
        {return dnnConversionExecute_F64(conversion, from, to);}


TEMPLATE_PREFIX dnnError_t dnnConvolutionCreateForward(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t border_type);
SPEC_PREFIX dnnError_t dnnConvolutionCreateForward<float>(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t border_type)
        {return dnnConvolutionCreateForward_F32(
        pConvolution,
        attributes,
        algorithm,
        dimension, srcSize, dstSize, filterSize,
        convolutionStrides, inputOffset, border_type);}

SPEC_PREFIX dnnError_t dnnConvolutionCreateForward<double>(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t border_type)
        {return dnnConvolutionCreateForward_F64(
        pConvolution,
        attributes,
        algorithm,
        dimension, srcSize, dstSize, filterSize,
        convolutionStrides, inputOffset, border_type);}


TEMPLATE_PREFIX dnnError_t dnnConvolutionCreateForwardBias(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t border_type);
SPEC_PREFIX dnnError_t dnnConvolutionCreateForwardBias<float>(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t border_type)
        {return dnnConvolutionCreateForwardBias_F32(
        pConvolution,
        attributes,
        algorithm,
        dimension, srcSize, dstSize, filterSize,
        convolutionStrides, inputOffset, border_type);}
SPEC_PREFIX dnnError_t dnnConvolutionCreateForwardBias<double>(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t border_type)
        {return dnnConvolutionCreateForwardBias_F64(
        pConvolution,
        attributes,
        algorithm,
        dimension, srcSize, dstSize, filterSize,
        convolutionStrides, inputOffset, border_type);}


TEMPLATE_PREFIX dnnError_t dnnConvolutionCreateBackwardData(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t border_type);
SPEC_PREFIX dnnError_t dnnConvolutionCreateBackwardData<float>(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t border_type)
{return dnnConvolutionCreateBackwardData_F32(
        pConvolution,
        attributes,
        algorithm,
        dimension, srcSize, dstSize, filterSize,
        convolutionStrides, inputOffset, border_type);}
SPEC_PREFIX dnnError_t dnnConvolutionCreateBackwardData<double>(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t border_type)
{return dnnConvolutionCreateBackwardData_F64(
        pConvolution,
        attributes,
        algorithm,
        dimension, srcSize, dstSize, filterSize,
        convolutionStrides, inputOffset, border_type);}

TEMPLATE_PREFIX dnnError_t dnnConvolutionCreateBackwardFilter(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t border_type);
SPEC_PREFIX dnnError_t dnnConvolutionCreateBackwardFilter<float>(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t border_type)
{return dnnConvolutionCreateBackwardFilter_F32(
        pConvolution,
        attributes,
        algorithm,
        dimension, srcSize, dstSize, filterSize,
        convolutionStrides, inputOffset, border_type);}
SPEC_PREFIX dnnError_t dnnConvolutionCreateBackwardFilter<double>(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t border_type)
{return dnnConvolutionCreateBackwardFilter_F64(
        pConvolution,
        attributes,
        algorithm,
        dimension, srcSize, dstSize, filterSize,
        convolutionStrides, inputOffset, border_type);}

TEMPLATE_PREFIX dnnError_t dnnConvolutionCreateBackwardBias(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t dstSize[]);
SPEC_PREFIX dnnError_t dnnConvolutionCreateBackwardBias<float>(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t dstSize[])
{return dnnConvolutionCreateBackwardBias_F32(
        pConvolution,
        attributes,
        algorithm,
        dimension, dstSize);}
SPEC_PREFIX dnnError_t dnnConvolutionCreateBackwardBias<double>(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t dstSize[])
{return dnnConvolutionCreateBackwardBias_F64(
        pConvolution,
        attributes,
        algorithm,
        dimension, dstSize);}

TEMPLATE_PREFIX dnnError_t dnnGroupsConvolutionCreateForward(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t border_type);
SPEC_PREFIX dnnError_t dnnGroupsConvolutionCreateForward<float>(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t border_type)
{return dnnGroupsConvolutionCreateForward_F32(
        pConvolution,
        attributes,
        algorithm,
        groups, dimension, srcSize, dstSize, filterSize,
        convolutionStrides, inputOffset, border_type);}
SPEC_PREFIX dnnError_t dnnGroupsConvolutionCreateForward<double>(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t border_type)
{return dnnGroupsConvolutionCreateForward_F64(
        pConvolution,
        attributes,
        algorithm,
        groups, dimension, srcSize, dstSize, filterSize,
        convolutionStrides, inputOffset, border_type);}

TEMPLATE_PREFIX dnnError_t dnnGroupsConvolutionCreateForwardBias(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t border_type);
SPEC_PREFIX dnnError_t dnnGroupsConvolutionCreateForwardBias<float>(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t border_type)
{return dnnGroupsConvolutionCreateForwardBias_F32(
        pConvolution,
        attributes,
        algorithm,
        groups, dimension, srcSize, dstSize, filterSize,
        convolutionStrides, inputOffset, border_type);}
SPEC_PREFIX dnnError_t dnnGroupsConvolutionCreateForwardBias<double>(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t border_type)
{return dnnGroupsConvolutionCreateForwardBias_F64(
        pConvolution,
        attributes,
        algorithm,
        groups, dimension, srcSize, dstSize, filterSize,
        convolutionStrides, inputOffset, border_type);}

TEMPLATE_PREFIX dnnError_t dnnGroupsConvolutionCreateBackwardData(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t border_type);
SPEC_PREFIX dnnError_t dnnGroupsConvolutionCreateBackwardData<float>(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t border_type)
{return dnnGroupsConvolutionCreateBackwardData_F32(
        pConvolution,
        attributes,
        algorithm,
        groups, dimension, srcSize, dstSize, filterSize,
        convolutionStrides, inputOffset, border_type);}
SPEC_PREFIX dnnError_t dnnGroupsConvolutionCreateBackwardData<double>(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t border_type)
{return dnnGroupsConvolutionCreateBackwardData_F64(
        pConvolution,
        attributes,
        algorithm,
        groups, dimension, srcSize, dstSize, filterSize,
        convolutionStrides, inputOffset, border_type);}


TEMPLATE_PREFIX dnnError_t dnnGroupsConvolutionCreateBackwardFilter(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t border_type);
SPEC_PREFIX dnnError_t dnnGroupsConvolutionCreateBackwardFilter<float>(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t border_type)
{return dnnGroupsConvolutionCreateBackwardFilter_F32(
        pConvolution,
        attributes,
        algorithm,
        groups, dimension, srcSize, dstSize, filterSize,
        convolutionStrides, inputOffset, border_type);}
SPEC_PREFIX dnnError_t dnnGroupsConvolutionCreateBackwardFilter<double>(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t border_type)
{return dnnGroupsConvolutionCreateBackwardFilter_F64(
        pConvolution,
        attributes,
        algorithm,
        groups, dimension, srcSize, dstSize, filterSize,
        convolutionStrides, inputOffset, border_type);}

TEMPLATE_PREFIX dnnError_t dnnGroupsConvolutionCreateBackwardBias(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t dstSize[]);
SPEC_PREFIX dnnError_t dnnGroupsConvolutionCreateBackwardBias<float>(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t dstSize[])
{return dnnGroupsConvolutionCreateBackwardBias_F32(
        pConvolution,
        attributes,
        algorithm,
        groups, dimension, dstSize);}
SPEC_PREFIX dnnError_t dnnGroupsConvolutionCreateBackwardBias<double>(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t dstSize[])
{return dnnGroupsConvolutionCreateBackwardBias_F64(
        pConvolution,
        attributes,
        algorithm,
        groups, dimension, dstSize);}

TEMPLATE_PREFIX dnnError_t dnnReLUCreateForward(
        dnnPrimitive_t* pRelu,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float negativeSlope);
SPEC_PREFIX dnnError_t dnnReLUCreateForward<float>(
        dnnPrimitive_t* pRelu,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float negativeSlope)
{return dnnReLUCreateForward_F32(
        pRelu,
        attributes,
        dataLayout, negativeSlope);}
SPEC_PREFIX dnnError_t dnnReLUCreateForward<double>(
        dnnPrimitive_t* pRelu,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float negativeSlope)
{return dnnReLUCreateForward_F64(
        pRelu,
        attributes,
        dataLayout, negativeSlope);}

TEMPLATE_PREFIX dnnError_t dnnReLUCreateBackward(
        dnnPrimitive_t* pRelu,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, float negativeSlope);
SPEC_PREFIX dnnError_t dnnReLUCreateBackward<float>(
        dnnPrimitive_t* pRelu,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, float negativeSlope)
{return dnnReLUCreateBackward_F32(
        pRelu,
        attributes,
        diffLayout, dataLayout, negativeSlope);}
SPEC_PREFIX dnnError_t dnnReLUCreateBackward<double>(
        dnnPrimitive_t* pRelu,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, float negativeSlope)
{return dnnReLUCreateBackward_F64(
        pRelu,
        attributes,
        diffLayout, dataLayout, negativeSlope);}

TEMPLATE_PREFIX dnnError_t dnnLRNCreateForward(
        dnnPrimitive_t* pLrn,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, size_t kernel_size, float alpha, float beta, float k);
SPEC_PREFIX dnnError_t dnnLRNCreateForward<float>(
        dnnPrimitive_t* pLrn,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, size_t kernel_size, float alpha, float beta, float k)
{return dnnLRNCreateForward_F32(
        pLrn,
        attributes,
        dataLayout, kernel_size, alpha, beta, k);}
SPEC_PREFIX dnnError_t dnnLRNCreateForward<double>(
        dnnPrimitive_t* pLrn,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, size_t kernel_size, float alpha, float beta, float k)
{return dnnLRNCreateForward_F64(
        pLrn,
        attributes,
        dataLayout, kernel_size, alpha, beta, k);}


TEMPLATE_PREFIX dnnError_t dnnLRNCreateBackward(
        dnnPrimitive_t* pLrn,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, size_t kernel_size, float alpha, float beta, float k);
SPEC_PREFIX dnnError_t dnnLRNCreateBackward<float>(
        dnnPrimitive_t* pLrn,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, size_t kernel_size, float alpha, float beta, float k)
{return dnnLRNCreateBackward_F32(
        pLrn,
        attributes,
        diffLayout, dataLayout, kernel_size, alpha, beta, k);}
SPEC_PREFIX dnnError_t dnnLRNCreateBackward<double>(
        dnnPrimitive_t* pLrn,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, size_t kernel_size, float alpha, float beta, float k)
{return dnnLRNCreateBackward_F64(
        pLrn,
        attributes,
        diffLayout, dataLayout, kernel_size, alpha, beta, k);}


TEMPLATE_PREFIX dnnError_t dnnPoolingCreateForward(
        dnnPrimitive_t* pPooling,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t op,
        const dnnLayout_t srcLayout,
        const size_t kernelSize[], const size_t kernelStride[],
        const int inputOffset[], const dnnBorder_t border_type);
SPEC_PREFIX dnnError_t dnnPoolingCreateForward<float>(
        dnnPrimitive_t* pPooling,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t op,
        const dnnLayout_t srcLayout,
        const size_t kernelSize[], const size_t kernelStride[],
        const int inputOffset[], const dnnBorder_t border_type)
{return dnnPoolingCreateForward_F32(
        pPooling,
        attributes,
        op,
        srcLayout,
        kernelSize, kernelStride,
        inputOffset, border_type);}
SPEC_PREFIX dnnError_t dnnPoolingCreateForward<double>(
        dnnPrimitive_t* pPooling,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t op,
        const dnnLayout_t srcLayout,
        const size_t kernelSize[], const size_t kernelStride[],
        const int inputOffset[], const dnnBorder_t border_type)
{return dnnPoolingCreateForward_F64(
        pPooling,
        attributes,
        op,
        srcLayout,
        kernelSize, kernelStride,
        inputOffset, border_type);}


TEMPLATE_PREFIX dnnError_t dnnPoolingCreateBackward(
        dnnPrimitive_t* pPooling,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t op,
        const dnnLayout_t srcLayout,
        const size_t kernelSize[], const size_t kernelStride[],
        const int inputOffset[], const dnnBorder_t border_type);
SPEC_PREFIX dnnError_t dnnPoolingCreateBackward<float>(
        dnnPrimitive_t* pPooling,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t op,
        const dnnLayout_t srcLayout,
        const size_t kernelSize[], const size_t kernelStride[],
        const int inputOffset[], const dnnBorder_t border_type)
{return dnnPoolingCreateBackward_F32(
        pPooling,
        attributes,
        op,
        srcLayout,
        kernelSize, kernelStride,
        inputOffset,border_type);}
SPEC_PREFIX dnnError_t dnnPoolingCreateBackward<double>(
        dnnPrimitive_t* pPooling,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t op,
        const dnnLayout_t srcLayout,
        const size_t kernelSize[], const size_t kernelStride[],
        const int inputOffset[], const dnnBorder_t border_type)
{return dnnPoolingCreateBackward_F64(
        pPooling,
        attributes,
        op,
        srcLayout,
        kernelSize, kernelStride,
        inputOffset,border_type);}

TEMPLATE_PREFIX dnnError_t dnnConcatCreate(
        dnnPrimitive_t *pConcat,
        dnnPrimitiveAttributes_t attributes,
        const size_t N,
        dnnLayout_t src[]);
SPEC_PREFIX dnnError_t dnnConcatCreate<float>(
        dnnPrimitive_t *pConcat,
        dnnPrimitiveAttributes_t attributes,
        const size_t N,
        dnnLayout_t src[])
{return dnnConcatCreate_F32(
        pConcat,
        attributes,
        N,
        src);}
SPEC_PREFIX dnnError_t dnnConcatCreate<double>(
        dnnPrimitive_t *pConcat,
        dnnPrimitiveAttributes_t attributes,
        const size_t N,
        dnnLayout_t src[])
{return dnnConcatCreate_F64(
        pConcat,
        attributes,
        N,
        src);}


TEMPLATE_PREFIX dnnError_t dnnSplitCreate(
        dnnPrimitive_t *pSplit,
        dnnPrimitiveAttributes_t attributes,
        const size_t N,
        dnnLayout_t src,
        size_t dst[]);
SPEC_PREFIX dnnError_t dnnSplitCreate<float>(
        dnnPrimitive_t *pSplit,
        dnnPrimitiveAttributes_t attributes,
        const size_t N,
        dnnLayout_t src,
        size_t dst[])
{return dnnSplitCreate_F32(
        pSplit,
        attributes,
        N,
        src,
        dst);}
SPEC_PREFIX dnnError_t dnnSplitCreate<double>(
        dnnPrimitive_t *pSplit,
        dnnPrimitiveAttributes_t attributes,
        const size_t N,
        dnnLayout_t src,
        size_t dst[])
{return dnnSplitCreate_F64(
        pSplit,
        attributes,
        N,
        src,
        dst);}

TEMPLATE_PREFIX dnnError_t dnnSumCreate(
        dnnPrimitive_t *pSum,
        dnnPrimitiveAttributes_t attributes,
        const size_t nSummands, dnnLayout_t layout, Dtype *coefficients);
SPEC_PREFIX dnnError_t dnnSumCreate<float>(
        dnnPrimitive_t *pSum,
        dnnPrimitiveAttributes_t attributes,
        const size_t nSummands, dnnLayout_t layout, float *coefficients)
{return dnnSumCreate_F32(
        pSum,
        attributes,
        nSummands,
        layout, coefficients);}
SPEC_PREFIX dnnError_t dnnSumCreate<double>(
        dnnPrimitive_t *pSum,
        dnnPrimitiveAttributes_t attributes,
        const size_t nSummands, dnnLayout_t layout, double *coefficients)
{return dnnSumCreate_F64(
        pSum,
        attributes,
        nSummands,
        layout, coefficients);}

TEMPLATE_PREFIX dnnError_t dnnBatchNormalizationCreateForward(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float eps, unsigned int flags);
SPEC_PREFIX dnnError_t dnnBatchNormalizationCreateForward<float>(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float eps, unsigned int flags)
{return dnnBatchNormalizationCreateForward_v2_F32(
        pBatchNormalization,
        attributes,
        dataLayout, eps, flags); }
SPEC_PREFIX dnnError_t dnnBatchNormalizationCreateForward<double>(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float eps, unsigned int flags)
{return dnnBatchNormalizationCreateForward_v2_F64(
        pBatchNormalization,
        attributes,
        dataLayout, eps, flags); }

TEMPLATE_PREFIX dnnError_t dnnBatchNormalizationCreateBackward(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float eps, unsigned int flags);
SPEC_PREFIX  dnnError_t dnnBatchNormalizationCreateBackward<float>(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float eps, unsigned int flags)
{return dnnBatchNormalizationCreateBackward_v2_F32(
        pBatchNormalization,
        attributes,
        dataLayout, eps, flags); }
SPEC_PREFIX dnnError_t dnnBatchNormalizationCreateBackward<double>(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float eps, unsigned int flags)
{return dnnBatchNormalizationCreateBackward_v2_F64(
        pBatchNormalization,
        attributes,
        dataLayout, eps, flags); }
#endif
