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

#ifndef CAFFE_CODE_GENERATORS_POOLING_H_
#define CAFFE_CODE_GENERATORS_POOLING_H_

#include <stdint.h>
#include <vector>

#if defined __x86_64__ || defined _M_X64
# define XBYAK_NO_OP_NAMES
# define XBYAK_USE_MMAP_ALLOCATOR
# include "../xbyak/xbyak_util.h"
#endif

#include "caffe/proto/caffe.pb.h"

namespace caffe {
// Declarations of CodeGenerator classes.

template <typename Dtype>
class PoolingLayer;

template <typename Dtype>
class Blob;

template <typename Dtype>
class PoolingCodeGeneratorForward
#if defined __x86_64__ || defined _M_X64
  : public ::Xbyak::CodeGenerator
#endif
{
 public:
  PoolingCodeGeneratorForward();
  ~PoolingCodeGeneratorForward();

  typedef void (Callback_t)(
    const Dtype* bottom_data,
    Dtype* top_data,
    int top_count,
    int batch_start,
    int batch_end,
    void* mask,
    int64_t channel_start,
    int64_t channel_end,
    PoolingLayer<Dtype>* layer,
    bool use_top_mask);

  Callback_t* Get_callback(
    PoolingLayer<Dtype>* layer,
    Blob<Dtype>* top,
    bool use_top_mask);

 private:
  void Create_callback(PoolingLayer<Dtype>* layer);

  static void Naive(
    const Dtype* bottom_data,
    Dtype* top_data,
    int top_count,
    int batch_start,
    int batch_end,
    void* mask,
    int64_t channel_start,
    int64_t channel_end,
    PoolingLayer<Dtype>* layer,
    bool use_top_mask);
  Callback_t* Callback;
  std::vector<int> Layer_output_shape_signature;
  bool Use_top_mask;
  PoolingParameter_PoolMethod Method;
};

template <typename Dtype>
class PoolingCodeGeneratorBackward
#if defined __x86_64__ || defined _M_X64
  : public ::Xbyak::CodeGenerator
#endif
{
 public:
  PoolingCodeGeneratorBackward();
  ~PoolingCodeGeneratorBackward();

  typedef void (Callback_t)(
    const Dtype* top_diff,
    Dtype* bottom_diff,
    int batch_start,
    int batch_end,
    int64_t channel_start,
    int64_t channel_end,
    bool use_top_mask,
    const void* mask,
    PoolingLayer<Dtype>* layer);

  Callback_t* Get_callback(PoolingLayer<Dtype>* layer, Blob<Dtype>* top);

 private:
  void Create_callback(PoolingLayer<Dtype>* layer);

  static void Naive(
    const Dtype* top_diff,
    Dtype* bottom_diff,
    int batch_start,
    int batch_end,
    int64_t channel_start,
    int64_t channel_end,
    bool use_top_mask,
    const void* mask,
    PoolingLayer<Dtype>* layer);
  Callback_t* Callback;
  std::vector<int> layer_output_shape_signature;
};
}  // namespace caffe

#endif  // CAFFE_CODE_GENERATORS_POOLING_H_
