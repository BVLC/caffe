#ifndef CAFFE_CODE_GENERATORS_POOLING_H_
#define CAFFE_CODE_GENERATORS_POOLING_H_

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

  typedef void (*Callback_t)(
    const Dtype* bottom_data,
    Dtype* top_data,
    int top_count,
    int batch_start,
    int batch_end,
    void* mask,
    PoolingLayer<Dtype>* layer,
    bool use_top_mask);

  Callback_t Get_callback(
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
    PoolingLayer<Dtype>* layer,
    bool use_top_mask);
  Callback_t Callback;
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

  typedef void (*Callback_t)(
    const Dtype* top_diff,
    Dtype* bottom_diff,
    int batch_start,
    int batch_end,
    bool use_top_mask,
    const void* mask,
    PoolingLayer<Dtype>* layer);

  Callback_t Get_callback(PoolingLayer<Dtype>* layer, Blob<Dtype>* top);

 private:
  void Create_callback(PoolingLayer<Dtype>* layer);

  static void Naive(
    const Dtype* top_diff,
    Dtype* bottom_diff,
    int batch_start,
    int batch_end,
    bool use_top_mask,
    const void* mask,
    PoolingLayer<Dtype>* layer);
  Callback_t Callback;
  std::vector<int> layer_output_shape_signature;
};
}  // namespace caffe

#endif  // CAFFE_CODE_GENERATORS_POOLING_H_
