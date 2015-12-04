#ifndef CAFFE_CODE_GENERATORS_RELU_H_
#define CAFFE_CODE_GENERATORS_RELU_H_

#include "caffe/proto/caffe.pb.h"
#include <vector>

#if defined __x86_64__ || defined _M_X64
# define XBYAK_NO_OP_NAMES
# define XBYAK_USE_MMAP_ALLOCATOR
# include "../xbyak/xbyak_util.h"
#endif

namespace caffe
{
// Declarations of CodeGenerator classes.

template <typename Dtype>
class ReLULayer;

template <typename Dtype>
class Blob;

template <typename Dtype>
class ReLUCodeGeneratorForward 
#if defined __x86_64__ || defined _M_X64
  : public ::Xbyak::CodeGenerator
#endif
{
public:
  ReLUCodeGeneratorForward();
  ~ReLUCodeGeneratorForward();

  typedef void (Callback_t)(
    Dtype* top_data, 
    const Dtype* bottom_data, 
    int count,
    Dtype negative_slope);

  Callback_t* Get_callback(ReLULayer<Dtype>* layer, Blob<Dtype>* top);

private:
  void Create_callback(ReLULayer<Dtype>* layer);

  static Callback_t Naive;
  Callback_t* Callback;
  std::vector<int> layer_output_shape_signature;
};

template <typename Dtype>
class ReLUCodeGeneratorBackward
#if defined __x86_64__ || defined _M_X64
  : public ::Xbyak::CodeGenerator
#endif
{
public:
  ReLUCodeGeneratorBackward();
  ~ReLUCodeGeneratorBackward();

  typedef void (Callback_t)(
    const Dtype* top_diff, 
    Dtype* bottom_diff,
    const Dtype* bottom_data, 
    int count,
    Dtype negative_slope);

  Callback_t* Get_callback(ReLULayer<Dtype>* layer, Blob<Dtype>* top);

private:
  void Create_callback(ReLULayer<Dtype>* layer);

  static Callback_t Naive;
  Callback_t* Callback;
  std::vector<int> layer_output_shape_signature;
};
}

#endif // CAFFE_CODE_GENERATORS_RELU_H_