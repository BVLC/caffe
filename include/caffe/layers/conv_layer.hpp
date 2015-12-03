#ifndef CAFFE_CONV_LAYER_HPP_
#define CAFFE_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {

/**
 * @brief Convolves the input image with a bank of learned filters,
 *        and (optionally) adds biases.
 *
 *   Caffe convolves by reduction to matrix multiplication. This achieves
 *   high-throughput and generality of input and filter dimensions but comes at
 *   the cost of memory for matrices. This makes use of efficiency in BLAS.
 *
 *   The input is "im2col" transformed to a channel K' x H x W data matrix
 *   for multiplication with the N x K' x H x W filter matrix to yield a
 *   N' x H x W output matrix that is then "col2im" restored. K' is the
 *   input channel * kernel height * kernel width dimension of the unrolled
 *   inputs so that the im2col matrix has a column for each input region to
 *   be filtered. col2im restores the output spatial structure by rolling up
 *   the output channel N' columns of the output matrix.
 */
template<typename Dtype>
class ConvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:
  explicit ConvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {
  }

  virtual inline const char* type() const {
    return "Convolution";
  }

  virtual uint_tp ForwardFlops() {
    uint_tp group = this->group_;
    uint_tp N = 1;
    uint_tp M = this->num_output_ / group;
    uint_tp K = this->channels_;
    const int_tp* kshape = this->kernel_shape_.cpu_data();
    for (int_tp i = 0; i < this->output_shape_.size(); ++i) {
      N *= this->output_shape_[i];
      K *= kshape[i];
    }
    K /= group;
    return group* (M * N * (2 * K - 1));
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() {
    return false;
  }
  virtual void compute_output_shape();
};

}  // namespace caffe

#endif  // CAFFE_CONV_LAYER_HPP_
