#ifndef CAFFE_BASE_CONVOLUTION_LAYER_HPP_
#define CAFFE_BASE_CONVOLUTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"

namespace caffe {

/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */
template<typename Dtype, typename MItype, typename MOtype>
class BaseConvolutionLayer : public Layer<Dtype, MItype, MOtype> {
 public:
  explicit BaseConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype, MItype, MOtype>(param), col_buffer_lock_id_(-1) {
  }
  virtual void LayerSetUp(const vector<Blob<MItype>*>& bottom,
                          const vector<Blob<MOtype>*>& top);
  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
                       const vector<Blob<MOtype>*>& top);

  virtual inline int_tp MinBottomBlobs() const {
    return 1;
  }
  virtual inline int_tp MinTopBlobs() const {
    return 1;
  }
  virtual inline bool EqualNumBottomTopBlobs() const {
    return true;
  }

 protected:
  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.
  void forward_cpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output,
                        bool skip_im2col = false);
  void forward_cpu_bias(Dtype* output, const Dtype* bias);
  void backward_cpu_gemm(const Dtype* input, const Dtype* weights,
                         Dtype* output);
  void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights);
  void backward_cpu_bias(Dtype* bias, const Dtype* input);

#ifndef CPU_ONLY
  void forward_gpu_gemm(vptr<const Dtype> col_input, const uint_tp col_input_off,
                        vptr<const Dtype> weights, vptr<Dtype> output,
                        const uint_tp output_off, bool skip_im2col = false);
  void forward_gpu_bias(vptr<Dtype> output, const uint_tp output_off,
                        vptr<const Dtype> bias);
  void backward_gpu_gemm(vptr<const Dtype> input, const uint_tp input_off,
                         vptr<const Dtype> weights, vptr<Dtype> col_output,
                         const uint_tp col_output_off);
  void weight_gpu_gemm(vptr<const Dtype> col_input, const uint_tp col_input_off,
                       vptr<const Dtype> output, const uint_tp output_off,
                       vptr<Dtype> weights);
  void backward_gpu_bias(vptr<Dtype> bias, vptr<const Dtype> input,
                         const uint_tp input_off);

  shared_ptr<Blob<Dtype> > col_buffer();
  void unlock_col_buffer();
#endif

  /// @brief The spatial dimensions of the input.
  inline int_tp input_shape(int_tp i) {
    return (*bottom_shape_)[channel_axis_ + i];
  }
  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  virtual bool reverse_dimensions() = 0;
  // Compute height_out_ and width_out_ from other parameters.
  virtual void compute_output_shape() = 0;

  /// @brief The spatial dimensions of a filter kernel.
  Blob<int_tp> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  Blob<int_tp> stride_;
  /// @brief The spatial dimensions of the padding.
  Blob<int_tp> pad_;
  /// @brief The spatial dimensions of the dilation.
  Blob<int_tp> dilation_;
  /// @brief The spatial dimensions of the convolution input.
  Blob<int_tp> conv_input_shape_;
  /// @brief The spatial dimensions of the col_buffer.
  vector<int_tp> col_buffer_shape_;
  /// @brief The spatial dimensions of the output.
  vector<int_tp> output_shape_;
  const vector<int_tp>* bottom_shape_;

  int_tp num_spatial_axes_;
  int_tp bottom_dim_;
  int_tp top_dim_;

  int_tp channel_axis_;
  int_tp num_;
  int_tp channels_;
  int_tp group_;
  int_tp out_spatial_dim_;
  int_tp weight_offset_;
  int_tp num_output_;
  bool bias_term_;
  bool is_1x1_;
  bool force_nd_im2col_;
  bool use_colbuffer_;

 private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_cpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), col_buff);
    }
  }
  inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_cpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_cpu(col_buff, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), data);
    }
  }

#ifndef CPU_ONLY
  inline void conv_im2col_gpu(vptr<const Dtype> data, vptr<Dtype> col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      this->device_->im2col(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    } else {
      this->device_->im2col_nd(data, num_spatial_axes_, num_kernels_im2col_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(),
          stride_.gpu_data(), dilation_.gpu_data(), col_buff);
    }
  }

  inline void conv_col2im_gpu(vptr<const Dtype> col_buff, vptr<Dtype> data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      this->device_->col2im(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      this->device_->col2im_nd(col_buff, num_spatial_axes_,
          num_kernels_col2im_, conv_input_shape_.gpu_data(),
          col_buffer_.gpu_shape(), kernel_shape_.gpu_data(), pad_.gpu_data(),
          stride_.gpu_data(), dilation_.gpu_data(), data);
    }
  }
#endif  // !CPU_ONLY

  int_tp num_kernels_im2col_;
  int_tp num_kernels_col2im_;
  int_tp conv_out_channels_;
  int_tp conv_in_channels_;
  int_tp conv_out_spatial_dim_;
  int_tp kernel_dim_;
  int_tp col_offset_;
  int_tp output_offset_;

  bool use_skernel_;

  Blob<Dtype> col_buffer_;
  shared_ptr<Blob<Dtype> > shared_col_buffer_;
  int_tp col_buffer_lock_id_;
  Blob<Dtype> bias_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_BASE_CONVOLUTION_LAYER_HPP_
