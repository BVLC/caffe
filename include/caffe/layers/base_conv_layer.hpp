#ifndef CAFFE_BASE_CONVOLUTION_LAYER_HPP_
#define CAFFE_BASE_CONVOLUTION_LAYER_HPP_

#include <boost/thread/tss.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {

/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */
template <typename Dtype> class BaseConvolutionLayer : public Layer<Dtype> {
public:
  explicit BaseConvolutionLayer(const LayerParameter &param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top) override;

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

protected:
  void Reshape_const(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top) const override;

  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.
  void forward_cpu_gemm(const Dtype *input, const Dtype *weights, Dtype *output,
                        bool skip_im2col = false) const;
  void forward_cpu_bias(Dtype *output, const Dtype *bias) const;

#ifndef CPU_ONLY
  void forward_gpu_gemm(const Dtype *col_input, const Dtype *weights,
                        Dtype *output, bool skip_im2col = false) const;
  void forward_gpu_bias(Dtype *output, const Dtype *bias) const;
  void backward_gpu_gemm(const Dtype *input, const Dtype *weights,
                         Dtype *col_output);
  void weight_gpu_gemm(const Dtype *col_input, const Dtype *output,
                       Dtype *weights);
  void backward_gpu_bias(Dtype *bias, const Dtype *input);
#endif

  /// @brief The spatial dimensions of the input.
  inline int input_shape(int channel_axis, int i) const {
    return (*bottom_shape_)[channel_axis + i];
  }
  // Compute height_out_ and width_out_ from other parameters.
  virtual vector<int> compute_output_shape() const = 0;

  /// @brief The spatial dimensions of a filter kernel.
  Blob<int> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  Blob<int> stride_;
  /// @brief The spatial dimensions of the padding.
  Blob<int> pad_;
  /// @brief The spatial dimensions of the dilation.
  Blob<int> dilation_;
  /// @brief The spatial dimensions of the convolution input.
  mutable ::boost::thread_specific_ptr<Blob<int>> conv_input_shape_ptr_;
  /// @brief The spatial dimensions of the output.
  mutable ::boost::thread_specific_ptr<vector<int>> bottom_shape_{
      [](vector<int> *p) {}};

  int num_spatial_axes_;
  int channel_axis_;
  int channels_;
  int group_;
  int num_output_;
  bool bias_term_;
  bool is_1x1_;
  bool force_nd_im2col_;

private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  inline void conv_im2col_cpu(const Dtype *data, Dtype *col_buff) const {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_cpu(data, channels_, conv_input_shape_ptr_->cpu_data()[1],
                 conv_input_shape_ptr_->cpu_data()[2],
                 kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                 pad_.cpu_data()[0], pad_.cpu_data()[1], stride_.cpu_data()[0],
                 stride_.cpu_data()[1], dilation_.cpu_data()[0],
                 dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_ptr_->cpu_data(),
                    col_buffer_ptr_->shape().data(), kernel_shape_.cpu_data(),
                    pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(),
                    col_buff);
    }
  }

#ifndef CPU_ONLY
  inline void conv_im2col_gpu(const Dtype *data, Dtype *col_buff) const {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_gpu(data, channels_, conv_input_shape_ptr_->cpu_data()[1],
                 conv_input_shape_ptr_->cpu_data()[2],
                 kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                 pad_.cpu_data()[0], pad_.cpu_data()[1], stride_.cpu_data()[0],
                 stride_.cpu_data()[1], dilation_.cpu_data()[0],
                 dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_gpu(
          data, num_spatial_axes_, channels_ * (*conv_out_spatial_dim_ptr_),
          conv_input_shape_ptr_->gpu_data(), col_buffer_ptr_->gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
          dilation_.gpu_data(), col_buff);
    }
  }

#endif

protected:
  // int conv_out_channels_;
  mutable ::boost::thread_specific_ptr<int> conv_out_spatial_dim_ptr_;
  int kernel_dim_;

  mutable ::boost::thread_specific_ptr<Blob<Dtype>> col_buffer_ptr_;
  mutable ::boost::thread_specific_ptr<Blob<Dtype>> bias_multiplier_ptr_;
};

} // namespace caffe

#endif // CAFFE_BASE_CONVOLUTION_LAYER_HPP_
