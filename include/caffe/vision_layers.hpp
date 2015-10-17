#ifndef CAFFE_VISION_LAYERS_HPP_
#define CAFFE_VISION_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "device.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea_im2col.hpp"
#endif

namespace caffe {

/**
 * @brief Computes a one edge per dimension 2D affinity graph
 * for a given segmentation/label map
 */
template<typename Dtype>
class AffinityLayer : public Layer<Dtype> {
 public:
  explicit AffinityLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {
  }

  virtual inline const char* type() const {
    return "Affinity";
  }

 protected:
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

 private:
  std::vector< shared_ptr< Blob<Dtype> > > min_index_;
  std::vector<int> offsets_;
};

/**
 * @brief Computes a connected components map from a segmentation map.
 */
template<typename Dtype>
class ConnectedComponentLayer : public Layer<Dtype> {
 public:
  explicit ConnectedComponentLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {
  }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const {
    return 1;
  }

  virtual inline int ExactNumTopBlobs() const {
    return 1;
  }

  virtual inline const char* type() const {
    return "ConnectedComponent";
  }

 protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom);

 private:
     cv::Mat FindBlobs(const int maxlabel, const cv::Mat &input);
};

/**
 * @brief Merges and crops feature maps for U-Net architectures.
 */
template<typename Dtype>
class MergeCropLayer : public Layer<Dtype> {
 public:
  explicit MergeCropLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
  }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const {
    return 2;
  }

  virtual inline const char* type() const {
    return "MergeCrop";
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

 private:
  vector<int> forward_;
  vector<int> backward_;
  Blob<int> shape_a_;
  Blob<int> shape_b_;
};

/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */
template<typename Dtype>
class BaseConvolutionLayer : public Layer<Dtype> {
 public:
  explicit BaseConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const {
    return 1;
  }
  virtual inline int MinTopBlobs() const {
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
  void forward_gpu_gemm(const Dtype* col_input, const int col_input_off,
                        const Dtype* weights, Dtype* output,
                        const int output_off, bool skip_im2col = false);
  void forward_gpu_bias(Dtype* output, const int output_off, const Dtype* bias);
  void backward_gpu_gemm(const Dtype* input, const int input_off,
                         const Dtype* weights, Dtype* col_output,
                         const int col_output_off);
  void weight_gpu_gemm(const Dtype* col_input, const int col_input_off,
                       const Dtype* output, const int output_off,
                       Dtype* weights);
  void backward_gpu_bias(Dtype* bias, const Dtype* input, const int input_off);

  shared_ptr< Blob<Dtype> > col_buffer();
#endif

  /// @brief The spatial dimensions of the input.
  inline int input_shape(int i) {
    return (*bottom_shape_)[channel_axis_ + i];
  }
  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  virtual bool reverse_dimensions() = 0;
  // Compute height_out_ and width_out_ from other parameters.
  virtual void compute_output_shape() = 0;

  /// @brief The spatial dimensions of a filter kernel.
  Blob<int> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  Blob<int> stride_;
  /// @brief The spatial dimensions of the padding.
  Blob<int> pad_;
  /// @brief The spatial dimension of the kernel stride.
  Blob<int> kstride_;
  /// @brief The spatial dimensions of the convolution input.
  Blob<int> conv_input_shape_;
  /// @brief The spatial dimensions of the col_buffer.
  vector<int> col_buffer_shape_;
  /// @brief The spatial dimensions of the output.
  vector<int> output_shape_;
  const vector<int>* bottom_shape_;

  int num_spatial_axes_;
  int bottom_dim_;
  int top_dim_;

  int channel_axis_;
  int num_;
  int channels_;
  int group_;
  int out_spatial_dim_;
  int weight_offset_;
  int num_output_;
  bool bias_term_;
  bool is_1x1_;
  bool force_nd_im2col_;

 private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_cpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), col_buff);
    }
  }
  inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_cpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1], data);
    } else {
      col2im_nd_cpu(col_buff, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), data);
    }
  }

#ifndef CPU_ONLY
#ifdef USE_CUDA
  inline void conv_im2col_gpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      if (this->use_skernel_) {
        im2col_sk_gpu(data, conv_in_channels_, conv_input_shape_.cpu_data()[1],
                      conv_input_shape_.cpu_data()[2],
                      kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                      pad_.cpu_data()[0], pad_.cpu_data()[1],
                      stride_.cpu_data()[0], stride_.cpu_data()[1],
                      kstride_.cpu_data()[0], kstride_.cpu_data()[1], col_buff);
      } else {
        im2col_gpu(data, conv_in_channels_, conv_input_shape_.cpu_data()[1],
                   conv_input_shape_.cpu_data()[2], kernel_shape_.cpu_data()[0],
                   kernel_shape_.cpu_data()[1], pad_.cpu_data()[0],
                   pad_.cpu_data()[1], stride_.cpu_data()[0],
                   stride_.cpu_data()[1], col_buff);
      }
    } else {
      if (this->use_skernel_) {
        im2col_ndsk_gpu(data, num_spatial_axes_, num_kernels_im2col_,
                        conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
                        kernel_shape_.gpu_data(), pad_.gpu_data(),
                        stride_.gpu_data(), kstride_.gpu_data(), col_buff);
      } else {
        im2col_nd_gpu(data, num_spatial_axes_, num_kernels_im2col_,
                      conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
                      kernel_shape_.gpu_data(), pad_.gpu_data(),
                      stride_.gpu_data(), col_buff);
      }
    }
  }

  inline void conv_col2im_gpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      if (this->use_skernel_) {
        col2im_sk_gpu(col_buff, conv_in_channels_,
            conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
            kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
            pad_.cpu_data()[0], pad_.cpu_data()[1],
            stride_.cpu_data()[0], stride_.cpu_data()[1],
            kstride_.cpu_data()[0], kstride_.cpu_data()[1], data);
      } else {
        col2im_gpu(col_buff, conv_in_channels_,
            conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
            kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
            pad_.cpu_data()[0], pad_.cpu_data()[1],
            stride_.cpu_data()[0], stride_.cpu_data()[1], data);
      }
    } else {
      if (this->use_skernel_) {
        col2im_ndsk_gpu(col_buff, num_spatial_axes_, num_kernels_col2im_,
                      conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
                      kernel_shape_.gpu_data(), pad_.gpu_data(),
                      stride_.gpu_data(), kstride_.gpu_data(), data);
      } else {
        col2im_nd_gpu(col_buff, num_spatial_axes_, num_kernels_col2im_,
                      conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
                      kernel_shape_.gpu_data(), pad_.gpu_data(),
                      stride_.gpu_data(), data);
      }
    }
  }
#endif  // USE_CUDA
#ifdef USE_GREENTEA
  inline void greentea_conv_im2col_gpu(const Dtype* data, const int data_off,
                                       Dtype* col_buff,
                                       const int col_buff_off) {
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(
        this->device_->id());

    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      if (this->use_skernel_) {
        greentea_im2col_sk_gpu<Dtype>(&program, &ctx, (cl_mem) data, data_off,
                                      conv_in_channels_,
                                      conv_input_shape_.cpu_data()[1],
                                      conv_input_shape_.cpu_data()[2],
                                      kernel_shape_.cpu_data()[0],
                                      kernel_shape_.cpu_data()[1],
                                      pad_.cpu_data()[0], pad_.cpu_data()[1],
                                      stride_.cpu_data()[0],
                                      stride_.cpu_data()[1],
                                      kstride_.cpu_data()[0],
                                      kstride_.cpu_data()[1],
                                      (cl_mem) col_buff);
      } else {
        greentea_im2col_gpu<Dtype>(&program, &ctx, (cl_mem) data, data_off,
                                   conv_in_channels_,
                                   conv_input_shape_.cpu_data()[1],
                                   conv_input_shape_.cpu_data()[2],
                                   kernel_shape_.cpu_data()[0],
                                   kernel_shape_.cpu_data()[1],
                                   pad_.cpu_data()[0], pad_.cpu_data()[1],
                                   stride_.cpu_data()[0], stride_.cpu_data()[1],
                                   (cl_mem) col_buff, col_buff_off);
      }
    } else {
      if (this->use_skernel_) {
        greentea_im2col_ndsk_gpu<Dtype>(&program, &ctx, (cl_mem) data, data_off,
                                        num_spatial_axes_, num_kernels_im2col_,
                                        (cl_mem) (conv_input_shape_.gpu_data()),
                                        (cl_mem) (col_buffer_.gpu_shape()),
                                        (cl_mem) (kernel_shape_.gpu_data()),
                                        (cl_mem) (pad_.gpu_data()),
                                        (cl_mem) (stride_.gpu_data()),
                                        (cl_mem) (kstride_.gpu_data()),
                                        (cl_mem) col_buff, col_buff_off);
      } else {
        greentea_im2col_nd_gpu<Dtype>(&program, &ctx, (cl_mem) data, data_off,
                                      num_spatial_axes_,
                                      0,
                                      num_kernels_im2col_,
                                      (cl_mem) (conv_input_shape_.gpu_data()),
                                      (cl_mem) (col_buffer_.gpu_shape()),
                                      (cl_mem) (kernel_shape_.gpu_data()),
                                      (cl_mem) (pad_.gpu_data()),
                                      (cl_mem) (stride_.gpu_data()),
                                      (cl_mem) col_buff, col_buff_off);
      }
    }
  }

  inline void greentea_conv_col2im_gpu(const Dtype* col_buff,
                                       const int col_buff_off, Dtype* data,
                                       const int data_off) {
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(
        this->device_->id());

    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      if (this->use_skernel_) {
        greentea_col2im_sk_gpu<Dtype>(&program, &ctx, (cl_mem) col_buff,
                                      conv_in_channels_,
                                      conv_input_shape_.cpu_data()[1],
                                      conv_input_shape_.cpu_data()[2],
                                      kernel_shape_.cpu_data()[0],
                                      kernel_shape_.cpu_data()[1],
                                      pad_.cpu_data()[0], pad_.cpu_data()[1],
                                      stride_.cpu_data()[0],
                                      stride_.cpu_data()[1],
                                      kstride_.cpu_data()[0],
                                      kstride_.cpu_data()[1], (cl_mem) data,
                                      data_off);
      } else {
        greentea_col2im_gpu<Dtype>(&program, &ctx, (cl_mem) col_buff,
                                   col_buff_off, conv_in_channels_,
                                   conv_input_shape_.cpu_data()[1],
                                   conv_input_shape_.cpu_data()[2],
                                   kernel_shape_.cpu_data()[0],
                                   kernel_shape_.cpu_data()[1],
                                   pad_.cpu_data()[0], pad_.cpu_data()[1],
                                   stride_.cpu_data()[0], stride_.cpu_data()[1],
                                   (cl_mem) data, data_off);
      }
    } else {
      if (this->use_skernel_) {
        greentea_col2im_ndsk_gpu<Dtype>(&program, &ctx, (cl_mem) col_buff,
                                        col_buff_off, num_spatial_axes_,
                                        num_kernels_col2im_,
                                        (cl_mem) (conv_input_shape_.gpu_data()),
                                        (cl_mem) (col_buffer_.gpu_shape()),
                                        (cl_mem) (kernel_shape_.gpu_data()),
                                        (cl_mem) (pad_.gpu_data()),
                                        (cl_mem) (stride_.gpu_data()),
                                        (cl_mem) (kstride_.gpu_data()),
                                        (cl_mem) data,
                                        data_off);
      } else {
        greentea_col2im_nd_gpu<Dtype>(&program, &ctx, (cl_mem) col_buff,
                                      col_buff_off, num_spatial_axes_,
                                      0,
                                      num_kernels_col2im_,
                                      (cl_mem) (conv_input_shape_.gpu_data()),
                                      (cl_mem) (col_buffer_.gpu_shape()),
                                      (cl_mem) (kernel_shape_.gpu_data()),
                                      (cl_mem) (pad_.gpu_data()),
                                      (cl_mem) (stride_.gpu_data()),
                                      (cl_mem) data,
                                      data_off);
      }
    }
  }
#endif  // USE_GREENTEA
#endif  // !CPU_ONLY

  int num_kernels_im2col_;
  int num_kernels_col2im_;
  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int kernel_dim_;
  int col_offset_;
  int output_offset_;

  bool use_skernel_;

  Blob<Dtype> col_buffer_;
  Blob<Dtype> bias_multiplier_;
};




template<typename Dtype>
class ConvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:
  explicit ConvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {
  }

  virtual inline const char* type() const {
    return "Convolution";
  }

  virtual size_t ForwardFlops() {
    size_t group = this->group_;
    size_t N = 1;
    size_t M = this->num_output_ / group;
    size_t K = this->channels_;
    const int* kshape = this->kernel_shape_.cpu_data();
    for (int i = 0; i < this->output_shape_.size(); ++i) {
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

template <typename Dtype>
class DeconvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:
  explicit DeconvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "Deconvolution"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return true; }
  virtual void compute_output_shape();
};

#ifdef USE_CUDNN
/*
 * @brief cuDNN implementation of ConvolutionLayer.
 *        Fallback to ConvolutionLayer for CPU mode.
 *
 * cuDNN accelerates convolution through forward kernels for filtering and bias
 * plus backward kernels for the gradient w.r.t. the filters, biases, and
 * inputs. Caffe + cuDNN further speeds up the computation through forward
 * parallelism across groups and backward parallelism across gradients.
 *
 * The CUDNN engine does not have memory overhead for matrix buffers. For many
 * input and filter regimes the CUDNN engine is faster than the CAFFE engine,
 * but for fully-convolutional models and large inputs the CAFFE engine can be
 * faster as long as it fits in memory.
 */
template <typename Dtype>
class CuDNNConvolutionLayer : public ConvolutionLayer<Dtype> {
 public:
  explicit CuDNNConvolutionLayer(const LayerParameter& param)
  : ConvolutionLayer<Dtype>(param), handles_setup_(false) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~CuDNNConvolutionLayer();

 protected:
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool handles_setup_;
  cudnnHandle_t* handle_;
  cudaStream_t* stream_;

  // algorithms for forward and backwards convolutions
  cudnnConvolutionFwdAlgo_t *fwd_algo_;
  cudnnConvolutionBwdFilterAlgo_t *bwd_filter_algo_;
  cudnnConvolutionBwdDataAlgo_t *bwd_data_algo_;

  vector<cudnnTensorDescriptor_t> bottom_descs_, top_descs_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  vector<cudnnConvolutionDescriptor_t> conv_descs_;
  int bottom_offset_, top_offset_, bias_offset_;

  size_t *workspace_fwd_sizes_;
  size_t *workspace_bwd_data_sizes_;
  size_t *workspace_bwd_filter_sizes_;
  size_t workspaceSizeInBytes;  // size of underlying storage
  void *workspaceData;  // underlying storage
  void **workspace;  // aliases into workspaceData
};
#endif

/**
 * @brief A helper for image operations that rearranges image regions into
 *        column vectors.  Used by ConvolutionLayer to perform convolution
 *        by matrix multiplication.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template<typename Dtype>
class Im2colLayer : public Layer<Dtype> {
 public:
  explicit Im2colLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const {
    return "Im2col";
  }
  virtual inline int ExactNumBottomBlobs() const {
    return 1;
  }
  virtual inline int ExactNumTopBlobs() const {
    return 1;
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

  /// @brief The spatial dimensions of a filter kernel.
  Blob<int> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  Blob<int> stride_;
  /// @brief The spatial dimensions of the padding.
  Blob<int> pad_;

  int num_spatial_axes_;
  int bottom_dim_;
  int top_dim_;

  int channel_axis_;
  int num_;
  int channels_;

  bool force_nd_im2col_;
};

// Forward declare PoolingLayer and SplitLayer for use in LRNLayer.
template<typename Dtype> class PoolingLayer;
template<typename Dtype> class SplitLayer;

/**
 * @brief Normalize the input in a local region across or within feature maps.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template<typename Dtype>
class LRNLayer : public Layer<Dtype> {
 public:
  explicit LRNLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const {
    return "LRN";
  }
  virtual inline int ExactNumBottomBlobs() const {
    return 1;
  }
  virtual inline int ExactNumTopBlobs() const {
    return 1;
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

  virtual void CrossChannelForward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top);
  virtual void CrossChannelForward_gpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top);
  virtual void WithinChannelForward(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top);
  virtual void CrossChannelBackward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom);
  virtual void CrossChannelBackward_gpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom);
  virtual void WithinChannelBackward(const vector<Blob<Dtype>*>& top,
                                     const vector<bool>& propagate_down,
                                     const vector<Blob<Dtype>*>& bottom);

  int size_;
  int pre_pad_;
  Dtype alpha_;
  Dtype beta_;
  Dtype k_;
  int num_;
  int channels_;
  int height_;
  int width_;

  // Fields used for normalization ACROSS_CHANNELS
  // scale_ stores the intermediate summing results
  Blob<Dtype> scale_;

  // Fields used for normalization WITHIN_CHANNEL
  shared_ptr<SplitLayer<Dtype> > split_layer_;
  vector<Blob<Dtype>*> split_top_vec_;
  shared_ptr<PowerLayer<Dtype> > square_layer_;
  Blob<Dtype> square_input_;
  Blob<Dtype> square_output_;
  vector<Blob<Dtype>*> square_bottom_vec_;
  vector<Blob<Dtype>*> square_top_vec_;
  shared_ptr<PoolingLayer<Dtype> > pool_layer_;
  Blob<Dtype> pool_output_;
  vector<Blob<Dtype>*> pool_top_vec_;
  shared_ptr<PowerLayer<Dtype> > power_layer_;
  Blob<Dtype> power_output_;
  vector<Blob<Dtype>*> power_top_vec_;
  shared_ptr<EltwiseLayer<Dtype> > product_layer_;
  Blob<Dtype> product_input_;
  vector<Blob<Dtype>*> product_bottom_vec_;
};

#ifdef USE_CUDNN

template <typename Dtype>
class CuDNNLRNLayer : public LRNLayer<Dtype> {
 public:
  explicit CuDNNLRNLayer(const LayerParameter& param)
      : LRNLayer<Dtype>(param), handles_setup_(false) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~CuDNNLRNLayer();

 protected:
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool handles_setup_;
  cudnnHandle_t             handle_;
  cudnnLRNDescriptor_t norm_desc_;
  cudnnTensorDescriptor_t bottom_desc_, top_desc_;

  int size_;
  Dtype alpha_, beta_, k_;
};

template <typename Dtype>
class CuDNNLCNLayer : public LRNLayer<Dtype> {
 public:
  explicit CuDNNLCNLayer(const LayerParameter& param)
      : LRNLayer<Dtype>(param), handles_setup_(false), tempDataSize(0),
        tempData1(NULL), tempData2(NULL) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~CuDNNLCNLayer();

 protected:
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool handles_setup_;
  cudnnHandle_t             handle_;
  cudnnLRNDescriptor_t norm_desc_;
  cudnnTensorDescriptor_t bottom_desc_, top_desc_;

  int size_, pre_pad_;
  Dtype alpha_, beta_, k_;

  size_t tempDataSize;
  void *tempData1, *tempData2;
};

#endif

/**
 * @brief Pools the input image by taking the max, average, etc. within regions.
 *
 * For whole image processing, reducing redundancy.
 */
template<typename Dtype>
class PoolingLayer : public Layer<Dtype> {
 public:
  explicit PoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

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

  virtual inline const char* type() const {
    return "Pooling";
  }
  virtual inline int ExactNumBottomBlobs() const {
    return 1;
  }
  virtual inline int MinTopBlobs() const {
    return 1;
  }
  // MAX POOL layers can output an extra top blob for the mask;
  // others can only output the pooled inputs.
  virtual inline int MaxTopBlobs() const {
    return
        (this->layer_param_.pooling_param().pool()
            == PoolingParameter_PoolMethod_MAX) ? 2 : 1;
  }

  Blob<int> kernel_shape_;
  Blob<int> ext_kernel_shape_;
  Blob<int> stride_;
  Blob<int> pad_;
  Blob<int> kstride_;
  Blob<int> size_;
  Blob<int> pooled_size_;

  int channel_axis_;
  int num_spatial_axes_;
  int channels_;

  bool use_skernel_;
  bool global_pooling_;

  int max_top_blobs_;
  Blob<Dtype> rand_idx_;
  Blob<int> max_idx_;
};

#ifdef USE_CUDNN
/*
 * @brief cuDNN implementation of PoolingLayer.
 *        Fallback to PoolingLayer for CPU mode.
 */
template <typename Dtype>
class CuDNNPoolingLayer : public PoolingLayer<Dtype> {
 public:
  explicit CuDNNPoolingLayer(const LayerParameter& param)
  : PoolingLayer<Dtype>(param), handles_setup_(false) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~CuDNNPoolingLayer();
  // Currently, cuDNN does not support the extra top blob.
  virtual inline int MinTopBlobs() const {return -1;}
  virtual inline int ExactNumTopBlobs() const {return 1;}

 protected:
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool handles_setup_;
  cudnnHandle_t handle_;
  cudnnTensorDescriptor_t bottom_desc_, top_desc_;
  cudnnPoolingDescriptor_t pooling_desc_;
  cudnnPoolingMode_t mode_;
};
#endif

/**
 * @brief Does spatial pyramid pooling on the input image
 *        by taking the max, average, etc. within regions
 *        so that the result vector of different sized
 *        images are of the same size.
 */
template<typename Dtype>
class SPPLayer : public Layer<Dtype> {
 public:
  explicit SPPLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SPP"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  // calculates the kernel and stride dimensions for the pooling layer,
  // returns a correctly configured LayerParameter for a PoolingLayer
  virtual LayerParameter GetPoolingParam(const int pyramid_level,
                                         const int bottom_h, const int bottom_w,
                                         const SPPParameter spp_param);

  int pyramid_height_;
  int bottom_h_, bottom_w_;
  int num_;
  int channels_;
  int kernel_h_, kernel_w_;
  int pad_h_, pad_w_;
  bool reshaped_first_time_;

  /// the internal Split layer that feeds the pooling layers
  shared_ptr<SplitLayer<Dtype> > split_layer_;
  /// top vector holder used in call to the underlying SplitLayer::Forward
  vector<Blob<Dtype>*> split_top_vec_;
  /// bottom vector holder used in call to the underlying PoolingLayer::Forward
  vector<vector<Blob<Dtype>*>*> pooling_bottom_vecs_;
  /// the internal Pooling layers of different kernel sizes
  vector<shared_ptr<PoolingLayer<Dtype> > > pooling_layers_;
  /// top vector holders used in call to the underlying PoolingLayer::Forward
  vector<vector<Blob<Dtype>*>*> pooling_top_vecs_;
  /// pooling_outputs stores the outputs of the PoolingLayers
  vector<Blob<Dtype>*> pooling_outputs_;
  /// the internal Flatten layers that the Pooling layers feed into
  vector<FlattenLayer<Dtype>*> flatten_layers_;
  /// top vector holders used in call to the underlying FlattenLayer::Forward
  vector<vector<Blob<Dtype>*>*> flatten_top_vecs_;
  /// flatten_outputs stores the outputs of the FlattenLayers
  vector<Blob<Dtype>*> flatten_outputs_;
  /// bottom vector holder used in call to the underlying ConcatLayer::Forward
  vector<Blob<Dtype>*> concat_bottom_vec_;
  /// the internal Concat layers that the Flatten layers feed into
  shared_ptr<ConcatLayer<Dtype> > concat_layer_;
};

}  // namespace caffe

#endif  // CAFFE_VISION_LAYERS_HPP_
