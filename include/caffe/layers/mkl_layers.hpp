#ifndef CAFFE_MKL2017_LAYERS_HPP_
#define CAFFE_MKL2017_LAYERS_HPP_

#include <string>
#include <vector>

#include "boost/enable_shared_from_this.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/neuron_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "mkl_dnn_cppwrapper.h"

namespace caffe {

template <typename Dtype, bool is_diff>
struct MKLMemoryDescriptor : PrvMemDescr,
    boost::enable_shared_from_this<MKLMemoryDescriptor<Dtype, is_diff> > {
  MKLMemoryDescriptor() : layout_usr(NULL), layout_int(NULL),
          internal_ptr(NULL), convert_to_int(NULL), convert_from_int(NULL),
          name("UKNOWN") {}
  ~MKLMemoryDescriptor() {
    dnnLayoutDelete<Dtype>(layout_usr);
    dnnLayoutDelete<Dtype>(layout_int);
    dnnReleaseBuffer<Dtype>(internal_ptr);
    dnnDelete<Dtype>(convert_to_int);
    dnnDelete<Dtype>(convert_from_int);
  }

  shared_ptr<MKLMemoryDescriptor<Dtype, is_diff> > get_shared_ptr() {
    return this->shared_from_this();
  }

  dnnLayout_t layout_usr;
  dnnLayout_t layout_int;
  Dtype* internal_ptr;
  dnnPrimitive_t convert_to_int;
  dnnPrimitive_t convert_from_int;
  std::string name;  // for debugging purposes
  void create_conversions() {
    if (layout_int
        && !dnnLayoutCompare<Dtype>(layout_usr, layout_int)) {
      CHECK(layout_usr);
      int status = dnnConversionCreate<Dtype>(&convert_to_int, layout_usr,
              layout_int);
      CHECK_EQ(status, 0) << "Failed creation convert_to_int with status "
              << status << "\n";
      status = dnnConversionCreate<Dtype>(&convert_from_int, layout_int,
              layout_usr);
      CHECK_EQ(status, 0) << "Failed creation convert_from_int with status "
              << status << "\n";
      status = dnnAllocateBuffer<Dtype>(
              reinterpret_cast<void **>(&internal_ptr), layout_int);
      CHECK_EQ(status, 0)
              << "Failed internal_ptr memory allocation with status "
              << status << "\n";

      caffe_set(prv_count(), Dtype(0), internal_ptr);
      // std::cout << "Conversions created.\n" << std::endl;
    }
  }
  virtual size_t prv_count() {
      return dnnLayoutGetMemorySize<Dtype>(layout_int) / sizeof(Dtype);
  }
  virtual void convert_from_prv(void* prv_ptr, void* cpu_ptr);
  virtual PrvDescrType get_descr_type() {return PRV_DESCR_MKL2017;}

  // The last get_converted_prv() argument is a hack for reusing
  // in backward a conversion done already in the forward direction.
  Dtype* get_converted_prv(Blob<Dtype> * blob, bool set_prv_ptr,
          MKLMemoryDescriptor<Dtype, is_diff>* converted_in_fwd = NULL);
};

template <typename Dtype>
struct MKLData : MKLMemoryDescriptor<Dtype, false>
{};

template <typename Dtype>
struct MKLDiff : MKLMemoryDescriptor<Dtype, true>
{};

template <typename Dtype>
class MKLConvolutionLayer : public ConvolutionLayer<Dtype> {
 public:
  explicit MKLConvolutionLayer(const LayerParameter& param);

  virtual inline const char* type() const { return "DnnConvolution"; }
  virtual ~MKLConvolutionLayer();

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
  // Customized methods
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void compute_output_shape();

 private:
  /* Fwd step */
  shared_ptr<MKLData<Dtype> > fwd_bottom_data, fwd_top_data, fwd_filter_data,
                                 fwd_bias_data;
  dnnPrimitive_t convolutionFwd;

  /* Bwd data step */
  shared_ptr<MKLDiff<Dtype> > bwdd_top_diff, bwdd_bottom_diff;
  shared_ptr<MKLData<Dtype> > bwdd_filter_data;
  dnnPrimitive_t convolutionBwdData;

  /* Bwd filter step */
  shared_ptr<MKLDiff<Dtype> > bwdf_top_diff, bwdf_filter_diff;
  shared_ptr<MKLData<Dtype> > bwdf_bottom_data;
  dnnPrimitive_t convolutionBwdFilter;

  /* Bwd bias step */
  shared_ptr<MKLDiff<Dtype> > bwdb_top_diff, bwdb_bias_diff;
  dnnPrimitive_t convolutionBwdBias;

  // TODO: temp. compatibility vs. older cafe
  size_t width_,
         height_,
         width_out_,
         height_out_,
         kernel_w_,
         kernel_h_,
         stride_w_,
         stride_h_;
  int    pad_w_,
         pad_h_;
};


/**
 * @brief Normalize the input in a local region across feature maps.
 */

template <typename Dtype>
class MKLLRNLayer : public Layer<Dtype> {
 public:
  explicit MKLLRNLayer(const LayerParameter& param)
      : Layer<Dtype>(param), layout_usr_(NULL) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~MKLLRNLayer();

  virtual inline const char* type() const { return "DnnLRN"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void CrossChannelForward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void CrossChannelBackward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void CrossChannelForward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void CrossChannelBackward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

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
 private:
  dnnPrimitive_t lrnFwd, lrnBwd;
  shared_ptr<MKLData<Dtype> > fwd_top_data;
  shared_ptr<MKLDiff<Dtype> > bwd_bottom_diff;
  Dtype *lrn_buffer_;
  dnnLayout_t layout_usr_;
};



template <typename Dtype>
class MKLPoolingLayer : public Layer<Dtype> {
 public:
  explicit MKLPoolingLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      fwd_top_data    (new MKLData<Dtype>()),
      fwd_bottom_data (new MKLData<Dtype>()),
      bwd_top_diff    (new MKLDiff<Dtype>()),
      bwd_bottom_diff (new MKLDiff<Dtype>()),
      poolingFwd(NULL), poolingBwd(NULL) {}
  ~MKLPoolingLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DnnPooling"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  // MAX POOL layers can output an extra top blob for the mask;
  // others can only output the pooled inputs.
  virtual inline int MaxTopBlobs() const {
    return (this->layer_param_.pooling_param().pool() ==
            PoolingParameter_PoolMethod_MAX) ? 2 : 1;
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

  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int pad_h_, pad_w_;
  int channels_;
  int height_, width_;
  int pooled_height_, pooled_width_;
  bool global_pooling_;
  Blob<Dtype> rand_idx_;
  Blob<size_t> max_idx_;

 private:
  size_t kernel_size[2],
         kernel_stride[4];
  int src_offset[2];
  shared_ptr<MKLData<Dtype> > fwd_top_data, fwd_bottom_data;
  shared_ptr<MKLDiff<Dtype> > bwd_top_diff, bwd_bottom_diff;

  dnnPrimitive_t poolingFwd, poolingBwd;
};

template <typename Dtype>
class MKLReLULayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides ReLUParameter relu_param,
   *     with ReLULayer options:
   *   - negative_slope (\b optional, default 0).
   *     the value @f$ \nu @f$ by which negative values are multiplied.
   */
  explicit MKLReLULayer(const LayerParameter& param)
    : NeuronLayer<Dtype>(param),
      fwd_bottom_data_ (new MKLData<Dtype>()),
      bwd_top_diff_    (new MKLDiff<Dtype>()) {}
  ~MKLReLULayer();

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DnnReLU"; }

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
  shared_ptr<MKLData<Dtype> > fwd_bottom_data_;
  shared_ptr<MKLDiff<Dtype> > bwd_top_diff_;
  dnnPrimitive_t reluFwd_, reluBwd_;
};

#ifdef USE_MKL2017_NEW_API
template <typename Dtype>
class MKLConcatLayer : public Layer<Dtype> {
 public:
  explicit MKLConcatLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
      fwd_top_data_    (new MKLData<Dtype>()),
      bwd_top_diff_    (new MKLDiff<Dtype>()) {
      }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);
  ~MKLConcatLayer();

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
  dnnPrimitive_t concatFwd_;
  dnnPrimitive_t concatBwd_;
  shared_ptr<MKLData<Dtype> > fwd_top_data_;
  vector<shared_ptr<MKLData<Dtype> > > fwd_bottom_data_;
  shared_ptr<MKLDiff<Dtype> > bwd_top_diff_;
  vector<shared_ptr<MKLDiff<Dtype> > > bwd_bottom_diff_;
  size_t *split_channels_;

  size_t width_;
  size_t height_;
  size_t channels_;
  size_t num_;
  size_t num_concats_;
};

template <typename Dtype>
class MKLBatchNormLayer : public Layer<Dtype> {
 public:
  explicit MKLBatchNormLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
        fwd_top_data    (new MKLData<Dtype>()),
        bwd_bottom_diff (new MKLDiff<Dtype>()),
        batchNormFwd(NULL), batchNormBwdData(NULL),
        batchNormBwdScaleShift(NULL) {
       }
  virtual ~MKLBatchNormLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MKLBatchNorm"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

//  Dtype moving_average_fraction_;
  Dtype eps_;
  bool use_weight_bias_;
  bool bias_term_;
  int num_;
  int channels_;
  int height_;
  int width_;

 private:
  shared_ptr<MKLData<Dtype> > fwd_top_data;
  shared_ptr<MKLDiff<Dtype> > bwd_bottom_diff;
  dnnPrimitive_t batchNormFwd, batchNormBwdData, batchNormBwdScaleShift;
  Dtype *workspace_buffer_;
  Dtype *scaleShift_buffer_;
  dnnLayout_t layout_usr_;
};

template <typename Dtype>
class MKLSplitLayer : public Layer<Dtype> {
 public:
  explicit MKLSplitLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
        bwd_bottom_diff (new MKLDiff<Dtype>()),
        sumPrimitive(NULL) {
       }
  virtual ~MKLSplitLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Split"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
  shared_ptr<MKLDiff<Dtype> > bwd_bottom_diff;
  vector<shared_ptr<MKLDiff<Dtype> > > bwd_top_diff;
  vector<Dtype> coeffs_;
  size_t num_tops;
  dnnPrimitive_t sumPrimitive;
};

template <typename Dtype>
class MKLEltwiseLayer : public Layer<Dtype> {
 public:
  explicit MKLEltwiseLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
        fwd_top_data       (new MKLData<Dtype>()),
        sumPrimitive(NULL) {
       }
  virtual ~MKLEltwiseLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Eltwise"; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
  shared_ptr<MKLData<Dtype> > fwd_top_data;
  vector<shared_ptr<MKLData<Dtype> > > fwd_bottom_data;

  dnnPrimitive_t sumPrimitive;
  dnnPrimitive_t convertPrimitive;

  EltwiseParameter_EltwiseOp op_;
  vector<Dtype> coeffs_;
  Blob<int> max_idx_;
  size_t num_bottoms;

  bool stable_prod_grad_;
};
#endif  // #ifdef USE_MKL2017_NEW_API

}  // namespace caffe
#endif  // #ifndef CAFFE_MKL2017_LAYERS_HPP_
