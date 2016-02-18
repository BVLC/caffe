#ifndef CAFFE_MKLDNN_LAYERS_HPP_
#define CAFFE_MKLDNN_LAYERS_HPP_

#include <string>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "boost/enable_shared_from_this.hpp"

#include "dnn.hpp"

namespace caffe {

template <typename Dtype, bool is_diff>
struct MklDnnMemoryDescriptor : PrvMemDescr, boost::enable_shared_from_this<MklDnnMemoryDescriptor<Dtype, is_diff> > {
  MklDnnMemoryDescriptor() : layout_usr(NULL), layout_int(NULL),
    internal_ptr(NULL), convert_to_int(NULL), convert_from_int(NULL), name("UKNOWN") {};
  ~MklDnnMemoryDescriptor()
  {
    dnnLayoutDelete<Dtype>(layout_usr);
    dnnLayoutDelete<Dtype>(layout_int);
    dnnReleaseBuffer<Dtype>(internal_ptr);
    dnnDelete<Dtype>(convert_to_int);
    dnnDelete<Dtype>(convert_from_int);
  }

  shared_ptr<MklDnnMemoryDescriptor<Dtype, is_diff> > get_shared_ptr() {
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
        && !dnnLayoutCompare<Dtype>(layout_usr, layout_int))
    {
      CHECK(layout_usr);
      int status = dnnConversionCreate<Dtype>(&convert_to_int, layout_usr , layout_int);
      CHECK(status == 0) << "Failed creation convert_to_int with status " << status << "\n";
      status = dnnConversionCreate<Dtype>(&convert_from_int, layout_int , layout_usr);
      CHECK(status == 0) << "Failed creation convert_from_int with status " << status << "\n";
      status = dnnAllocateBuffer<Dtype>((void **)&internal_ptr, layout_int);
      CHECK(status == 0) << "Failed internal_ptr memory allocation with status " << status << "\n";

      memset(internal_ptr, 0, dnnLayoutGetMemorySize<Dtype>(layout_int));
    }
  }
  virtual size_t prv_count() {return dnnLayoutGetMemorySize<Dtype>(layout_int) / sizeof(Dtype);};
  virtual void convert_from_prv(void* prv_ptr, void* cpu_ptr);
  virtual PrvDescrType get_descr_type() {return PRV_DESCR_MKLDNN;};
  Dtype* get_converted_prv(Blob<Dtype> * blob, bool test_prv_layout, bool set_prv_ptr=true);
};

template <typename Dtype>
struct MklDnnData : MklDnnMemoryDescriptor<Dtype, false>
{};

template <typename Dtype>
struct MklDnnDiff : MklDnnMemoryDescriptor<Dtype, true>
{};

template <typename Dtype>
class MklDnnConvolutionLayer : public ConvolutionLayer<Dtype> {
public:
  explicit MklDnnConvolutionLayer(const LayerParameter& param);

  virtual inline const char* type() const { return "DnnConvolution"; }
  virtual ~MklDnnConvolutionLayer();

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // Customized methods
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void compute_output_shape();

private:
  /* Fwd step */
  shared_ptr<MklDnnData<Dtype> > fwd_bottom_data, fwd_top_data, fwd_filter_data, fwd_bias_data;
  dnnPrimitive_t convolutionFwd;

  /* Bwd data step */
  shared_ptr<MklDnnDiff<Dtype> > bwdd_top_diff, bwdd_bottom_diff;
  shared_ptr<MklDnnData<Dtype> > bwdd_filter_data;
  dnnPrimitive_t convolutionBwdData;

#ifndef BWDD_DISABLE_PAD_REMOVING
  /* Temporary workaround for removing padding from bwdd_bottom_diff */
  shared_ptr<MklDnnDiff<Dtype> > bwdd_bottom_diff_no_padding;
  dnnPrimitive_t convert_to_bottom_diff_no_padding;
#endif

  /* Bwd filter step */
  shared_ptr<MklDnnDiff<Dtype> > bwdf_top_diff, bwdf_filter_diff;
  shared_ptr<MklDnnData<Dtype> > bwdf_bottom_data;
  dnnPrimitive_t convolutionBwdFilter;

  /* Bwd bias step */
  shared_ptr<MklDnnDiff<Dtype> > bwdb_top_diff, bwdb_bias_diff;
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
class MklDnnLRNLayer : public Layer<Dtype> {
 public:
  explicit MklDnnLRNLayer(const LayerParameter& param)
      : Layer<Dtype>(param), layout_usr_(NULL) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~MklDnnLRNLayer();

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
  shared_ptr<MklDnnData<Dtype> > fwd_top_data;
  shared_ptr<MklDnnDiff<Dtype> > bwd_bottom_diff;
  Dtype *lrn_buffer_;
  dnnLayout_t layout_usr_;
};



template <typename Dtype>
class MklDnnPoolingLayer : public Layer<Dtype> {
public:
  explicit MklDnnPoolingLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      fwd_top_data    (new MklDnnData<Dtype>()),
      fwd_bottom_data (new MklDnnData<Dtype>()),
      bwd_top_diff    (new MklDnnDiff<Dtype>()),
      bwd_bottom_diff (new MklDnnDiff<Dtype>()),
      poolingFwd(NULL), poolingBwd(NULL)
  {}
  ~MklDnnPoolingLayer();
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
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

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
  shared_ptr<MklDnnData<Dtype> > fwd_top_data, fwd_bottom_data;
  shared_ptr<MklDnnDiff<Dtype> > bwd_top_diff, bwd_bottom_diff;

  dnnPrimitive_t poolingFwd, poolingBwd;
};

template <typename Dtype>
class MklDnnReLULayer : public NeuronLayer<Dtype> {
public:
  /**
   * @param param provides ReLUParameter relu_param,
   *     with ReLULayer options:
   *   - negative_slope (\b optional, default 0).
   *     the value @f$ \nu @f$ by which negative values are multiplied.
   */
  explicit MklDnnReLULayer(const LayerParameter& param)
    : NeuronLayer<Dtype>(param) {}
  ~MklDnnReLULayer();

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DnnReLU"; }

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
  dnnPrimitive_t reluFwd_, reluBwd_;
};

} // namespace caffe
#endif // #ifndef CAFFE_MKLDNN_LAYERS_HPP_
