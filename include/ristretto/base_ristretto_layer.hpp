#ifndef CAFFE_BASE_RISTRETTO_LAYER_HPP_
#define CAFFE_BASE_RISTRETTO_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides quantization methods used by other Ristretto layers.
 */
template <typename Dtype>
class BaseRistrettoLayer{
 public:
  explicit BaseRistrettoLayer();
 protected:
  void QuantizeLayerOutputs_cpu(Dtype* top_data, const int top_count);
  void QuantizeLayerOutputs_gpu(Dtype* top_data, const int top_count);
  void QuantizeWeights_cpu(Dtype* weight, const int cnt_weight, Dtype* bias,
      const int cnt_bias, const int rounding);
  void QuantizeWeights_gpu(Dtype* weight, const int cnt_weight, Dtype* bias,
      const int cnt_bias, const int rounding);
  /**
   * @brief Trim data to dynamic fixed point.
   * @param fl The number of bits in fractional part.
   */
  void Trim2FixedPoint_cpu(Dtype* data, const int cnt, const int bit_width,
      const int rounding, int fl);
  void Trim2FixedPoint_gpu(Dtype* data, const int cnt, const int bit_width,
      const int rounding, int fl);
  /**
   * @brief Trim data to mini floating point.
   * @param bw_mant The number of bits used to represent the mantissa.
   * @param bw_exp The number of bits used to represent the exponent.
   */
  void Trim2FloatingPoint_cpu(Dtype* data, const int cnt, const int bw_mant,
      const int bw_exp, const int rounding);
  void Trim2FloatingPoint_gpu(Dtype* data, const int cnt, const int bw_mant,
      const int bw_exp, const int rounding);
  /**
   * @brief Trim data to power-of-two numbers.
   * @param min_exp The smallest quantized value is 2^min_exp.
   * @param min_exp The largest quantized value is 2^max_exp.
   */
  void Trim2PowerOf2_cpu(Dtype* data, const int cnt, const int min_exp,
      const int max_exp, const int rounding);
  void Trim2PowerOf2_gpu(Dtype* data, const int cnt, const int min_exp,
      const int max_exp, const int rounding);
  /**
   * @brief Generate random number in [0,1) range.
   */
  inline double RandUniform_cpu();
  // The number of bits used for dynamic fixed point parameters and layer
  // outputs.
  int bw_params_, bw_layer_out_;
  // The fractional length of dynamic fixed point numbers.
  int fl_params_, fl_layer_out_;
  // The number of bits used to represent mantissa and exponent of mini floating
  // point numbers.
  int fp_mant_, fp_exp_;
  // Power-of-two numbers are in range +/- [2^min_exp, 2^max_exp].
  int pow_2_min_exp_, pow_2_max_exp_;
  // The rounding mode for quantization and the quantization scheme.
  int rounding_, precision_;
};

/**
 * @brief Convolutional layer with quantized layer parameters and outputs.
 */
template <typename Dtype>
class ConvolutionRistrettoLayer : public ConvolutionLayer<Dtype>,
      public BaseRistrettoLayer<Dtype> {
 public:
  explicit ConvolutionRistrettoLayer(const LayerParameter& param);
  virtual inline const char* type() const { return "ConvolutionRistretto"; }

 protected:
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  vector<shared_ptr<Blob<Dtype> > > weights_quantized_;
};

/**
 * @brief Inner product (fully connected) layer with quantized layer parameters
 * and outputs.
 */
template <typename Dtype>
class FcRistrettoLayer : public InnerProductLayer<Dtype>,
      public BaseRistrettoLayer<Dtype>{
 public:
  explicit FcRistrettoLayer(const LayerParameter& param);
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "FcRistretto"; }
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
  vector<shared_ptr<Blob<Dtype> > > weights_quantized_;
};

/**
 * @brief Data layer with quantized images.
 */
template <typename Dtype>
class DataRistrettoLayer : public BasePrefetchingDataLayer<Dtype>,
      public BaseRistrettoLayer<Dtype>{
 public:
  explicit DataRistrettoLayer(const LayerParameter& param);
  virtual ~DataRistrettoLayer();
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "DataRistretto"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  DataReader reader_;
};

/**
 * @brief Local response normalization (LRN) layer with mini floating point
 * layer inputs, intermediate results and outputs.
 */
template <typename Dtype>
class LRNRistrettoLayer : public LRNLayer<Dtype>,
      public BaseRistrettoLayer<Dtype>{
 public:
  explicit LRNRistrettoLayer(const LayerParameter& param);
  virtual inline const char* type() const { return "LRNRistretto"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  void Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void CrossChannelForward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
};

}  // namespace caffe

#endif  // CAFFE_BASE_RISTRETTO_LAYER_HPP_
