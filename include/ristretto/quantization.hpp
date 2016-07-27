#ifndef QUANTIZATION_HPP_
#define QUANTIZATION_HPP_

#include "caffe/caffe.hpp"

using caffe::string;
using caffe::vector;
using caffe::Net;

/**
 * @brief Approximate 32-bit floating point networks.
 *
 * This is the Ristretto tool. Use it to generate file descriptions of networks
 * which use reduced word width arithmetic.
 */
class Quantization {
public:
  explicit Quantization(string model, string weights, string model_quantized,
      int iterations, string trimming_mode, double error_margin, string gpus);
  void QuantizeNet();
private:
  void CheckWritePermissions(const string path);
  void SetGpu();
  /**
   * @brief Score network.
   * @param accuracy Reports the network's accuracy according to
   * accuracy_number.
   * @param do_stats: Find the maximal values in each layer.
   * @param score_number The accuracy layer that matters.
   *
   * For networks with multiple accuracy layers, set score_number to the
   * appropriate value. For example, for BVLC GoogLeNet, use score_number=7.
   */
  void RunForwardBatches(const int iterations, Net<float>* caffe_net,
      float* accuracy, const bool do_stats = false, const int score_number = 0);
  /**
   * @brief Quantize convolutional and fully connected layers to dynamic fixed
   * point.
   * The parameters and layer activations get quantized and the resulting
   * network will be tested.
   * Find the required number of bits required for parameters and layer
   * activations (which might differ from each other).
   */
  void Quantize2DynamicFixedPoint();
  /**
   * @brief Quantize convolutional and fully connected layers to minifloat.
   * Parameters and layer activations share the same numerical representation.
   * This simulates hardware arithmetic which uses IEEE-754 standard (with some
   * small optimizations).
   */
  void Quantize2MiniFloat();
  /**
   * @brief Quantize convolutional and fully connected parameters to
   * integer-power-of-two numbers.
   * Activations in convolutional and fully connected layers are quantized to
   * dynamic fixed point.
   * The parameters (excluding bias) can be written as +/-2^exp where exp
   * is in [-8,..,-1].
   * In a hardware implementation, the parameters can be represented with 4
   * bits. 1 bits is required for the sign, and 3 bits are required to store the
   * exponent.
   * The quantized layers don't need any multipliers in hardware.
   */
  void Quantize2IntegerPowerOf2Weights();
  /**
   * @brief Change network to dynamic fixed point.
   */
  void EditNetDescriptionDynamicFixedPoint(caffe::NetParameter* param,
      const string layers_2_quantize, const string network_part,
      const int bw_conv, const int bw_fc, const int bw_in, const int bw_out);
  /**
   * @brief Change network to minifloat.
   */
  void EditNetDescriptionMiniFloat(caffe::NetParameter* param,
      const int bitwidth);
  /**
   * @brief Change network parameters to integer-power-of-two numbers.
   */
  void EditNetDescriptionIntegerPowerOf2Weights(caffe::NetParameter* param);
  /**
   * @brief Find the integer length for dynamic fixed point parameters of a
   * certain layer.
   */
  int GetIntegerLengthParams(const string layer_name);
  /**
   * @brief Find the integer length for dynamic fixed point inputs of a certain
   * layer.
   */
  int GetIntegerLengthIn(const string layer_name);
  /**
   * @brief Find the integer length for dynamic fixed point outputs of a certain
   * layer.
   */
  int GetIntegerLengthOut(const string layer_name);

  string model_;
  string weights_;
  string model_quantized_;
  int iterations_;
  string trimming_mode_;
  double error_margin_;
  string gpus_;
  float test_score_baseline_;
  // The maximal absolute values of layer inputs, parameters and
  // layer outputs.
  vector<float> max_in_, max_params_, max_out_;

  // The integer bits for dynamic fixed point layer inputs, parameters and
  // layer outputs.
  vector<int> il_in_, il_params_, il_out_;
  // The name of the layers that need to be quantized to dynamic fixed point.
  vector<string> layer_names_;
  // The number of bits used for dynamic fixed point layer inputs, parameters
  // and layer outputs.
  int bw_in_, bw_conv_params_, bw_fc_params_, bw_out_;

  // The number of bits used for minifloat exponent.
  int exp_bits_;
};

#endif // QUANTIZATION_HPP_
