#ifndef QUANTIZATION_HPP_
#define QUANTIZATION_HPP_

#include "caffe/caffe.hpp"

using caffe::string;
using caffe::vector;
using caffe::Net;

/**
 * @brief Approximate 32-bit floating point networks.
 */
class Quantization {
public:
  explicit Quantization(string model, string weights, string model_quantized,
      int iterations, string trimming_mode, double error_margin, string gpus);
  void QuantizeNet();
private:
  void CheckWritePermissions(const string path);
  void SetGpu();
  //score_number: assume the net just has one accuracy layer
  /**
   * @brief Score network.
   * @param accuracy Reports the network's accuracy according to
   * accuracy_number.
   * @param score_number The accuracy layer that matters.
   *
   * For networks with multiple accuracy layers, set score_number to the
   * appropriate value. For example, if you are interested in the third accuracy
   * layer's output, set score_number to 2.
   */
  void RunForwardBatches(const int iterations, Net<float>* caffe_net,
      float* accuracy, const int score_number = 0);
  /**
   * @brief Quantize convolutional and fully connected layers to dynamic fixed
   * point.
   * The parameters and layer outputs get quantized and the resulting network
   * will be tested.
   * This finds the required number of bits required for parameters and layer
   * outputs (which might differ from each other).
   */
  void Quantize2FixedPoint();
  /**
   * @brief Quantize convolutional and fully connected layers to mini floating
   * point.
   * Parameters and layer outputs share the same numerical representation.
   * This simulates hardware arithmetic which uses IEEE-754 standard (with some
   * small optimizations).
   */
  void Quantize2MiniFloatingPoint();
  /**
   * @Quantize convolutional and fully connected parameters to power-of-two
   * numbers.
   * The parameters (excluding bias) can be written as +/-2^exp where exp
   * is in [-8,..,-1].
   * In a hardware implementation, the parameters can be represented with 4
   * bits. 1 bits is required for the sign, and 3 bits are required to store the
   * exponent. Experiments show that other exponents such as 0 and -9 are not
   * important for a good network accuracy.
   * The quantized layers don't need any multipliers in hardware.
   */
  void Quantize2PowerOf2Weights();
  /**
   * @brief Change network to dynamic fixed point.
   */
  void EditNetDescriptionFixedPoint(caffe::NetParameter* param,
      const string layers_2_quantize, const string network_part,
      const int bw_conv, const int bw_fc, const int bw_out);
  /**
   * @brief Change network to mini floating point.
   */
  void EditNetDescriptionMiniFloat(caffe::NetParameter* param,
      const int bitwidth);
  /**
   * @brief Change network parameters to power-of-two numbers.
   */
  void EditNetDescriptionPower2Weights(caffe::NetParameter* param);
  /**
   * @brief Find the integer length for fixed point parameters of a certain
   * layer.
   */
  int GetIntegerLengthParams(const string layer_name);
  /**
   * @brief Find the integer length for fixed point outputs of a certain layer.
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
  Net<float>* baseline_net_;
  float accuracy_drop_threashold_;

  // The integer bits for fixed point parameters and layer outputs.
  vector<int> il_params_, il_out_;
  // The name of the layers that need to be quantized to fixed point.
  vector<string> layer_names_;
  // The number of bits used for fixed point parameters and layer outputs.
  int bw_conv_params_, bw_fc_params_, bw_out_;

  // The number of bits used for mini floating point exponent.
  int exp_bits_;
};

#endif // QUANTIZATION_HPP_
