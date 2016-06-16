#include "boost/algorithm/string.hpp"

#include "caffe/caffe.hpp"
#include "ristretto/quantization.hpp"

using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::string;
using caffe::vector;
using caffe::Blob;
using caffe::LayerParameter;
using caffe::NetParameter;

Quantization::Quantization(string model, string weights, string model_quantized,
      int iterations, string trimming_mode, double error_margin, string gpus){
  this->model_ = model;
  this->weights_ = weights;
  this->model_quantized_ = model_quantized;
  this->iterations_ = iterations;
  this->trimming_mode_ = trimming_mode;
  this->error_margin_ = error_margin;
  this->gpus_ = gpus;

  // Could possibly improve choice of exponent. Experiments show LeNet needs
  // 4bits, but the saturation border is at 3bits.
  this->exp_bits_ = 4;
  this->accuracy_drop_threashold_ = 2;
}

void Quantization::QuantizeNet() {
  CheckWritePermissions(model_quantized_);
  SetGpu();
  // Run the reference floating point network to find baseline accuracy
  baseline_net_ = new Net<float>(model_, caffe::TEST);
  baseline_net_->CopyTrainedLayersFrom(weights_);
  float accuracy;
  RunForwardBatches(this->iterations_, baseline_net_, &accuracy);
  test_score_baseline_ = accuracy;
  // Do network quantization and scoring.
  if (trimming_mode_ == "fixed_point") {
    Quantize2FixedPoint();
  } else if (trimming_mode_ == "mini_floating_point") {
    Quantize2MiniFloatingPoint();
  } else if (trimming_mode_ == "power_of_2_weights") {
    Quantize2PowerOf2Weights();
  } else {
    LOG(ERROR) << "Unknown trimming mode: " << trimming_mode_;
  }
}

void Quantization::CheckWritePermissions(const string path) {
  std::ofstream probe_ofs(path.c_str());
  if (probe_ofs.good()) {
    probe_ofs.close();
    std::remove(path.c_str());
  } else {
    LOG(FATAL) << "Missing write permissions";
  }
}

void Quantization::SetGpu() {
  // Parse GPU ids or use all available devices
  vector<int> gpus;
  if (gpus_ == "all") {
    int count = 0;
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus.push_back(i);
    }
  } else if (gpus_.size()) {
    vector<string> strings;
    boost::split(strings, gpus_, boost::is_any_of(","));
    for (int i = 0; i < strings.size(); ++i) {
      gpus.push_back(boost::lexical_cast<int>(strings[i]));
    }
  } else {
    CHECK_EQ(gpus.size(), 0);
  }
  // Set device id and mode
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
}

void Quantization::RunForwardBatches(const int iterations,
      Net<float>* caffe_net, float* accuracy, const int score_number) {
  LOG(INFO) << "Running for " << iterations << " iterations.";
  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result =
        caffe_net->Forward(bottom_vec, &iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net->blob_names()[
            caffe_net->output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net->blob_names()[
        caffe_net->output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net->blob_loss_weights()[
        caffe_net->output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }
  *accuracy = test_score[score_number] / iterations;
}

void Quantization::Quantize2FixedPoint() {
  // Find the integer length for fixed point numbers
  vector<float> max_out, max_params;
  baseline_net_->RangeInLayers(&layer_names_, &max_out, &max_params);
  // The integer length is chosen such that no saturation occurs.
  // This approximation assumes an infinitely long factional part for integer
  // numbers.
  // For layer outputs, we reduce the integer length by one bit.
  for (int i = 0; i < layer_names_.size(); ++i) {
    il_out_.push_back((int)ceil(log2(max_out[i])));
    il_params_.push_back((int)ceil(log2(max_params[i]) + 1));
  }
  // Debug
  /*
  for (int k = 0; k < layer_names_.size(); ++k) {
    LOG(INFO) << "Layer " << layer_names_[k] <<
        ", integer length output=" << il_out_[k] <<
        ", integer length parameters=" << il_params_[k];
  }
  */

  delete baseline_net_;

  // Score net with fixed point convolution parameters
  NetParameter param;
  caffe::ReadNetParamsFromTextFileOrDie(model_, &param);
  param.mutable_state()->set_phase(caffe::TEST);
  vector<int> test_bw_conv_params;
  vector<float> test_scores_conv_params;
  float accuracy;
  Net<float>* caffe_net;
  for (int bitwidth = 16; bitwidth > 0; bitwidth /= 2) {
    EditNetDescriptionFixedPoint(&param, "Convolution", "Parameters", bitwidth,
        -1, -1);
    caffe_net = new Net<float>(param, NULL);
    caffe_net->CopyTrainedLayersFrom(weights_);
    RunForwardBatches(iterations_, caffe_net, &accuracy);
    test_bw_conv_params.push_back(bitwidth);
    test_scores_conv_params.push_back(accuracy);
    delete caffe_net;
    if ( accuracy * accuracy_drop_threashold_ < test_score_baseline_ ) break;
  }

  // Score net with fixed point inner product parameters
  caffe::ReadNetParamsFromTextFileOrDie(model_, &param);
  param.mutable_state()->set_phase(caffe::TEST);
  vector<int> test_bw_fc_params;
  vector<float> test_scores_fc_params;
  for (int bitwidth = 16; bitwidth > 0; bitwidth /= 2) {
    EditNetDescriptionFixedPoint(&param, "InnerProduct", "Parameters", -1,
        bitwidth, -1);
    caffe_net = new Net<float>(param, NULL);
    caffe_net->CopyTrainedLayersFrom(weights_);
    RunForwardBatches(iterations_, caffe_net, &accuracy);
    test_bw_fc_params.push_back(bitwidth);
    test_scores_fc_params.push_back(accuracy);
    delete caffe_net;
    if ( accuracy * accuracy_drop_threashold_ < test_score_baseline_ ) break;
  }

  // Score net with fixed point layer outputs
  caffe::ReadNetParamsFromTextFileOrDie(model_, &param);
  param.mutable_state()->set_phase(caffe::TEST);
  vector<int> test_bw_layer_out;
  vector<float> test_scores_layer_out;
  for (int bitwidth = 16; bitwidth > 0; bitwidth /= 2) {
    EditNetDescriptionFixedPoint(&param, "Convolution_and_InnerProduct",
        "Output", -1, -1, bitwidth);
    caffe_net = new Net<float>(param, NULL);
    caffe_net->CopyTrainedLayersFrom(weights_);
    RunForwardBatches(iterations_, caffe_net, &accuracy);
    test_bw_layer_out.push_back(bitwidth);
    test_scores_layer_out.push_back(accuracy);
    delete caffe_net;
    if ( accuracy * accuracy_drop_threashold_ < test_score_baseline_ ) break;
  }

  // Choose bitwidth for different network parts
  for (int i = 0; i < test_scores_conv_params.size(); ++i) {
    if (test_scores_conv_params[i] + error_margin_ / 100 >=
          test_score_baseline_)
      bw_conv_params_ = test_bw_conv_params[i];
    else
      break;
  }
  for (int i = 0; i < test_scores_fc_params.size(); ++i) {
    if (test_scores_fc_params[i] + error_margin_ / 100 >=
          test_score_baseline_)
      bw_fc_params_ = test_bw_fc_params[i];
    else
      break;
  }
  for (int i=0; i<test_scores_layer_out.size(); ++i) {
    if (test_scores_layer_out[i] + error_margin_ / 100 >=
          test_score_baseline_)
      bw_out_ = test_bw_layer_out[i];
    else
      break;
  }

  // Score fixed point network
  // This network combines fixed point parameters in convolutional and inner
  // product layers, as well as fixed point outputs in these layers.
  caffe::ReadNetParamsFromTextFileOrDie(model_, &param);
  param.mutable_state()->set_phase(caffe::TEST);
  EditNetDescriptionFixedPoint(&param, "Convolution_and_InnerProduct",
      "Parameters_and_Output", bw_conv_params_, bw_fc_params_, bw_out_);
  caffe_net = new Net<float>(param, NULL);
  caffe_net->CopyTrainedLayersFrom(weights_);
  RunForwardBatches(iterations_, caffe_net, &accuracy);
  delete caffe_net;
  param.release_state();
  WriteProtoToTextFile(param, model_quantized_);

  // Write summary of fixed point analysis to log
  LOG(INFO) << "------------------------------";
  LOG(INFO) << "Network accuracy analysis for";
  LOG(INFO) << "Convolutional (CONV) and fully";
  LOG(INFO) << "connected (FC) layers.";
  LOG(INFO) << "Baseline 32bit float: " << test_score_baseline_;
  LOG(INFO) << "Fixed point CONV weights: ";
  for (int j = 0; j < test_scores_conv_params.size(); ++j) {
    LOG(INFO) << test_bw_conv_params[j] << "bit: \t" <<
        test_scores_conv_params[j];
  }
  LOG(INFO) << "Fixed point FC weights: ";
  for (int j = 0; j < test_scores_fc_params.size(); ++j) {
    LOG(INFO) << test_bw_fc_params[j] << "bit: \t" << test_scores_fc_params[j];
  }
  LOG(INFO) << "Fixed point layer outputs:";
  for (int j = 0; j < test_scores_layer_out.size(); ++j) {
    LOG(INFO) << test_bw_layer_out[j] << "bit: \t" << test_scores_layer_out[j];
  }
  LOG(INFO) << "Fixed point net:";
  LOG(INFO) << bw_conv_params_ << "bit CONV weights,";
  LOG(INFO) << bw_fc_params_ << "bit FC weights,";
  LOG(INFO) << bw_out_ << "bit layer outputs:";
  LOG(INFO) << "Accuracy: " << accuracy;
  LOG(INFO) << "Please fine-tune.";
}

void Quantization::Quantize2MiniFloatingPoint() {
  // Find the number of bits required for exponent
  vector<float> max_out, max_params;
  baseline_net_->RangeInLayers(&layer_names_, &max_out, &max_params);
  // The exponent bits are chosen such that no saturation occurs.
  // This approximation assumes an infinitely long mantissa.
  // Parameters are ignored, since they are normally smaller than layer outputs
  vector<int> exp_out;
  for ( int i = 0; i < layer_names_.size(); ++i) {
    exp_out.push_back(ceil(log2(log2(max_out[i]) - 1) + 1));
    exp_bits_ = std::max( exp_bits_, exp_out[i]);
  }
  // Debug
  for ( int k = 0; k < layer_names_.size(); ++k) {
    LOG(INFO) << "Layer " << layer_names_[k] <<
        ", exp bits output=" << exp_out[k];
  }
  delete baseline_net_;

  // Score net with floating point parameters and floating point layer outputs
  NetParameter param;
  caffe::ReadNetParamsFromTextFileOrDie(model_, &param);
  param.mutable_state()->set_phase(caffe::TEST);
  vector<int> test_bitwidth;
  vector<float> test_scores;
  float accuracy;
  Net<float>* caffe_net;
  // Test the net with different bit-widths
  for (int bitwidth = 16; bitwidth - 1 - exp_bits_ > 0; bitwidth /= 2) {
    EditNetDescriptionMiniFloat(&param, bitwidth);
    caffe_net = new Net<float>(param, NULL);
    caffe_net->CopyTrainedLayersFrom(weights_);
    RunForwardBatches(iterations_, caffe_net, &accuracy);
    test_bitwidth.push_back(bitwidth);
    test_scores.push_back(accuracy);
    delete caffe_net;
    if ( accuracy * accuracy_drop_threashold_ < test_score_baseline_ ) break;
  }

  // Choose bitwidth for network
  int best_bitwidth = 0;
  for(int i = 0; i < test_scores.size(); ++i) {
    if (test_scores[i] + error_margin_ / 100 >= test_score_baseline_)
      best_bitwidth = test_bitwidth[i];
    else
      break;
  }

  // Write prototxt file of net with best bitwidth
  caffe::ReadNetParamsFromTextFileOrDie(model_, &param);
  EditNetDescriptionMiniFloat(&param, best_bitwidth);
  WriteProtoToTextFile(param, model_quantized_);

  // Write summary of floating point analysis to log
  LOG(INFO) << "------------------------------";
  LOG(INFO) << "Network accuracy analysis for";
  LOG(INFO) << "Convolutional (CONV) and fully";
  LOG(INFO) << "connected (FC) layers.";
  LOG(INFO) << "Baseline 32bit float: " << test_score_baseline_;
  LOG(INFO) << "Mini floating point net:";
  for(int j = 0; j < test_scores.size(); ++j) {
    LOG(INFO) << test_bitwidth[j] << "bit: \t" << test_scores[j];
  }
  LOG(INFO) << "Please fine-tune.";
}

void Quantization::Quantize2PowerOf2Weights(){
  // We don't need to analyze layer outputs and parameter ranges.
  delete baseline_net_;

  // Score net with power-of-two weights
  NetParameter param;
  caffe::ReadNetParamsFromTextFileOrDie(model_, &param);
  param.mutable_state()->set_phase(caffe::TEST);
  float accuracy;
  Net<float>* caffe_net;
  EditNetDescriptionPower2Weights(&param);
  caffe_net = new Net<float>(param, NULL);
  caffe_net->CopyTrainedLayersFrom(weights_);
  RunForwardBatches(iterations_, caffe_net, &accuracy);
  delete caffe_net;

  // Write prototxt file of quantized net
  param.release_state();
  WriteProtoToTextFile(param, model_quantized_);

  // Write summary of floating point analysis to log
  LOG(INFO) << "------------------------------";
  LOG(INFO) << "Network accuracy analysis for";
  LOG(INFO) << "Power-of-two weights for";
  LOG(INFO) << "Convolutional (CONV) and fully";
  LOG(INFO) << "connected (FC) layers.";
  LOG(INFO) << "Baseline 32bit float: " << test_score_baseline_;
  LOG(INFO) << "Quantized net:";
  LOG(INFO) << "4bit: \t" << accuracy;
  LOG(INFO) << "Please fine-tune.";
}

void Quantization::EditNetDescriptionFixedPoint(NetParameter* param,
      const string layers_2_quantize, const string net_part,
      const int bw_conv, const int bw_fc, const int bw_out) {
  for (int i = 0; i < param->layer_size(); ++i) {
    // if this layer should be quantized ...
    if ((layers_2_quantize == "Convolution" ||
        layers_2_quantize == "Convolution_and_InnerProduct") &&
        (param->layer(i).type() == "Convolution" ||
        param->layer(i).type() == "ConvolutionRistretto")) {
      // quantize parameters
      if (net_part == "Parameters" || net_part == "Parameters_and_Output") {
        LayerParameter* param_layer = param->mutable_layer(i);
        param_layer->set_type("ConvolutionRistretto");
        param_layer->mutable_quantization_param()->set_fl_params(bw_conv -
            GetIntegerLengthParams(param->layer(i).name()));
        param_layer->mutable_quantization_param()->set_bw_params(bw_conv);
      }
      // quantize outputs
      if (net_part == "Output" || net_part == "Parameters_and_Output") {
        LayerParameter* param_layer = param->mutable_layer(i);
        param_layer->set_type("ConvolutionRistretto");
        param_layer->mutable_quantization_param()->set_fl_layer_out(bw_out -
            GetIntegerLengthOut(param->layer(i).name()));
        param_layer->mutable_quantization_param()->set_bw_layer_out(bw_out);
      }
    }
    // if this layer should be quantized ...
    if ((layers_2_quantize == "InnerProduct" ||
        layers_2_quantize == "Convolution_and_InnerProduct") &&
        (param->layer(i).type() == "InnerProduct" ||
        param->layer(i).type() == "FcRistretto")) {
      // quantize parameters
      if (net_part == "Parameters" || net_part == "Parameters_and_Output") {
        LayerParameter* param_layer = param->mutable_layer(i);
        param_layer->set_type("FcRistretto");
        param_layer->mutable_quantization_param()->set_fl_params(bw_fc -
            GetIntegerLengthParams(param->layer(i).name()));
        param_layer->mutable_quantization_param()->set_bw_params(bw_fc);
      }
      // quantize outputs
      if (net_part == "Output" || net_part == "Parameters_and_Output") {
        LayerParameter* param_layer = param->mutable_layer(i);
        param_layer->set_type("FcRistretto");
        param_layer->mutable_quantization_param()->set_fl_layer_out(bw_out -
            GetIntegerLengthOut(param->layer(i).name()) );
        param_layer->mutable_quantization_param()->set_bw_layer_out(bw_out);
      }
    }
  }
}

void Quantization::EditNetDescriptionMiniFloat(NetParameter* param,
      const int bitwidth) {
  caffe::QuantizationParameter_Precision precision =
        caffe::QuantizationParameter_Precision_MINI_FLOATING_POINT;
  for (int i = 0; i < param->layer_size(); ++i) {
    if ( param->layer(i).type() == "Convolution" ||
          param->layer(i).type() == "ConvolutionRistretto") {
      LayerParameter* param_layer = param->mutable_layer(i);
      param_layer->set_type("ConvolutionRistretto");
      param_layer->mutable_quantization_param()->set_precision(precision);
      param_layer->mutable_quantization_param()->set_mant_bits(bitwidth
          - exp_bits_ - 1);
      param_layer->mutable_quantization_param()->set_exp_bits(exp_bits_);
    } else if ( param->layer(i).type() == "InnerProduct" ||
          param->layer(i).type() == "FcRistretto") {
      LayerParameter* param_layer = param->mutable_layer(i);
      param_layer->set_type("FcRistretto");
      param_layer->mutable_quantization_param()->set_precision(precision);
      param_layer->mutable_quantization_param()->set_mant_bits(bitwidth
          - exp_bits_ - 1);
      param_layer->mutable_quantization_param()->set_exp_bits(exp_bits_);
    }
  }
}

void Quantization::EditNetDescriptionPower2Weights(NetParameter* param) {
  caffe::QuantizationParameter_Precision precision =
      caffe::QuantizationParameter_Precision_POWER_2_WEIGHTS;
  for (int i = 0; i < param->layer_size(); ++i) {
    if ( param->layer(i).type() == "Convolution" ||
          param->layer(i).type() == "ConvolutionRistretto") {
      LayerParameter* param_layer = param->mutable_layer(i);
      param_layer->set_type("ConvolutionRistretto");
      param_layer->mutable_quantization_param()->set_precision(precision);
      param_layer->mutable_quantization_param()->set_exp_min(-8);
      param_layer->mutable_quantization_param()->set_exp_max(-1);
    } else if ( param->layer(i).type() == "InnerProduct" ||
          param->layer(i).type() == "FcRistretto") {
      LayerParameter* param_layer = param->mutable_layer(i);
      param_layer->set_type("FcRistretto");
      param_layer->mutable_quantization_param()->set_precision(precision);
      param_layer->mutable_quantization_param()->set_exp_min(-8);
      param_layer->mutable_quantization_param()->set_exp_max(-1);
    }
  }
}

int Quantization::GetIntegerLengthParams(const string layer_name) {
  int pos = find(layer_names_.begin(), layer_names_.end(), layer_name)
      - layer_names_.begin();
  return il_params_[pos];
}

int Quantization::GetIntegerLengthOut(const string layer_name) {
  int pos = find(layer_names_.begin(), layer_names_.end(), layer_name)
      - layer_names_.begin();
  return il_out_[pos];
}
