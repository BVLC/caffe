#include "caffe/util/insert_conversions.hpp"

namespace caffe {

void InsertConversions(const NetParameter& param, NetParameter* param_convert) {
  // Initialize by copying from the input NetParameter.
  param_convert->CopyFrom(param);
  param_convert->clear_layer();
  map<string, DataType> blob_data_types;
  map<string, string> blob_name_to_layer_name;
  for (int_tp i = 0; i < param.layer_size(); ++i) {
    const LayerParameter& layer_param = param.layer(i);
    vector<string> layer_bottom_names(layer_param.bottom_size());
    for (int_tp j = 0; j < layer_param.bottom_size(); ++j) {
      const string& blob_name = layer_param.bottom(j);
      string convert_blob_name = blob_name;
      if (blob_data_types.find(blob_name) ==
          blob_data_types.end()) {
        LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
                   << layer_param.name() << "', bottom index " << j << ")";
      } else {
        if (layer_param.bottom_data_type() != blob_data_types[blob_name]) {
          convert_blob_name = ConversionBlobName(
              blob_name_to_layer_name[blob_name], blob_name, j);
          LayerParameter* convert_layer_param = param_convert->add_layer();
          float loss_weight = 0.0;
          if (j < std::min(layer_param.loss_weight_size(),
                           layer_param.top_size())) {
            loss_weight = layer_param.loss_weight(j);
          }
          const QuantizerParameter* ref_quant_param = nullptr;
          if (layer_param.bottom_quantizer_size() > j) {
            ref_quant_param = &(layer_param.bottom_quantizer(j));
          }
          ConfigureConversionLayer(blob_name_to_layer_name[blob_name],
                                   blob_name, j, loss_weight,
                                   convert_layer_param,
                                   blob_data_types[blob_name],
                                   layer_param.bottom_data_type(),
                                   ref_quant_param);
        }
      }
      layer_bottom_names[j] = convert_blob_name;
    }
    LayerParameter* copy_layer_param = param_convert->add_layer();
    copy_layer_param->CopyFrom(param.layer(i));
    for (int_tp j = 0; j < layer_bottom_names.size(); ++j) {
      while (copy_layer_param->bottom_quantizer_size() <= j) {
        copy_layer_param->add_bottom_quantizer();
      }
      // Need to preserve the original blob name for quantization purposes
      if (!copy_layer_param->bottom_quantizer(j).has_name()) {
        copy_layer_param->mutable_bottom_quantizer(j)->set_name(
            copy_layer_param->bottom(j));
      }
      copy_layer_param->set_bottom(j, layer_bottom_names[j]);
    }
    for (int_tp j = 0; j < layer_param.top_size(); ++j) {
      const string& blob_name = layer_param.top(j);
      blob_data_types[blob_name] = layer_param.top_data_type();
      blob_name_to_layer_name[blob_name] = layer_param.name();
    }
  }
}

void ConfigureConversionLayer(const string& layer_name, const string& blob_name,
    const int_tp blob_idx, const float loss_weight,
    LayerParameter* convert_layer_param, DataType bottom_data_type,
    DataType top_data_type, const QuantizerParameter* ref_quant_param) {
  QuantizerParameter quant_param;
  if (ref_quant_param) {
    quant_param.CopyFrom(*ref_quant_param);
  }
  quant_param.set_name(blob_name);

  convert_layer_param->Clear();
  QuantizerParameter* bottom_quant_param = convert_layer_param->
      add_bottom_quantizer();
  bottom_quant_param->CopyFrom(quant_param);
  QuantizerParameter* top_quant_param = convert_layer_param->
      add_top_quantizer();
  top_quant_param->CopyFrom(quant_param);
  convert_layer_param->add_bottom(blob_name);
  convert_layer_param->set_name(ConversionLayerName(layer_name, blob_name,
                                                    blob_idx));
  convert_layer_param->set_type("Quantizer");
  convert_layer_param->set_compute_data_type(bottom_data_type);
  convert_layer_param->set_top_data_type(top_data_type);
  convert_layer_param->set_bottom_data_type(bottom_data_type);
  convert_layer_param->add_top(ConversionBlobName(layer_name, blob_name,
                                                  blob_idx));
  convert_layer_param->add_loss_weight(loss_weight);
}

string ConversionLayerName(const string& layer_name, const string& blob_name,
    const int_tp blob_idx) {
  ostringstream convert_layer_name;
  convert_layer_name << blob_name << "_" << layer_name << "_" << blob_idx
      << "_converted";
  return convert_layer_name.str();
}

string ConversionBlobName(const string& layer_name, const string& blob_name,
    const int_tp blob_idx) {
  ostringstream convert_blob_name;
  convert_blob_name << blob_name << "_" << layer_name << "_" << blob_idx
      << "_converted";
  return convert_blob_name.str();
}

}
