#ifndef CAFFE_UTIL_INSERT_CONVERSIONS_HPP_
#define CAFFE_UTIL_INSERT_CONVERSIONS_HPP_

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

void InsertConversions(const NetParameter& param, NetParameter* param_convert);

void ConfigureConversionLayer(const string& layer_name, const string& blob_name,
    const int_tp blob_idx, const float loss_weight,
    LayerParameter* convert_layer_param, DataType bottom_data_type,
    DataType top_data_type, const QuantizerParameter* ref_quant_param);

string ConversionLayerName(const string& layer_name, const string& blob_name,
    const int_tp blob_idx);

string ConversionBlobName(const string& layer_name, const string& blob_name,
    const int_tp blob_idx);

}  // namespace caffe


#endif  // CAFFE_UTIL_INSERT_CONVERSIONS_HPP_
