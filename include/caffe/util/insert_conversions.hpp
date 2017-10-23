#ifndef CAFFE_UTIL_INSERT_CONVERSIONS_HPP_
#define CAFFE_UTIL_INSERT_CONVERSIONS_HPP_

#include <string>

#include "caffe/proto/caffe.pb.h"

namespace caffe {

void InsertConversions(const NetParameter& param, NetParameter* param_convert);

void ConfigureConvertLayer(const string& layer_name, const string& blob_name,
    const int_tp blob_idx, const int_tp split_count, const float loss_weight,
    LayerParameter* split_layer_param);

string ConvertLayerName(const string& layer_name, const string& blob_name,
    const int_tp blob_idx);

string ConvertBlobName(const string& layer_name, const string& blob_name,
    const int_tp blob_idx, const int_tp split_idx);

}  // namespace caffe


#endif  // CAFFE_UTIL_INSERT_CONVERSIONS_HPP_
