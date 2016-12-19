#ifdef CAFFE_MSL

#ifndef _CAFFE_UTIL_INSERT_BIAS_LAYER_HPP_
#define _CAFFE_UTIL_INSERT_BIAS_LAYER_HPP_

#include <string>

#include "caffe/proto/caffe.pb.h"

namespace caffe {

void SeparateBias(const NetParameter& param, NetParameter* param_split); 

}  // namespace caffe

#endif  // CAFFE_UTIL_INSERT_BIAS_LAYER_HPP_

#endif /* CAFFE_MSL */