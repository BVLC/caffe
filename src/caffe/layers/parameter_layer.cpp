#include "caffe/layers/parameter_layer.hpp"

namespace caffe {

INSTANTIATE_CLASS_3T(ParameterLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T(ParameterLayer, (double), (double), (double));

REGISTER_LAYER_CLASS(Parameter);
REGISTER_LAYER_CLASS_INST(Parameter, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(Parameter, (double), (double), (double));


}  // namespace caffe
