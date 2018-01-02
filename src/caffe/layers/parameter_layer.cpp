#include "caffe/layers/parameter_layer.hpp"

namespace caffe {

INSTANTIATE_CLASS_3T_GUARDED(ParameterLayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(ParameterLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(ParameterLayer, (double), (double), (double));

REGISTER_LAYER_CLASS(Parameter);
REGISTER_LAYER_CLASS_INST(Parameter, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(Parameter, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(Parameter, (double), (double), (double));


}  // namespace caffe
