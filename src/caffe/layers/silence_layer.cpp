#include <vector>

#include "caffe/layers/silence_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

#ifdef CPU_ONLY
STUB_GPU(SilenceLayer);
#endif

INSTANTIATE_CLASS(SilenceLayer);
REGISTER_LAYER_CLASS(Silence);

}  // namespace caffe
