#include <vector>

#include "caffe/layers/input_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void InputLayer<Dtype, MItype, MOtype>::LayerSetUp(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top) {
  const int num_top = top.size();
  const InputParameter& param = this->layer_param_.input_param();
  const int num_shape = param.shape_size();
  CHECK(num_shape == 0 || num_shape == 1 || num_shape == num_top)
      << "Must specify 'shape' once, once per top blob, or not at all: "
      << num_top << " tops vs. " << num_shape << " shapes.";
  if (num_shape > 0) {
    for (int i = 0; i < num_top; ++i) {
      const int shape_index = (param.shape_size() == 1) ? 0 : i;
      top[i]->Reshape(param.shape(shape_index));
    }
  }
}

INSTANTIATE_CLASS_3T_GUARDED(InputLayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(InputLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(InputLayer, (double), (double), (double));

REGISTER_LAYER_CLASS(Input);
REGISTER_LAYER_CLASS_INST(Input, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(Input, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(Input, (double), (double), (double));

}  // namespace caffe
