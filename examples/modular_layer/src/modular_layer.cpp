#if defined(COMPILE_MODULAR_LAYER)

#include <caffe/caffe.hpp>

#include <vector>

/*
 * Example of how to use this layer in prototxt files
 * layer {
 *   name: "layer_module"
 *   type: "Module"
 *   bottom: "some_bottom"
 *   top: "some_top"
 *   module_param {
 *     library: "modular_layer"
 *     type: "ExampleModular"
 *     param_str: "5"
 *     #creator_symbol: "CreateExampleModularLayer" # Only when overriding
 *     #deleter_symbol: "DeleteExampleModularlayer" # Only when overriding
 *   }
 * }
*/

namespace caffe {

template <typename Dtype>
class ExampleModularLayer: public Layer<Dtype> {
 public:
  explicit ExampleModularLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}

  void LayerSetUp(const std::vector<Blob<Dtype>*>&,
    const std::vector<Blob<Dtype>*>&) {
    // parse params from param_str
    // this param is a string type, but can represent
    // int, float, json, yaml... whatever the module implementation requires
    // (in this case it is parsed as an int)
    some_param = std::stoi(this->layer_param_.module_param().param_str());
  }

  virtual inline const char* type() const { return "ExampleModular"; }

 protected:
  virtual void Reshape(const vector<Blob<Dtype>*>&,
      const vector<Blob<Dtype>*>&) {}

  virtual void Forward_cpu(const vector<Blob<Dtype>*>&,
      const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(1, 1, 1, some_param);
  }

  virtual void Backward_cpu(const vector<Blob<Dtype>*>&,
      const vector<bool>&,
      const vector<Blob<Dtype>*>&) {}

  int some_param;
};

EXPORT_LAYER_MODULE_CLASS(ExampleModular);

// the above macro can be expanded to the code below

/*
template <typename Dtype>
Layer<Dtype> * CreateExampleModularLayer(const LayerParameter& param) {
  return new ExampleModularLayer<Dtype>(param);
}
EXPORT_LAYER_MODULE_CREATOR(ExampleModular, CreateExampleModularLayer);
*/

// for which the last macro can be expanded as follows:

/*
extern "C" Layer<float> * CreateExampleModularLayer_float(
  const LayerParameter& param
) {
  return new ExampleModularLayer<float>(param);
}

extern "C" Layer<double> * CreateExampleModularLayer_double(
  const LayerParameter& param
) {
  return new ExampleModularLayer<double>(param);
}
*/

// This function is not necessary, by default Caffe
// will already call 'delete layer;', but this can be
// used to override deletion behaviour.
/*
template <typename Dtype>
void DeleteExampleModularLayer(Layer<Dtype> * layer) {
  delete layer;
}
EXPORT_LAYER_MODULE_DELETER(DeleteExampleModularLayer);
*/

}  // namespace caffe

#else

// Caffe insists on building this file in their build system
// so we give it a main to build
int main() {
  return 0;
}

#endif
