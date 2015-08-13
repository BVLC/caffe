#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


namespace caffe {

template <typename Dtype>
void PyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PyParameter py_param = this->layer_param_.py_param();
  ASSERT(py_param.param_fillers_size() == 0 ||
      py_param.param_fillers_size() == py_param.param_shapes_size(),
      "Mismatched number of param fillers (" << py_param.param_fillers_size()
      << ") and param shapes (" << py_param.param_shapes_size() << ")");

  this->blobs_.resize(py_param.param_shapes_size());
  for (int i = 0; i < py_param.param_shapes_size(); ++i) {
    vector<int> shape;
    for (int j = 0; j < py_param.param_shapes(i).dimension_size(); ++j) {
      shape.push_back(py_param.param_shapes(i).dimension(j));
    }
    this->blobs_[i].reset(new Blob<Dtype>(shape));
  }
  for (int i = 0; i < py_param.param_fillers_size(); ++i) {
    shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(
        py_param.param_fillers(i)));
    filler->Fill(this->blobs_[i].get());
  }
}

INSTANTIATE_CLASS(PyLayer);
REGISTER_LAYER_CLASS(Py);

}  // namespace caffe
