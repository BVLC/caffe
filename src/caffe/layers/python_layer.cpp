#ifdef USE_PYTHON_LAYER
#include <boost/python.hpp>
#include <Python.h>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/python_layer.hpp"

namespace bp = boost::python;

namespace caffe {

template <typename Dtype>
vector<PyBlob<Dtype> > PythonLayer<Dtype>::PythonBlobVector(
    const vector<Blob<Dtype>*>& vec) {
  return vector<PyBlob<Dtype> >(vec.begin(), vec.end());
}

template <typename Dtype>
void PythonLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
  Py_Initialize();
  init_caffe();

  try {
    bp::object module_ = bp::import(
        this->layer_param_.python_param().module().c_str());
    layer_ = module_.attr(this->layer_param_.python_param().layer().c_str())();

    layer_.attr("setup")(PythonBlobVector(bottom), PythonBlobVector(top));
  } catch (bp::error_already_set) {
    PyErr_Print();
    throw;
  }
}

template <typename Dtype>
void PythonLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  try {
    layer_.attr("reshape")(PythonBlobVector(bottom), PythonBlobVector(top));
  } catch (bp::error_already_set) {
    PyErr_Print();
    throw;
  }
}

template <typename Dtype>
void PythonLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  try {
    layer_.attr("forward")(PythonBlobVector(bottom), PythonBlobVector(top));
  } catch (bp::error_already_set) {
    PyErr_Print();
    throw;
  }
}

template <typename Dtype>
void PythonLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  try {
    layer_.attr("backward")(PythonBlobVector(top), propagate_down,
        PythonBlobVector(bottom));
  } catch (bp::error_already_set) {
    PyErr_Print();
    throw;
  }
}

INSTANTIATE_CLASS(PythonLayer);

}  // namespace caffe
#endif
