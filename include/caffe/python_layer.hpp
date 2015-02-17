#ifndef CAFFE_PYTHON_LAYER_HPP_
#define CAFFE_PYTHON_LAYER_HPP_

#include <boost/python.hpp>
#include <vector>

#include "caffe/layer.hpp"

namespace bp = boost::python;

namespace caffe {

template <typename Dtype>
class PythonLayer : public Layer<Dtype> {
 public:
  PythonLayer(PyObject* self, const LayerParameter& param)
      : Layer<Dtype>(param), self_(self) { }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    try {
      bp::call_method<bp::object>(self_, "setup", bottom, top);
    } catch (bp::error_already_set) {
      PyErr_Print();
      throw;
    }
  }

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    try {
      bp::call_method<bp::object>(self_, "reshape", bottom, top);
    } catch (bp::error_already_set) {
      PyErr_Print();
      throw;
    }
  }

  virtual inline const char* type() const { return "Python"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    try {
      bp::call_method<bp::object>(self_, "forward", bottom, top);
    } catch (bp::error_already_set) {
      PyErr_Print();
      throw;
    }
  }
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    try {
      bp::call_method<bp::object>(self_, "backward", top, propagate_down,
          bottom);
    } catch (bp::error_already_set) {
      PyErr_Print();
      throw;
    }
  }

 private:
  PyObject* self_;
};

}  // namespace caffe

#endif
