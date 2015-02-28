#ifndef CAFFE_PYTHON_LAYER_HPP_
#define CAFFE_PYTHON_LAYER_HPP_

#include <boost/python.hpp>

#include <string>
#include <vector>

#include "caffe/layer.hpp"

namespace bp = boost::python;

namespace caffe {

#define PYTHON_LAYER_ERROR() { \
  PyObject *petype, *pevalue, *petrace; \
  PyErr_Fetch(&petype, &pevalue, &petrace); \
  bp::object etype(bp::handle<>(bp::borrowed(petype))); \
  bp::object evalue(bp::handle<>(bp::borrowed(bp::allow_null(pevalue)))); \
  bp::object etrace(bp::handle<>(bp::borrowed(bp::allow_null(petrace)))); \
  bp::object sio(bp::import("StringIO").attr("StringIO")()); \
  bp::import("traceback").attr("print_exception")( \
    etype, evalue, etrace, bp::object(), sio); \
  LOG(INFO) << bp::extract<string>(sio.attr("getvalue")())(); \
  PyErr_Restore(petype, pevalue, petrace); \
  throw; \
}

template <typename Dtype>
class PythonLayer : public Layer<Dtype> {
 public:
  PythonLayer(PyObject* self, const LayerParameter& param)
      : Layer<Dtype>(param), self_(bp::handle<>(bp::borrowed(self))) { }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    try {
      self_.attr("param_str_") = bp::str(
        this->layer_param_.python_param().param_str());
      self_.attr("setup")(bottom, top);
    } catch (bp::error_already_set) {
      PYTHON_LAYER_ERROR();
    }
  }

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    try {
      self_.attr("reshape")(bottom, top);
    } catch (bp::error_already_set) {
      PYTHON_LAYER_ERROR();
    }
  }

  virtual inline const char* type() const { return "Python"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    try {
      self_.attr("forward")(bottom, top);
    } catch (bp::error_already_set) {
      PYTHON_LAYER_ERROR();
    }
  }
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    try {
      self_.attr("backward")(top, propagate_down, bottom);
    } catch (bp::error_already_set) {
      PYTHON_LAYER_ERROR();
    }
  }

 private:
  bp::object self_;
};

}  // namespace caffe

#endif
