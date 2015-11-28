#ifdef WITH_PYTHON_LAYER

#include "caffe/python_layer.hpp"

#include <boost/thread.hpp>
#include <vector>

namespace caffe {

// A class to initialize Python environment, called by init_python_environment()
class PyInitializer {
 private:
  PyInitializer() {
    Py_Initialize();
    PyEval_InitThreads();
    PyEval_ReleaseLock();
  }
  friend void init_python_environment();
};

void init_python_environment() {
  static PyInitializer py_initializer;
}

class AcquirePyGIL {
 public:
  AcquirePyGIL() {
    state = PyGILState_Ensure();
  }
  ~AcquirePyGIL() {
    PyGILState_Release(state);
  }
 private:
  PyGILState_STATE state;
};

template <typename Dtype>
void PythonLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  AcquirePyGIL lock;
  self_.attr("param_str") = bp::str(
      this->layer_param_.python_param().param_str());
  self_.attr("setup")(bottom, top);
}

template <typename Dtype>
void PythonLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  AcquirePyGIL lock;
  self_.attr("reshape")(bottom, top);
}

template <typename Dtype>
void PythonLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  AcquirePyGIL lock;
  self_.attr("forward")(bottom, top);
}

template <typename Dtype>
void PythonLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  AcquirePyGIL lock;
  self_.attr("backward")(top, propagate_down, bottom);
}

INSTANTIATE_CLASS(PythonLayer);

}  // namespace caffe

#endif  // #ifdef WITH_PYTHON_LAYER
