// Copyright Yangqing Jia 2013
// pycaffe provides a wrapper of the caffe::Net class as well as some
// caffe::Caffe functions so that one could easily call it from Python.
// Note that for python, we will simply use float as the data type.

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <numpy/arrayobject.h>
#include "caffe/caffe.hpp"

// Temporary solution for numpy < 1.7 versions: old macro.
#ifndef NPY_ARRAY_C_CONTIGUOUS
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#endif


using namespace caffe;
using boost::python::extract;
using boost::python::len;
using boost::python::list;
using boost::python::object;
using boost::python::handle;
using boost::python::vector_indexing_suite;


// wrap shared_ptr<Blob<float> > in a class that we construct in C++ and pass
//  to Python
class CaffeBlob {
 public:

  CaffeBlob(const shared_ptr<Blob<float> > &blob)
      : blob_(blob) {}

  CaffeBlob()
  {}

  int num() const { return blob_->num(); }
  int channels() const { return blob_->channels(); }
  int height() const { return blob_->height(); }
  int width() const { return blob_->width(); }
  int count() const { return blob_->count(); }

  bool operator == (const CaffeBlob &other)
  {
      return this->blob_ == other.blob_;
  }

 protected:
  shared_ptr<Blob<float> > blob_;
};


// we need another wrapper (used as boost::python's HeldType) that receives a
//  self PyObject * which we can use as ndarray.base, so that data/diff memory
//  is not freed while still being used in Python
class CaffeBlobWrap : public CaffeBlob {
 public:
  CaffeBlobWrap(PyObject *p, shared_ptr<Blob<float> > &blob)
      : CaffeBlob(blob), self_(p) {}

  CaffeBlobWrap(PyObject *p, const CaffeBlob &blob)
      : CaffeBlob(blob), self_(p) {}

  object get_data()
  {
      npy_intp dims[] = {num(), channels(), height(), width()};

      PyObject *obj = PyArray_SimpleNewFromData(4, dims, NPY_FLOAT32,
                                                blob_->mutable_cpu_data());
      PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(obj), self_);
      Py_INCREF(self_);
      handle<> h(obj);

      return object(h);
  }

  object get_diff()
  {
      npy_intp dims[] = {num(), channels(), height(), width()};

      PyObject *obj = PyArray_SimpleNewFromData(4, dims, NPY_FLOAT32,
                                                blob_->mutable_cpu_diff());
      PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(obj), self_);
      Py_INCREF(self_);
      handle<> h(obj);

      return object(h);
  }

 private:
  PyObject *self_;
};



// A simple wrapper over CaffeNet that runs the forward process.
struct CaffeNet
{
  CaffeNet(string param_file, string pretrained_param_file) {
    net_.reset(new Net<float>(param_file));
    net_->CopyTrainedLayersFrom(pretrained_param_file);
  }

  virtual ~CaffeNet() {}

  inline void check_array_against_blob(
      PyArrayObject* arr, Blob<float>* blob) {
    CHECK(PyArray_FLAGS(arr) & NPY_ARRAY_C_CONTIGUOUS);
    CHECK_EQ(PyArray_NDIM(arr), 4);
    CHECK_EQ(PyArray_ITEMSIZE(arr), 4);
    npy_intp* dims = PyArray_DIMS(arr);
    CHECK_EQ(dims[0], blob->num());
    CHECK_EQ(dims[1], blob->channels());
    CHECK_EQ(dims[2], blob->height());
    CHECK_EQ(dims[3], blob->width());
  }

  // The actual forward function. It takes in a python list of numpy arrays as
  // input and a python list of numpy arrays as output. The input and output
  // should all have correct shapes, are single-precisionabcdnt- and c contiguous.
  void Forward(list bottom, list top) {
    vector<Blob<float>*>& input_blobs = net_->input_blobs();
    CHECK_EQ(len(bottom), input_blobs.size());
    CHECK_EQ(len(top), net_->num_outputs());
    // First, copy the input
    for (int i = 0; i < input_blobs.size(); ++i) {
      object elem = bottom[i];
      PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(elem.ptr());
      check_array_against_blob(arr, input_blobs[i]);
      switch (Caffe::mode()) {
      case Caffe::CPU:
        memcpy(input_blobs[i]->mutable_cpu_data(), PyArray_DATA(arr),
            sizeof(float) * input_blobs[i]->count());
        break;
      case Caffe::GPU:
        cudaMemcpy(input_blobs[i]->mutable_gpu_data(), PyArray_DATA(arr),
            sizeof(float) * input_blobs[i]->count(), cudaMemcpyHostToDevice);
        break;
      default:
        LOG(FATAL) << "Unknown Caffe mode.";
      }  // switch (Caffe::mode())
    }
    //LOG(INFO) << "Start";
    const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled();
    //LOG(INFO) << "End";
    for (int i = 0; i < output_blobs.size(); ++i) {
      object elem = top[i];
      PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(elem.ptr());
      check_array_against_blob(arr, output_blobs[i]);
      switch (Caffe::mode()) {
      case Caffe::CPU:
        memcpy(PyArray_DATA(arr), output_blobs[i]->cpu_data(),
            sizeof(float) * output_blobs[i]->count());
        break;
      case Caffe::GPU:
        cudaMemcpy(PyArray_DATA(arr), output_blobs[i]->gpu_data(),
            sizeof(float) * output_blobs[i]->count(), cudaMemcpyDeviceToHost);
        break;
      default:
        LOG(FATAL) << "Unknown Caffe mode.";
      }  // switch (Caffe::mode())
    }
  }

  void Backward(list top_diff, list bottom_diff) {
    vector<Blob<float>*>& output_blobs = net_->output_blobs();
    vector<Blob<float>*>& input_blobs = net_->input_blobs();
    CHECK_EQ(len(bottom_diff), input_blobs.size());
    CHECK_EQ(len(top_diff), output_blobs.size());
    // First, copy the output diff
    for (int i = 0; i < output_blobs.size(); ++i) {
      object elem = top_diff[i];
      PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(elem.ptr());
      check_array_against_blob(arr, output_blobs[i]);
      switch (Caffe::mode()) {
      case Caffe::CPU:
        memcpy(output_blobs[i]->mutable_cpu_diff(), PyArray_DATA(arr),
            sizeof(float) * output_blobs[i]->count());
        break;
      case Caffe::GPU:
        cudaMemcpy(output_blobs[i]->mutable_gpu_diff(), PyArray_DATA(arr),
            sizeof(float) * output_blobs[i]->count(), cudaMemcpyHostToDevice);
        break;
      default:
        LOG(FATAL) << "Unknown Caffe mode.";
      }  // switch (Caffe::mode())
    }
    //LOG(INFO) << "Start";
    net_->Backward();
    //LOG(INFO) << "End";
    for (int i = 0; i < input_blobs.size(); ++i) {
      object elem = bottom_diff[i];
      PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(elem.ptr());
      check_array_against_blob(arr, input_blobs[i]);
      switch (Caffe::mode()) {
      case Caffe::CPU:
        memcpy(PyArray_DATA(arr), input_blobs[i]->cpu_diff(),
            sizeof(float) * input_blobs[i]->count());
        break;
      case Caffe::GPU:
        cudaMemcpy(PyArray_DATA(arr), input_blobs[i]->gpu_diff(),
            sizeof(float) * input_blobs[i]->count(), cudaMemcpyDeviceToHost);
        break;
      default:
        LOG(FATAL) << "Unknown Caffe mode.";
      }  // switch (Caffe::mode())
    }
  }

  // The caffe::Caffe utility functions.
  void set_mode_cpu() { Caffe::set_mode(Caffe::CPU); }
  void set_mode_gpu() { Caffe::set_mode(Caffe::GPU); }
  void set_phase_train() { Caffe::set_phase(Caffe::TRAIN); }
  void set_phase_test() { Caffe::set_phase(Caffe::TEST); }
  void set_device(int device_id) { Caffe::SetDevice(device_id); }

  vector<CaffeBlob> blobs() {
      return vector<CaffeBlob>(net_->blobs().begin(), net_->blobs().end());
  }

  vector<CaffeBlob> params() {
      return vector<CaffeBlob>(net_->params().begin(), net_->params().end());
  }

  // The pointer to the internal caffe::Net instant.
	shared_ptr<Net<float> > net_;
};



// The boost python module definition.
BOOST_PYTHON_MODULE(pycaffe)
{

  boost::python::class_<CaffeNet>(
      "CaffeNet", boost::python::init<string, string>())
      .def("Forward",         &CaffeNet::Forward)
      .def("Backward",        &CaffeNet::Backward)
      .def("set_mode_cpu",    &CaffeNet::set_mode_cpu)
      .def("set_mode_gpu",    &CaffeNet::set_mode_gpu)
      .def("set_phase_train", &CaffeNet::set_phase_train)
      .def("set_phase_test",  &CaffeNet::set_phase_test)
      .def("set_device",      &CaffeNet::set_device)
      .def("blobs",           &CaffeNet::blobs)
      .def("params",          &CaffeNet::params)
  ;

  boost::python::class_<CaffeBlob, CaffeBlobWrap>(
      "CaffeBlob", boost::python::no_init)
      .add_property("num",      &CaffeBlob::num)
      .add_property("channels", &CaffeBlob::channels)
      .add_property("height",   &CaffeBlob::height)
      .add_property("width",    &CaffeBlob::width)
      .add_property("count",    &CaffeBlob::count)
      .add_property("data",     &CaffeBlobWrap::get_data)
      .add_property("diff",     &CaffeBlobWrap::get_diff)
  ;

  boost::python::class_<vector<CaffeBlob> >("BlobVec")
      .def(vector_indexing_suite<vector<CaffeBlob>, true>());

  import_array();

}
