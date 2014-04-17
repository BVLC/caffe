// Copyright 2014 BVLC and contributors.
// pycaffe provides a wrapper of the caffe::Net class as well as some
// caffe::Caffe functions so that one could easily call it from Python.
// Note that for python, we will simply use float as the data type.

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "boost/python.hpp"
#include "boost/python/suite/indexing/vector_indexing_suite.hpp"
#include "numpy/arrayobject.h"

// these need to be included after boost on OS X
#include <string>  // NOLINT(build/include_order)
#include <vector>  // NOLINT(build/include_order)
#include <fstream>  // NOLINT

#include "caffe/caffe.hpp"

// Temporary solution for numpy < 1.7 versions: old macro, no promises.
// You're strongly advised to upgrade to >= 1.7.
#ifndef NPY_ARRAY_C_CONTIGUOUS
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#define PyArray_SetBaseObject(arr, x) (PyArray_BASE(arr) = (x))
#endif


using namespace caffe;  // NOLINT(build/namespaces)
using boost::python::extract;
using boost::python::len;
using boost::python::list;
using boost::python::object;
using boost::python::handle;
using boost::python::vector_indexing_suite;

// for convenience, check that input files can be opened, and raise an
// exception that boost will send to Python if not (caffe could still crash
// later if the input files are disturbed before they are actually used, but
// this saves frustration in most cases)
static void CheckFile(const string& filename) {
    std::ifstream f(filename.c_str());
    if (!f.good()) {
      f.close();
      throw std::runtime_error("Could not open file " + filename);
    }
    f.close();
}

// wrap shared_ptr<Blob<float> > in a class that we construct in C++ and pass
//  to Python
class CaffeBlob {
 public:
  CaffeBlob(const shared_ptr<Blob<float> > &blob, const string& name)
      : blob_(blob), name_(name) {}

  string name() const { return name_; }
  int num() const { return blob_->num(); }
  int channels() const { return blob_->channels(); }
  int height() const { return blob_->height(); }
  int width() const { return blob_->width(); }
  int count() const { return blob_->count(); }

  // this is here only to satisfy boost's vector_indexing_suite
  bool operator == (const CaffeBlob &other) {
      return this->blob_ == other.blob_;
  }

 protected:
  shared_ptr<Blob<float> > blob_;
  string name_;
};


// we need another wrapper (used as boost::python's HeldType) that receives a
//  self PyObject * which we can use as ndarray.base, so that data/diff memory
//  is not freed while still being used in Python
class CaffeBlobWrap : public CaffeBlob {
 public:
  CaffeBlobWrap(PyObject *p, const CaffeBlob &blob)
      : CaffeBlob(blob), self_(p) {}

  object get_data() {
      npy_intp dims[] = {num(), channels(), height(), width()};

      PyObject *obj = PyArray_SimpleNewFromData(4, dims, NPY_FLOAT32,
                                                blob_->mutable_cpu_data());
      PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(obj), self_);
      Py_INCREF(self_);
      handle<> h(obj);

      return object(h);
  }

  object get_diff() {
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


class CaffeLayer {
 public:
  CaffeLayer(const shared_ptr<Layer<float> > &layer, const string &name)
    : layer_(layer), name_(name) {}

  string name() const { return name_; }
  vector<CaffeBlob> blobs() {
    vector<CaffeBlob> result;
    for (int i = 0; i < layer_->blobs().size(); ++i) {
      result.push_back(CaffeBlob(layer_->blobs()[i], name_));
    }
    return result;
  }

  // this is here only to satisfy boost's vector_indexing_suite
  bool operator == (const CaffeLayer &other) {
      return this->layer_ == other.layer_;
  }

 protected:
  shared_ptr<Layer<float> > layer_;
  string name_;
};


// A simple wrapper over CaffeNet that runs the forward process.
struct CaffeNet {
  // For cases where parameters will be determined later by the Python user,
  // create a Net with unallocated parameters (which will not be zero-filled
  // when accessed).
  explicit CaffeNet(string param_file) {
    Init(param_file);
  }

  CaffeNet(string param_file, string pretrained_param_file) {
    Init(param_file);
    CheckFile(pretrained_param_file);
    net_->CopyTrainedLayersFrom(pretrained_param_file);
  }

  explicit CaffeNet(shared_ptr<Net<float> > net)
      : net_(net) {}

  void Init(string param_file) {
    CheckFile(param_file);
    net_.reset(new Net<float>(param_file));
  }


  virtual ~CaffeNet() {}

  // this function is mostly redundant with the one below, but should go away
  // with new pycaffe
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

  // generate Python exceptions for badly shaped or discontiguous arrays
  inline void check_contiguous_array(PyArrayObject* arr, string name,
      int channels, int height, int width) {
    if (!(PyArray_FLAGS(arr) & NPY_ARRAY_C_CONTIGUOUS)) {
      throw std::runtime_error(name + " must be C contiguous");
    }
    if (PyArray_NDIM(arr) != 4) {
      throw std::runtime_error(name + " must be 4-d");
    }
    if (PyArray_TYPE(arr) != NPY_FLOAT32) {
      throw std::runtime_error(name + " must be float32");
    }
    if (PyArray_DIMS(arr)[1] != channels) {
      throw std::runtime_error(name + " has wrong number of channels");
    }
    if (PyArray_DIMS(arr)[2] != height) {
      throw std::runtime_error(name + " has wrong height");
    }
    if (PyArray_DIMS(arr)[3] != width) {
      throw std::runtime_error(name + " has wrong width");
    }
  }

  // The actual forward function. It takes in a python list of numpy arrays as
  // input and a python list of numpy arrays as output. The input and output
  // should all have correct shapes, are single-precisionabcdnt- and
  // c contiguous.
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
    // LOG(INFO) << "Start";
    const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled();
    // LOG(INFO) << "End";
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
    // LOG(INFO) << "Start";
    net_->Backward();
    // LOG(INFO) << "End";
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

  void ForwardPrefilled() {
    net_->ForwardPrefilled();
  }

  void set_input_arrays(object data_obj, object labels_obj) {
    // check that this network has an input MemoryDataLayer
    shared_ptr<MemoryDataLayer<float> > md_layer =
      boost::dynamic_pointer_cast<MemoryDataLayer<float> >(net_->layers()[0]);
    if (!md_layer) {
      throw std::runtime_error("set_input_arrays may only be called if the"
          " first layer is a MemoryDataLayer");
    }

    // check that we were passed appropriately-sized contiguous memory
    PyArrayObject* data_arr =
        reinterpret_cast<PyArrayObject*>(data_obj.ptr());
    PyArrayObject* labels_arr =
        reinterpret_cast<PyArrayObject*>(labels_obj.ptr());
    check_contiguous_array(data_arr, "data array", md_layer->datum_channels(),
        md_layer->datum_height(), md_layer->datum_width());
    check_contiguous_array(labels_arr, "labels array", 1, 1, 1);
    if (PyArray_DIMS(data_arr)[0] != PyArray_DIMS(labels_arr)[0]) {
      throw std::runtime_error("data and labels must have the same first"
          " dimension");
    }
    if (PyArray_DIMS(data_arr)[0] % md_layer->batch_size() != 0) {
      throw std::runtime_error("first dimensions of input arrays must be a"
          " multiple of batch size");
    }

    // hold references
    input_data_ = data_obj;
    input_labels_ = labels_obj;

    md_layer->Reset(static_cast<float*>(PyArray_DATA(data_arr)),
        static_cast<float*>(PyArray_DATA(labels_arr)),
        PyArray_DIMS(data_arr)[0]);
  }

  // The caffe::Caffe utility functions.
  void set_mode_cpu() { Caffe::set_mode(Caffe::CPU); }
  void set_mode_gpu() { Caffe::set_mode(Caffe::GPU); }
  void set_phase_train() { Caffe::set_phase(Caffe::TRAIN); }
  void set_phase_test() { Caffe::set_phase(Caffe::TEST); }
  void set_device(int device_id) { Caffe::SetDevice(device_id); }

  vector<CaffeBlob> blobs() {
    vector<CaffeBlob> result;
    for (int i = 0; i < net_->blobs().size(); ++i) {
      result.push_back(CaffeBlob(net_->blobs()[i], net_->blob_names()[i]));
    }
    return result;
  }

  vector<CaffeLayer> layers() {
    vector<CaffeLayer> result;
    for (int i = 0; i < net_->layers().size(); ++i) {
      result.push_back(CaffeLayer(net_->layers()[i], net_->layer_names()[i]));
    }
    return result;
  }

  // The pointer to the internal caffe::Net instant.
  shared_ptr<Net<float> > net_;
  // if taking input from an ndarray, we need to hold references
  object input_data_;
  object input_labels_;
};

class CaffeSGDSolver {
 public:
  explicit CaffeSGDSolver(const string& param_file) {
    // as in CaffeNet, (as a convenience, not a guarantee), create a Python
    // exception if param_file can't be opened
    CheckFile(param_file);
    solver_.reset(new SGDSolver<float>(param_file));
    // we need to explicitly store the net wrapper, rather than constructing
    // it on the fly, so that it can hold references to Python objects
    net_.reset(new CaffeNet(solver_->net()));
  }

  shared_ptr<CaffeNet> net() { return net_; }
  void Solve() { return solver_->Solve(); }
  void SolveResume(const string& resume_file) {
    CheckFile(resume_file);
    return solver_->Solve(resume_file);
  }

 protected:
  shared_ptr<CaffeNet> net_;
  shared_ptr<SGDSolver<float> > solver_;
};


// The boost python module definition.
BOOST_PYTHON_MODULE(_caffe) {
  // below, we prepend an underscore to methods that will be replaced
  //  in Python
  boost::python::class_<CaffeNet, shared_ptr<CaffeNet> >(
      "Net", boost::python::init<string, string>())
      .def(boost::python::init<string>())
      .def("Forward",           &CaffeNet::Forward)
      .def("ForwardPrefilled",  &CaffeNet::ForwardPrefilled)
      .def("Backward",          &CaffeNet::Backward)
      .def("set_mode_cpu",      &CaffeNet::set_mode_cpu)
      .def("set_mode_gpu",      &CaffeNet::set_mode_gpu)
      .def("set_phase_train",   &CaffeNet::set_phase_train)
      .def("set_phase_test",    &CaffeNet::set_phase_test)
      .def("set_device",        &CaffeNet::set_device)
      .add_property("_blobs",   &CaffeNet::blobs)
      .add_property("layers",   &CaffeNet::layers)
      .def("_set_input_arrays", &CaffeNet::set_input_arrays);

  boost::python::class_<CaffeBlob, CaffeBlobWrap>(
      "Blob", boost::python::no_init)
      .add_property("name",     &CaffeBlob::name)
      .add_property("num",      &CaffeBlob::num)
      .add_property("channels", &CaffeBlob::channels)
      .add_property("height",   &CaffeBlob::height)
      .add_property("width",    &CaffeBlob::width)
      .add_property("count",    &CaffeBlob::count)
      .add_property("data",     &CaffeBlobWrap::get_data)
      .add_property("diff",     &CaffeBlobWrap::get_diff);

  boost::python::class_<CaffeLayer>(
      "Layer", boost::python::no_init)
      .add_property("name",  &CaffeLayer::name)
      .add_property("blobs", &CaffeLayer::blobs);

  boost::python::class_<CaffeSGDSolver, boost::noncopyable>(
      "SGDSolver", boost::python::init<string>())
      .add_property("net", &CaffeSGDSolver::net)
      .def("solve",        &CaffeSGDSolver::Solve)
      .def("solve",        &CaffeSGDSolver::SolveResume);

  boost::python::class_<vector<CaffeBlob> >("BlobVec")
      .def(vector_indexing_suite<vector<CaffeBlob>, true>());

  boost::python::class_<vector<CaffeLayer> >("LayerVec")
      .def(vector_indexing_suite<vector<CaffeLayer>, true>());

  import_array();
}
