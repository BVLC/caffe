// pycaffe provides a wrapper of the caffe::Net class as well as some
// caffe::Caffe functions so that one could easily call it from Python.
// Note that for Python, we will simply use float as the data type.

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

using boost::python::dict;
using boost::python::extract;
using boost::python::len;
using boost::python::list;
using boost::python::object;
using boost::python::handle;
using boost::python::vector_indexing_suite;

namespace caffe {

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
// to Python
class PyBlob {
 public:
  PyBlob(const shared_ptr<Blob<float> > &blob, const string& name)
      : blob_(blob), name_(name) {}

  string name() const { return name_; }
  int num() const { return blob_->num(); }
  int channels() const { return blob_->channels(); }
  int height() const { return blob_->height(); }
  int width() const { return blob_->width(); }
  int count() const { return blob_->count(); }

  // this is here only to satisfy boost's vector_indexing_suite
  bool operator == (const PyBlob &other) {
      return this->blob_ == other.blob_;
  }

 protected:
  shared_ptr<Blob<float> > blob_;
  string name_;
};


// We need another wrapper (used as boost::python's HeldType) that receives a
// self PyObject * which we can use as ndarray.base, so that data/diff memory
// is not freed while still being used in Python.
class PyBlobWrap : public PyBlob {
 public:
  PyBlobWrap(PyObject *p, const PyBlob &blob)
      : PyBlob(blob), self_(p) {}

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


class PyLayer {
 public:
  PyLayer(const shared_ptr<Layer<float> > &layer, const string &name)
    : layer_(layer), name_(name) {}

  string name() const { return name_; }
  vector<PyBlob> blobs() {
    vector<PyBlob> result;
    for (int i = 0; i < layer_->blobs().size(); ++i) {
      result.push_back(PyBlob(layer_->blobs()[i], name_));
    }
    return result;
  }

  // this is here only to satisfy boost's vector_indexing_suite
  bool operator == (const PyLayer &other) {
      return this->layer_ == other.layer_;
  }

 protected:
  shared_ptr<Layer<float> > layer_;
  string name_;
};


// A simple wrapper over PyNet that runs the forward process.
struct PyNet {
  // For cases where parameters will be determined later by the Python user,
  // create a Net with unallocated parameters (which will not be zero-filled
  // when accessed).
  explicit PyNet(string param_file) {
    Init(param_file);
  }

  PyNet(string param_file, string pretrained_param_file) {
    Init(param_file);
    CheckFile(pretrained_param_file);
    net_->CopyTrainedLayersFrom(pretrained_param_file);
  }

  explicit PyNet(shared_ptr<Net<float> > net)
      : net_(net) {}

  void Init(string param_file) {
    CheckFile(param_file);
    net_.reset(new Net<float>(param_file));
  }


  virtual ~PyNet() {}

  // Generate Python exceptions for badly shaped or discontiguous arrays.
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

  void Forward(int start, int end) {
    net_->ForwardFromTo(start, end);
  }

  void Backward(int start, int end) {
    net_->BackwardFromTo(start, end);
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

  // save the network weights to binary proto for net surgeries.
  void save(string filename) {
    NetParameter net_param;
    net_->ToProto(&net_param, false);
    WriteProtoToBinaryFile(net_param, filename.c_str());
  }

  // The caffe::Caffe utility functions.
  void set_mode_cpu() { Caffe::set_mode(Caffe::CPU); }
  void set_mode_gpu() { Caffe::set_mode(Caffe::GPU); }
  void set_phase_train() { Caffe::set_phase(Caffe::TRAIN); }
  void set_phase_test() { Caffe::set_phase(Caffe::TEST); }
  void set_device(int device_id) { Caffe::SetDevice(device_id); }

  vector<PyBlob> blobs() {
    vector<PyBlob> result;
    for (int i = 0; i < net_->blobs().size(); ++i) {
      result.push_back(PyBlob(net_->blobs()[i], net_->blob_names()[i]));
    }
    return result;
  }

  vector<PyLayer> layers() {
    vector<PyLayer> result;
    for (int i = 0; i < net_->layers().size(); ++i) {
      result.push_back(PyLayer(net_->layers()[i], net_->layer_names()[i]));
    }
    return result;
  }

  list inputs() {
    list input_blob_names;
    for (int i = 0; i < net_->input_blob_indices().size(); ++i) {
      input_blob_names.append(
          net_->blob_names()[net_->input_blob_indices()[i]]);
    }
    return input_blob_names;
  }

  list outputs() {
    list output_blob_names;
    for (int i = 0; i < net_->output_blob_indices().size(); ++i) {
      output_blob_names.append(
          net_->blob_names()[net_->output_blob_indices()[i]]);
    }
    return output_blob_names;
  }

  // The pointer to the internal caffe::Net instant.
  shared_ptr<Net<float> > net_;
  // Input preprocessing configuration attributes.
  dict mean_;
  dict input_scale_;
  dict raw_scale_;
  dict channel_swap_;
  // if taking input from an ndarray, we need to hold references
  object input_data_;
  object input_labels_;
};

class PySGDSolver {
 public:
  explicit PySGDSolver(const string& param_file) {
    // as in PyNet, (as a convenience, not a guarantee), create a Python
    // exception if param_file can't be opened
    CheckFile(param_file);
    solver_.reset(new SGDSolver<float>(param_file));
    // we need to explicitly store the net wrapper, rather than constructing
    // it on the fly, so that it can hold references to Python objects
    net_.reset(new PyNet(solver_->net()));
  }

  shared_ptr<PyNet> net() { return net_; }
  void Solve() { return solver_->Solve(); }
  void SolveResume(const string& resume_file) {
    CheckFile(resume_file);
    return solver_->Solve(resume_file);
  }

 protected:
  shared_ptr<PyNet> net_;
  shared_ptr<SGDSolver<float> > solver_;
};


// The boost_python module definition.
BOOST_PYTHON_MODULE(_caffe) {
  // below, we prepend an underscore to methods that will be replaced
  // in Python
  boost::python::class_<PyNet, shared_ptr<PyNet> >(
      "Net", boost::python::init<string, string>())
      .def(boost::python::init<string>())
      .def("_forward",              &PyNet::Forward)
      .def("_backward",             &PyNet::Backward)
      .def("set_mode_cpu",          &PyNet::set_mode_cpu)
      .def("set_mode_gpu",          &PyNet::set_mode_gpu)
      .def("set_phase_train",       &PyNet::set_phase_train)
      .def("set_phase_test",        &PyNet::set_phase_test)
      .def("set_device",            &PyNet::set_device)
      .add_property("_blobs",       &PyNet::blobs)
      .add_property("layers",       &PyNet::layers)
      .add_property("inputs",       &PyNet::inputs)
      .add_property("outputs",      &PyNet::outputs)
      .add_property("mean",         &PyNet::mean_)
      .add_property("input_scale",  &PyNet::input_scale_)
      .add_property("raw_scale",    &PyNet::raw_scale_)
      .add_property("channel_swap", &PyNet::channel_swap_)
      .def("_set_input_arrays",     &PyNet::set_input_arrays)
      .def("save",                  &PyNet::save);

  boost::python::class_<PyBlob, PyBlobWrap>(
      "Blob", boost::python::no_init)
      .add_property("name",     &PyBlob::name)
      .add_property("num",      &PyBlob::num)
      .add_property("channels", &PyBlob::channels)
      .add_property("height",   &PyBlob::height)
      .add_property("width",    &PyBlob::width)
      .add_property("count",    &PyBlob::count)
      .add_property("data",     &PyBlobWrap::get_data)
      .add_property("diff",     &PyBlobWrap::get_diff);

  boost::python::class_<PyLayer>(
      "Layer", boost::python::no_init)
      .add_property("name",  &PyLayer::name)
      .add_property("blobs", &PyLayer::blobs);

  boost::python::class_<PySGDSolver, boost::noncopyable>(
      "SGDSolver", boost::python::init<string>())
      .add_property("net", &PySGDSolver::net)
      .def("solve",        &PySGDSolver::Solve)
      .def("solve",        &PySGDSolver::SolveResume);

  boost::python::class_<vector<PyBlob> >("BlobVec")
      .def(vector_indexing_suite<vector<PyBlob>, true>());

  boost::python::class_<vector<PyLayer> >("LayerVec")
      .def(vector_indexing_suite<vector<PyLayer>, true>());

  import_array();
}

}  // namespace caffe
