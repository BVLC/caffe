#ifndef PYTHON_CAFFE__CAFFE_HPP_
#define PYTHON_CAFFE__CAFFE_HPP_

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <numpy/arrayobject.h>

// these need to be included after boost on OS X
#include <string>  // NOLINT(build/include_order)
#include <vector>  // NOLINT(build/include_order)

#include "caffe/caffe.hpp"

namespace bp = boost::python;
using boost::shared_ptr;

namespace caffe {

// wrap shared_ptr<Blob> in a class that we construct in C++ and pass
// to Python
template <typename Dtype>
class PyBlob {
 public:
  PyBlob(const shared_ptr<Blob<Dtype> > &blob, const string& name)
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
  shared_ptr<Blob<Dtype> > blob_;
  string name_;
};

// We need another wrapper (used as boost::python's HeldType) that receives a
// self PyObject * which we can use as ndarray.base, so that data/diff memory
// is not freed while still being used in Python.
class PyBlobWrap : public PyBlob<float> {
 public:
  PyBlobWrap(PyObject *p, const PyBlob<float> &blob)
      : PyBlob<float>(blob), self_(p) {}

  bp::object get_data();
  bp::object get_diff();

 private:
  PyObject *self_;
};

class PyLayer {
 public:
  PyLayer(const shared_ptr<Layer<float> > &layer, const string &name)
    : layer_(layer), name_(name) {}

  string name() const { return name_; }
  vector<PyBlob<float> > blobs() {
    vector<PyBlob<float> > result;
    for (int i = 0; i < layer_->blobs().size(); ++i) {
      result.push_back(PyBlob<float>(layer_->blobs()[i], name_));
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

class PyNet {
 public:
  // For cases where parameters will be determined later by the Python user,
  // create a Net with unallocated parameters (which will not be zero-filled
  // when accessed).
  explicit PyNet(string param_file) { Init(param_file); }
  PyNet(string param_file, string pretrained_param_file);
  explicit PyNet(shared_ptr<Net<float> > net)
      : net_(net) {}
  virtual ~PyNet() {}

  void Init(string param_file);


  // Generate Python exceptions for badly shaped or discontiguous arrays.
  inline void check_contiguous_array(PyArrayObject* arr, string name,
      int channels, int height, int width);

  void Forward(int start, int end) { net_->ForwardFromTo(start, end); }
  void Backward(int start, int end) { net_->BackwardFromTo(start, end); }

  void set_input_arrays(bp::object data_obj, bp::object labels_obj);

  // Save the network weights to binary proto for net surgeries.
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

  vector<PyBlob<float> > blobs() {
    vector<PyBlob<float> > result;
    for (int i = 0; i < net_->blobs().size(); ++i) {
      result.push_back(PyBlob<float>(net_->blobs()[i], net_->blob_names()[i]));
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

  bp::list inputs() {
    bp::list input_blob_names;
    for (int i = 0; i < net_->input_blob_indices().size(); ++i) {
      input_blob_names.append(
          net_->blob_names()[net_->input_blob_indices()[i]]);
    }
    return input_blob_names;
  }

  bp::list outputs() {
    bp::list output_blob_names;
    for (int i = 0; i < net_->output_blob_indices().size(); ++i) {
      output_blob_names.append(
          net_->blob_names()[net_->output_blob_indices()[i]]);
    }
    return output_blob_names;
  }

  // Input preprocessing configuration attributes. These are public for
  // direct access from Python.
  bp::dict mean_;
  bp::dict input_scale_;
  bp::dict raw_scale_;
  bp::dict channel_swap_;

 protected:
  // The pointer to the internal caffe::Net instant.
  shared_ptr<Net<float> > net_;
  // if taking input from an ndarray, we need to hold references
  bp::object input_data_;
  bp::object input_labels_;
};

class PySGDSolver {
 public:
  explicit PySGDSolver(const string& param_file);

  shared_ptr<PyNet> net() { return net_; }
  void Solve() { return solver_->Solve(); }
  void SolveResume(const string& resume_file);

 protected:
  shared_ptr<PyNet> net_;
  shared_ptr<SGDSolver<float> > solver_;
};

}  // namespace caffe

#endif
