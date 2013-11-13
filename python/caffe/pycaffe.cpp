// Copyright Yangqing Jia 2013
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include "caffe/caffe.hpp"


using namespace caffe;
using namespace boost::python;

// For python, we will simply use float.
// A simple wrapper over CaffeNet that runs the forward process.
struct CaffeNet
{
  CaffeNet(string param_file, string pretrained_param_file,
      boost::python::list bottom) {
    vector<int> bottom_vec;
    for (int i = 0; i < boost::python::len(bottom); ++i) {
      bottom_vec.push_back(boost::python::extract<int>(bottom[i]));
    }
    net_.reset(new Net<float>(param_file, bottom_vec));
    net_->CopyTrainedLayersFrom(pretrained_param_file);
  }

  virtual ~CaffeNet() {}

  void Forward(boost::python::list bottom, boost::python::list top) {
    vector<Blob<float>*>& input_blobs = net_->input_blobs();
    CHECK_EQ(boost::python::len(bottom), input_blobs.size());
    CHECK_EQ(boost::python::len(top), net_->num_outputs());
    // First, copy the input
    for (int i = 0; i < input_blobs.size(); ++i) {
      boost::python::object elem = bottom[i];
      PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(elem.ptr());
      CHECK_EQ(PyArray_NDIM(arr), 4);
      CHECK_EQ(PyArray_ITEMSIZE(arr), 4);
      npy_intp* dims = PyArray_DIMS(arr);
      CHECK_EQ(dims[0], input_blobs[i]->num());
      CHECK_EQ(dims[1], input_blobs[i]->channels());
      CHECK_EQ(dims[2], input_blobs[i]->height());
      CHECK_EQ(dims[3], input_blobs[i]->width());
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
    LOG(INFO) << "Start";
    const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled();
    LOG(INFO) << "End";
    for (int i = 0; i < output_blobs.size(); ++i) {
      boost::python::object elem = top[i];
      PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(elem.ptr());
      CHECK(PyArray_FLAGS(arr) & NPY_ARRAY_C_CONTIGUOUS);
      CHECK_EQ(PyArray_NDIM(arr), 4);
      CHECK_EQ(PyArray_ITEMSIZE(arr), 4);
      npy_intp* dims = PyArray_DIMS(arr);
      CHECK_EQ(dims[0], output_blobs[i]->num());
      CHECK_EQ(dims[1], output_blobs[i]->channels());
      CHECK_EQ(dims[2], output_blobs[i]->height());
      CHECK_EQ(dims[3], output_blobs[i]->width());
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
  
  void set_mode_cpu() { Caffe::set_mode(Caffe::CPU); }
  void set_mode_gpu() { Caffe::set_mode(Caffe::GPU); }
  
  void set_phase_train() { Caffe::set_phase(Caffe::TRAIN); }
  void set_phase_test() { Caffe::set_phase(Caffe::TEST); }

	shared_ptr<Net<float> > net_;
};

BOOST_PYTHON_MODULE(pycaffe)
{
  class_<CaffeNet>("CaffeNet", init<string, string, boost::python::list>())
      .def("Forward", &CaffeNet::Forward)
      .def("set_mode_cpu", &CaffeNet::set_mode_cpu)
      .def("set_mode_gpu", &CaffeNet::set_mode_gpu)
      .def("set_phase_train", &CaffeNet::set_phase_train)
      .def("set_phase_test", &CaffeNet::set_phase_test)
  ;
}