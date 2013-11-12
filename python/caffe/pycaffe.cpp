#include <boost/python.hpp>
#include "caffe/caffe.hpp"

using namespace caffe;
using namespace boost::python;

// For python, we will simply use float. This is wrapped in a
#define PTYPE float

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

  string Forward(const string& input_blobs) {
    return net_->Forward(input_blobs);
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