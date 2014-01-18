// Copyright Yangqing Jia 2013
// pycaffe provides a wrapper of the caffe::Net class as well as some
// caffe::Caffe functions so that one could easily call it from Python.
// Note that for python, we will simply use float as the data type.

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "boost/python.hpp"
#include "boost/python/suite/indexing/vector_indexing_suite.hpp"
#include "numpy/arrayobject.h"

#include <string>  // NOLINT(build/include_order)
#include <vector>  // NOLINT(build/include_order)

#include "caffe/caffe.hpp"
#include "stitch_pyramid/PyramidStitcher.h" //also includes JPEGImage, Patchwork, etc

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


// wrap shared_ptr<Blob<float> > in a class that we construct in C++ and pass
//  to Python
class CaffeBlob {
 public:
  CaffeBlob(const shared_ptr<Blob<float> > &blob, const string& name)
      : blob_(blob), name_(name) {}

  explicit CaffeBlob(const shared_ptr<Blob<float> > &blob)
      : blob_(blob) {}

  CaffeBlob()
  {}

  string name() const { return name_; }
  int num() const { return blob_->num(); }
  int channels() const { return blob_->channels(); }
  int height() const { return blob_->height(); }
  int width() const { return blob_->width(); }
  int count() const { return blob_->count(); }

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
  CaffeBlobWrap(PyObject *p, const shared_ptr<Blob<float> > &blob)
      : CaffeBlob(blob), self_(p) {}

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



// A simple wrapper over CaffeNet that runs the forward process.
struct CaffeNet {
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

  //float* -> numpy -> boost python (which can be returned to Python)
  boost::python::object array_to_boostPython_4d(float* pyramid_float, 
                                                int batchsize, int depth_, int MaxHeight_, int MaxWidth_)
  {

    npy_intp dims[4] = {batchsize, depth_, MaxHeight_, MaxWidth_}; //in floats
    PyArrayObject* pyramid_float_npy = (PyArrayObject*)PyArray_New( &PyArray_Type, 4, dims, NPY_FLOAT, 0, pyramid_float, 0, 0, 0 ); //not specifying strides

    //thanks: stackoverflow.com/questions/19185574 
    boost::python::object pyramid_float_npy_boost(boost::python::handle<>((PyObject*)pyramid_float_npy));
    return pyramid_float_npy_boost;
  } 

  // for now, one batch at a time. (later, can modify this to allocate & fill a >1 batch 4d array)
  // @param  jpeg = typically a plane from Patchwork, in packed JPEG<uint8_t> [RGB,RGB,RGB] format
  // @return numpy float array of jpeg, in unpacked [RRRRR..,GGGGG..,BBBBB..] format
  PyArrayObject* JPEGImage_to_numpy_float(JPEGImage &jpeg){

    int depth = jpeg.depth();
    int height = jpeg.height();
    int width = jpeg.width();
    int batchsize = 1;
    npy_intp dims[4] = {batchsize, depth, height, width};

    PyArrayObject* jpeg_float_npy = (PyArrayObject*)PyArray_New( &PyArray_Type, 4, dims, NPY_FLOAT, 0, 0, 0, 0, 0 ); //numpy malloc

    //TODO: make sure of RGB vs BGR (just for documentation purposes)

    //copy jpeg into jpeg_float_npy
    for(int y=0; y<height; y++){
        for(int x=0; x<width; x++){
            for(int ch=0; ch<depth; ch++){
                //jpeg:           row-major, packed RGB, RGB, ...          uint8_t.
                //jpeg_float_npy: row-major, unpacked RRRR..,GGGG..,BBBB.. float.
                *((float *)PyArray_GETPTR4( jpeg_float_npy, 0, ch, y, x)) = jpeg.bits()[y*width*depth + x*depth + ch];
            }
        }
    }

    return jpeg_float_npy;
  }

  //obtains resultPlane dims with shared ptr to net_
  PyArrayObject* allocate_resultPlane(){

    int batchsize = net_->output_blobs()[0]->num();
    int depth = net_->output_blobs()[0]->channels();
    int width = net_->output_blobs()[0]->width();
    int height = net_->output_blobs()[0]->height();
    npy_intp dims[4] = {batchsize, depth, height, width};
    printf("    in allocate_resultPlane(). OUTPUT_BLOBS... batchsize=%d, depth=%d, width=%d, height=%d \n", batchsize, depth, width, height);

    //PyArrayObject* resultPlane = NULL; //stub
    PyArrayObject* resultPlane = (PyArrayObject*)PyArray_New( &PyArray_Type, 4, dims, NPY_FLOAT, 0, 0, 0, 0, 0 ); //numpy malloc
    return resultPlane;
  }

  //void extract_featpyramid(string file){
  boost::python::list extract_featpyramid(string file){

    int padding = 8;
    int interval = 10;
    int planeDim = net_->input_blobs()[0]->width(); //assume that all preallocated blobs are same size

    assert(net_->input_blobs()[0]->width() == net_->input_blobs()[0]->height()); //assume square planes in Caffe. (can relax this if necessary)
    assert(net_->input_blobs()[0]->num() == 1); //for now, one plane at a time.)
    //TODO: verify that top-upsampled version of input img fits within planeDim

    //TODO: think about how to use image_mean.
    //        ignoring image_mean for now, because the 'subtract image mean from whatever input region we get'
    //        thing in r-cnn feels kinda silly.
 
    Patchwork patchwork = stitch_pyramid(file, padding, interval, planeDim); 

    int planeID = 0; //TODO: append multiple blobs to blobs_{top,bottom}, iterating to planes_.size()
                     //         then, run Forward() on the list of blobs.

    //prep input data
    JPEGImage* currPlane = &(patchwork.planes_[planeID]);
    PyArrayObject* currPlane_npy = JPEGImage_to_numpy_float(*currPlane); //TODO: agree on dereference and/or pass-by-ref JPEGImage currPlane
    boost::python::object currPlane_npy_boost(boost::python::handle<>((PyObject*)currPlane_npy)); //numpy -> wrap in boost
    boost::python::list blobs_bottom; //input to Caffe::Forward
    blobs_bottom.append(currPlane_npy_boost); //put the output array in list [list length = 1, because batchsize = 1]

    //prep output space
    //PyArrayObject* resultPlane_npy = (PyArrayObject*)PyArray_NewLikeArray(currPlane_npy, NPY_KEEPORDER, NULL, 1); //same size/shape as currPlane_npy
    PyArrayObject* resultPlane_npy = allocate_resultPlane(); //gets resultPlane dims from shared ptr to net_->output_blobs()
    boost::python::object resultPlane_npy_boost(boost::python::handle<>((PyObject*)resultPlane_npy)); //numpy -> wrap in boost
    boost::python::list blobs_top; //output buffer for Caffe::Forward
    blobs_top.append(resultPlane_npy_boost); //put the output array in list [list length = 1, because batchsize = 1]
    
    Forward(blobs_bottom, blobs_top); //lists of blobs... bottom=input planes, top=output descriptors

    printf("\n\n    in pycaffe.cpp extract_featpyramid(). planeDim=%d\n", planeDim);

    return blobs_bottom; //for debugging only (stitched pyramid in RGB)
    //return blobs_top; //output plane(s)
  }

  //void testIO(){ } //dummy example

  //return a list containing one 4D numpy/boost array. (toy example)
  boost::python::list testIO()
  {
    int batchsize = 1;
    int depth_ = 1;
    int MaxHeight_ = 10;
    int MaxWidth_ = 10;

    //prepare data that we'll send to Python
    float* pyramid_float = (float*)malloc(sizeof(float) * batchsize * depth_ * MaxHeight_ * MaxWidth_);
    memset(pyramid_float, 0, sizeof(float) * batchsize * depth_ * MaxHeight_ * MaxWidth_);
    pyramid_float[10] = 123; //test -- see if it shows up in Python

    boost::python::object pyramid_float_npy_boost = array_to_boostPython_4d(pyramid_float, batchsize, depth_, MaxHeight_, MaxWidth_);

    boost::python::list blobs_top_boost; //list to return
    blobs_top_boost.append(pyramid_float_npy_boost); //put the output array in list

    return blobs_top_boost; //compile error: return-statement with no value  
  }

  void testString(string st){
    printf("    string from python: %s \n", st.c_str());
  }

  void testInt(int i){
    printf("    int from python: %d \n", i);
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

  vector<CaffeBlob> params() {
      vector<CaffeBlob> result;
      int ix = 0;
      for (int i = 0; i < net_->layers().size(); ++i) {
        for (int j = 0; j < net_->layers()[i]->blobs().size(); ++j) {
          result.push_back(
              CaffeBlob(net_->params()[ix], net_->layer_names()[i]));
          ix++;
        }
      }
      return result;
  }

  // The pointer to the internal caffe::Net instant.
  shared_ptr<Net<float> > net_;
};



// The boost python module definition.
BOOST_PYTHON_MODULE(pycaffe) {
  boost::python::class_<CaffeNet>(
      "CaffeNet", boost::python::init<string, string>())
      .def("Forward",         &CaffeNet::Forward)
      .def("Backward",        &CaffeNet::Backward)
      .def("set_mode_cpu",    &CaffeNet::set_mode_cpu)
      .def("set_mode_gpu",    &CaffeNet::set_mode_gpu)
      .def("set_phase_train", &CaffeNet::set_phase_train)
      .def("set_phase_test",  &CaffeNet::set_phase_test)
      .def("set_device",      &CaffeNet::set_device)
      .def("testIO",          &CaffeNet::testIO) //Forrest's test (return a numpy array)
      .def("testString",      &CaffeNet::testString) 
      .def("testInt",         &CaffeNet::testInt)
      .def("extract_featpyramid",         &CaffeNet::extract_featpyramid) //NEW
      .def("blobs",           &CaffeNet::blobs)
      .def("params",          &CaffeNet::params);

  boost::python::class_<CaffeBlob, CaffeBlobWrap>(
      "CaffeBlob", boost::python::no_init)
      .add_property("name",     &CaffeBlob::name)
      .add_property("num",      &CaffeBlob::num)
      .add_property("channels", &CaffeBlob::channels)
      .add_property("height",   &CaffeBlob::height)
      .add_property("width",    &CaffeBlob::width)
      .add_property("count",    &CaffeBlob::count)
      .add_property("data",     &CaffeBlobWrap::get_data)
      .add_property("diff",     &CaffeBlobWrap::get_diff);

  boost::python::class_<vector<CaffeBlob> >("BlobVec")
      .def(vector_indexing_suite<vector<CaffeBlob>, true>());

  import_array();
}
