// Produce deprecation warnings (needs to come before arrayobject.h inclusion).
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <boost/make_shared.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/exception_translator.hpp>
#include <boost/python/implicit.hpp>
#include <boost/python/init.hpp>
#include <boost/python/module.hpp>
#include <boost/python/object.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/variant.hpp>
#include <numpy/arrayobject.h>

// these need to be included after boost on OS X
#include <string>  // NOLINT(build/include_order)
#include <vector>  // NOLINT(build/include_order)
#include <fstream>  // NOLINT

#include "caffe/blob_creator.hpp"
#include "caffe/caffe.hpp"
#include "caffe/definitions.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/layers/python_layer.hpp"
#include "caffe/sgd_solvers.hpp"

// Temporary solution for numpy < 1.7 versions: old macro, no promises.
// You're strongly advised to upgrade to >= 1.7.
#ifndef NPY_ARRAY_C_CONTIGUOUS
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#define PyArray_SetBaseObject(arr, x) (PyArray_BASE(arr) = (x))
#endif

// Numpy integer type corresponding to int_tp
#ifdef USE_INDEX_64
#define npy_int_tp npy_long
#define npy_uint_tp npy_ulong
#else
#define npy_int_tp npy_intp
#define npy_uint_tp npy_uintp
#endif  // USE_INDEX_64

/* Fix to avoid registration warnings in pycaffe (#3960) */
#define BP_REGISTER_SHARED_PTR_TO_PYTHON(PTR, TYPES) do { \
  const boost::python::type_info info = \
    boost::python::type_id<shared_ptr<PTR<BOOST_PP_SEQ_ENUM(TYPES)> > >(); \
  const boost::python::converter::registration* reg = \
    boost::python::converter::registry::query(info); \
  if (reg == NULL) { \
    bp::register_ptr_to_python<shared_ptr<PTR<BOOST_PP_SEQ_ENUM(TYPES)> > >(); \
  } else if ((*reg).m_to_python == NULL) { \
    bp::register_ptr_to_python<shared_ptr<PTR<BOOST_PP_SEQ_ENUM(TYPES)> > >(); \
  } \
} while (0)

#define BP_REGISTER_SHARED_PTR_TO_PYTHON_NO_TEMPLATE(PTR) do { \
  const boost::python::type_info info = \
    boost::python::type_id<shared_ptr<PTR > >(); \
  const boost::python::converter::registration* reg = \
    boost::python::converter::registry::query(info); \
  if (reg == NULL) { \
    bp::register_ptr_to_python<shared_ptr<PTR > >(); \
  } else if ((*reg).m_to_python == NULL) { \
    bp::register_ptr_to_python<shared_ptr<PTR > >(); \
  } \
} while (0)

#if defined(_MSC_VER) && (_MSC_FULL_VER >= 190024210)
// Workaround for VS 2015 Update 3 which breaks boost python
// See: http://stackoverflow.com/questions/38261530/unresolved-external-symbols-since-visual-studio-2015-update-3-boost-python-link
// and https://msdn.microsoft.com/vs-knownissues/vs2015-update3
#define BP_GET_POINTER(cls) \
namespace boost { \
template <> \
const volatile caffe::cls * \
get_pointer(const volatile caffe::cls *c) { \
    return c; \
} \
}

#define BP_GET_POINTER_T(cls, dtype) BP_GET_POINTER(cls<dtype>)

// forward declare the NCCL class
// in case we are not using NCCL
namespace caffe {
template <typename Dtype> class NCCL;
}

BP_GET_POINTER_T(Net, float);
BP_GET_POINTER_T(Layer, float);
BP_GET_POINTER_T(Solver, float);
BP_GET_POINTER_T(SGDSolver, float);
BP_GET_POINTER_T(NesterovSolver, float);
BP_GET_POINTER_T(AdaGradSolver, float);
BP_GET_POINTER_T(RMSPropSolver, float);
BP_GET_POINTER_T(AdaDeltaSolver, float);
BP_GET_POINTER_T(AdamSolver, float);
// BP_GET_POINTER_T(NCCL, float);
BP_GET_POINTER(Timer);
BP_GET_POINTER(LayerParameter);
BP_GET_POINTER(NetParameter);
BP_GET_POINTER(NetState);

#endif

namespace bp = boost::python;

namespace caffe {
// Data type coupling code
NPY_TYPES proto_to_npy_type(DataType proto_data_type) {
  switch(proto_data_type) {
    case CAFFE_HALF:
      return NPY_FLOAT16;
    case CAFFE_FLOAT:
      return NPY_FLOAT32;
    case CAFFE_DOUBLE:
      return NPY_FLOAT64;
    case CAFFE_INT8_QUANTIZED:
      return NPY_BYTE;
    case CAFFE_INT16_QUANTIZED:
      return NPY_SHORT;
    case CAFFE_INT32_QUANTIZED:
      return NPY_INT;
    case CAFFE_INT64_QUANTIZED:
      return NPY_INT64;
    default:
      return NPY_FLOAT32;
  }
}

DataType npy_to_proto_type(NPY_TYPES npy_data_type) {
  switch (npy_data_type) {
    case NPY_FLOAT16:
      return CAFFE_HALF;
    case NPY_FLOAT32:
      return CAFFE_FLOAT;
    case NPY_FLOAT64:
      return CAFFE_DOUBLE;
    case NPY_BYTE:
      return CAFFE_INT8_QUANTIZED;
    case NPY_SHORT:
      return CAFFE_INT16_QUANTIZED;
    case NPY_INT:
      return CAFFE_INT32_QUANTIZED;
    case NPY_INT64:
      return CAFFE_INT64_QUANTIZED;
    default:
      return CAFFE_FLOAT;
  }
}

typedef boost::variant<half_fp, float, double,
                       uint8_t, uint16_t, uint32_t,
                       uint64_t> variant_proto_types;
typedef boost::variant<half_fp*, float*, double*,
                       uint8_t*, uint16_t*, uint32_t*,
                       uint64_t*> variant_proto_ptr_types;
typedef boost::variant<vector<half_fp>, vector<float>, vector<double>,
                       vector<uint8_t>, vector<uint16_t>, vector<uint32_t>,
                       vector<uint64_t> > variant_proto_vec_types;

struct variant_proto_types_to_object : boost::static_visitor<PyObject*> {
  static result_type convert(variant_proto_types const &v) {
    return boost::apply_visitor(variant_proto_types_to_object(), v);
  }

  template<typename T>
  result_type operator()(T const &t) const {
    return boost::python::incref(boost::python::object(t).ptr());
  }
};

struct variant_proto_ptr_types_to_object : boost::static_visitor<PyObject*> {
  static result_type convert(variant_proto_ptr_types const &v) {
    return boost::apply_visitor(variant_proto_ptr_types_to_object(), v);
  }

  template<typename T>
  result_type operator()(T const &t) const {
    return boost::python::incref(boost::python::object(t).ptr());
  }
};

struct variant_proto_vec_types_to_object : boost::static_visitor<PyObject*> {
  static result_type convert(variant_proto_vec_types const &v) {
    return boost::apply_visitor(variant_proto_vec_types_to_object(), v);
  }

  template<typename T>
  result_type operator()(T const &t) const {
    return boost::python::incref(boost::python::object(t).ptr());
  }
};


// Selecting mode.
void set_mode_cpu() { Caffe::set_mode(Caffe::CPU); }
void set_mode_gpu() { Caffe::set_mode(Caffe::GPU); }
void select_device(int id, bool listId) { Caffe::SelectDevice(id, listId); }
void set_devices(bp::tuple args) {
  vector<int> devices(bp::len(args));
  for (int i = 0; i < bp::len(args); ++i) {
    devices[i] = bp::extract<int>(args[i]);
  }
  Caffe::SetDevices(devices);
}


void InitLog() {
  ::google::InitGoogleLogging("");
#ifndef _MSC_VER
  // this symbol is undefined on windows
  ::google::InstallFailureSignalHandler();
#endif  // _MSC_VER
}

void InitLogLevel(int level) {
  FLAGS_minloglevel = level;
  InitLog();
}

void InitLogLevelPipe(int level, bool std_err) {
  FLAGS_minloglevel = level;
  FLAGS_logtostderr = std_err;
  InitLog();
}

void Log(const string& s) {
  LOG(INFO) << s;
}

void set_random_seed(unsigned int seed) { Caffe::set_random_seed(seed,
                                          Caffe::GetDefaultDevice()); }

// For convenience, check that input files can be opened, and raise an
// exception that boost will send to Python if not (caffe could still crash
// later if the input files are disturbed before they are actually used, but
// this saves frustration in most cases).
static void CheckFile(const string& filename) {
    std::ifstream f(filename.c_str());
    if (!f.good()) {
      f.close();
      throw std::runtime_error("Could not open file " + filename);
    }
    f.close();
}

void CheckContiguousArray(PyArrayObject* arr, string name,
    vector<int_tp> shape) {
  if (!(PyArray_FLAGS(arr) & NPY_ARRAY_C_CONTIGUOUS)) {
    throw std::runtime_error(name + " must be C contiguous");
  }
  // This does not have to hold anymore
  /*
  if (PyArray_NDIM(arr) != 4) {
    throw std::runtime_error(name + " must be 4-d");
  }
  */
  // This also does not have to hold anymore
  /*
  if (PyArray_TYPE(arr) != NPY_FLOAT32) {
    throw std::runtime_error(name + " must be float32");
  }
  */
  for (int_tp i = 1; i < PyArray_NDIM(arr); ++i) {
    if (PyArray_DIMS(arr)[i] != shape[i]) {
      throw std::runtime_error(
          "Shape dimension " + std::to_string(i) + " has wrong size ("
              + std::to_string(static_cast<int_tp>
                  (PyArray_DIMS(arr)[i])) + " vs. "
              + std::to_string(shape[i]) + ")");
    }
  }
}

// Net constructor
shared_ptr<NetBase> Net_Init(string network_file, int phase, int level,
                             const bp::object& stages,
                             const bp::object& weights) {
  CheckFile(network_file);

  // Convert stages from list to vector
  vector<string> stages_vector;
  if (!stages.is_none()) {
    for (int i = 0; i < len(stages); i++) {
      stages_vector.push_back(bp::extract<string>(stages[i]));
    }
  }

  NetParameter param;
  ReadNetParamsFromTextFileOrDie(network_file, &param);

  // Initialize net
  shared_ptr<NetBase> net;

  if (param.has_data_type()) {
    switch(param.data_type()) {
      case CAFFE_HALF:
#ifdef USE_HALF
        net.reset(new Net<half_fp>(network_file, static_cast<Phase>(phase),
                             Caffe::GetDefaultDevice(), level, &stages_vector));
#endif  // USE_HALF
        break;
      case CAFFE_DOUBLE:
#ifdef USE_DOUBLE
        net.reset(new Net<double>(network_file, static_cast<Phase>(phase),
                             Caffe::GetDefaultDevice(), level, &stages_vector));
#endif  // USE_DOUBLE
        break;
      case CAFFE_FLOAT:
      default:
#ifdef USE_SINGLE
        net.reset(new Net<float>(network_file, static_cast<Phase>(phase),
                             Caffe::GetDefaultDevice(), level, &stages_vector));
#endif  // USE_SINGLE
        break;
    }
  } else {
    net.reset(new Net<float>(network_file, static_cast<Phase>(phase),
                             Caffe::GetDefaultDevice(), level, &stages_vector));
  }

  // Load weights
  if (!weights.is_none()) {
    string weights_file_str = bp::extract<string>(weights);
    CheckFile(weights_file_str);
    net->CopyTrainedLayersFrom(weights_file_str);
  }

  return net;
}

void Net_Save(const NetBase& net, string filename, bool write_diff = false) {
  NetParameter net_param;
  net.ToProto(&net_param, write_diff);
  WriteProtoToBinaryFile(net_param, filename.c_str());
}
BOOST_PYTHON_FUNCTION_OVERLOADS(Net_SaveOverloads, Net_Save, 2, 3);

void Net_SaveHDF5(const NetBase& net, string filename) {
  net.ToHDF5(filename);
}

void Net_LoadHDF5(NetBase* net, string filename) {
  net->CopyTrainedLayersFromHDF5(filename.c_str());
}

template<typename Dtype>
void Net_SetInputArraysHelper(NetBase* net, LayerBase* layer,
                          PyArrayObject* data_arr,  PyArrayObject* labels_arr) {
  // Check that this network has an input MemoryDataLayer
  MemoryDataLayer<Dtype, Dtype, Dtype>* md_layer =
    dynamic_cast<MemoryDataLayer<Dtype, Dtype, Dtype>*>(layer);
  if (!md_layer) {
    throw std::runtime_error("set_input_arrays may only be called if the"
        " first layer is a MemoryDataLayer");
  }

  Dtype* pyarray_data_data = nullptr;
  Dtype* pyarray_data_labels = nullptr;

  int_tp num = 0;

  // check that we were passed appropriately-sized contiguous memory
  if (data_arr != nullptr) {
    CheckContiguousArray(data_arr, "data array", md_layer->shape());
    if (PyArray_DIMS(data_arr)[0] % md_layer->batch_size() != 0) {
      throw std::runtime_error("first dimensions of input arrays must be a"
          " multiple of batch size");
    }
    pyarray_data_data = static_cast<Dtype*>(PyArray_DATA(data_arr));
    num = PyArray_DIMS(data_arr)[0];
  }
  if (labels_arr != nullptr) {
    CheckContiguousArray(labels_arr, "labels array", md_layer->label_shape());
    if (PyArray_DIMS(labels_arr)[0] % md_layer->batch_size() != 0) {
      throw std::runtime_error("first dimensions of input arrays must be a"
          " multiple of batch size");
    }
    pyarray_data_labels = static_cast<Dtype*>(PyArray_DATA(labels_arr));
    num = PyArray_DIMS(data_arr)[0];
  }
  if (data_arr != nullptr && labels_arr != nullptr &&
      PyArray_DIMS(data_arr)[0] != PyArray_DIMS(labels_arr)[0]) {
    throw std::runtime_error("data and labels must have the same first"
        " dimension");
  }

  md_layer->Reset(pyarray_data_data, pyarray_data_labels, num);
}

void Net_SetLayerInputArrays(NetBase* net, LayerBase* layer,
                             bp::object data_obj, bp::object labels_obj) {
  PyArrayObject* data_arr = nullptr;
  PyArrayObject* labels_arr = nullptr;
  int data_obj_type = -1;
  int label_obj_type = -1;
  int obj_type = -1;

  if (data_obj.ptr() != bp::object().ptr()) {
    data_arr = reinterpret_cast<PyArrayObject*>(data_obj.ptr());
    data_obj_type = PyArray_TYPE(data_arr);
    obj_type = label_obj_type;
  }
  if (labels_obj.ptr() != bp::object().ptr()) {
    labels_arr = reinterpret_cast<PyArrayObject*>(labels_obj.ptr());
    label_obj_type = PyArray_TYPE(data_arr);
    obj_type = label_obj_type;
  }

  if (data_arr != nullptr && labels_arr != nullptr &&
      (data_obj_type != label_obj_type)) {
    throw std::runtime_error("Data and label array must be of the same type");
  }

  switch(obj_type) {
    case NPY_HALF:
#ifdef USE_HALF
      Net_SetInputArraysHelper<half_fp>(net, layer, data_arr, labels_arr);
#endif  // USE_HALF
      break;
    case NPY_DOUBLE:
#ifdef USE_DOUBLE
      Net_SetInputArraysHelper<double>(net, layer, data_arr, labels_arr);
#endif  // USE_DOUBLE
      break;
    case NPY_BYTE:
#ifdef USE_INT_QUANT_8
      Net_SetInputArraysHelper<uint8_t>(net, layer, data_arr, labels_arr);
#endif  // USE_INT_QUANT_8
      break;
    case NPY_SHORT:
#ifdef USE_INT_QUANT_16
      Net_SetInputArraysHelper<uint16_t>(net, layer, data_arr, labels_arr);
#endif  // USE_INT_QUANT_16
      break;
    case NPY_INT:
#ifdef USE_INT_QUANT_32
      Net_SetInputArraysHelper<uint32_t>(net, layer, data_arr, labels_arr);
#endif  // USE_INT_QUANT_32
      break;
    case NPY_INT64:
#ifdef USE_INT_QUANT_64
      Net_SetInputArraysHelper<uint64_t>(net, layer, data_arr, labels_arr);
#endif  // USE_INT_QUANT_64
      break;
    case NPY_FLOAT32:
    default:
#ifdef USE_SINGLE
      Net_SetInputArraysHelper<float>(net, layer, data_arr, labels_arr);
#endif  // USE_SINGLE
      break;
  }
}

void Net_SetInputArrays(NetBase* net, int index, bp::object data_obj,
    bp::object labels_obj) {
  LayerBase* layer = net->layers()[index].get();
  Net_SetLayerInputArrays(net, layer, data_obj, labels_obj);
}

variant_proto_vec_types Net_get_blob_loss_weights(NetBase* net) {
  switch(net->data_type()) {
    case CAFFE_HALF:
#ifdef USE_HALF
      return static_cast<Net<half_fp>*>(net)->blob_loss_weights();
#endif  // USE_HALF
    case CAFFE_DOUBLE:
#ifdef USE_DOUBLE
      return static_cast<Net<double>*>(net)->blob_loss_weights();
#endif  // USE_DOUBLE
    case CAFFE_FLOAT:
    default:
#ifdef USE_SINGLE
      return static_cast<Net<float>*>(net)->blob_loss_weights();
#endif  // USE_SINGLE
  }
}

SolverBase* GetSolver(const SolverParameter& solver_param) {
  if (solver_param.has_data_type()) {
    switch(solver_param.data_type()) {
      case CAFFE_HALF:
#ifdef USE_HALF
        return SolverRegistry<float>::CreateSolver(solver_param,
                                                   Caffe::GetDefaultDevice());
#endif  // USE_HALF
      case CAFFE_DOUBLE:
#ifdef USE_DOUBLE
        return SolverRegistry<float>::CreateSolver(solver_param,
                                                   Caffe::GetDefaultDevice());
#endif  // USE_DOUBLE
      case CAFFE_FLOAT:
      default:
#ifdef USE_SINGLE
        return SolverRegistry<float>::CreateSolver(solver_param,
                                                   Caffe::GetDefaultDevice());
#endif  // USE_SINGLE
    }
  } else {
    return SolverRegistry<float>::CreateSolver(solver_param,
                                               Caffe::GetDefaultDevice());
  }
}

SolverBase* GetSolverFromFile(const string& filename) {
  SolverParameter param;
  ReadSolverParamsFromTextFileOrDie(filename, &param);
  return GetSolver(param);
}

struct NdarrayConverterGenerator {
  template <typename T> struct apply;
};

template<>
struct NdarrayConverterGenerator::apply<variant_proto_ptr_types> {
  struct type {
    PyObject* operator() (variant_proto_ptr_types data) const {
      // Just store the data pointer, and add the shape information in postcall.
      switch(data.which()) {
        case 0:
          return PyArray_SimpleNewFromData(0, NULL, NPY_FLOAT16,
                                           boost::get<half_fp*>(data));
          break;
        case 1:
          return PyArray_SimpleNewFromData(0, NULL, NPY_FLOAT32,
                                           boost::get<float*>(data));
          break;
        case 2:
          return PyArray_SimpleNewFromData(0, NULL, NPY_FLOAT64,
                                           boost::get<double*>(data));
          break;
        case 3:
          return PyArray_SimpleNewFromData(0, NULL, NPY_BYTE,
                                           boost::get<uint8_t*>(data));
          break;
        case 4:
          return PyArray_SimpleNewFromData(0, NULL, NPY_SHORT,
                                           boost::get<uint16_t*>(data));
          break;
        case 5:
          return PyArray_SimpleNewFromData(0, NULL, NPY_INT,
                                           boost::get<uint32_t*>(data));
          break;
        case 6:
          return PyArray_SimpleNewFromData(0, NULL, NPY_INT64,
                                           boost::get<uint64_t*>(data));
          break;
      }
    }
    const PyTypeObject* get_pytype() {
      return &PyArray_Type;
    }
  };
};

struct NdarrayCallPolicies : public bp::default_call_policies {
  typedef NdarrayConverterGenerator result_converter;
  PyObject* postcall(PyObject* pyargs, PyObject* result) {
    bp::object pyblob = bp::extract<bp::tuple>(pyargs)()[0];
    shared_ptr<BlobBase> blob =
        bp::extract<shared_ptr<BlobBase> >(pyblob);
    // Free the temporary pointer-holding array, and construct a new one with
    // the shape information from the blob.
    void* data = PyArray_DATA(reinterpret_cast<PyArrayObject*>(result));
    Py_DECREF(result);
    const int_tp num_axes = blob->num_axes();
    vector<npy_int_tp> dims(blob->shape().begin(), blob->shape().end());
    PyObject *arr_obj = PyArray_SimpleNewFromData(num_axes, dims.data(),
                                    proto_to_npy_type(blob->data_type()), data);
    // SetBaseObject steals a ref, so we need to INCREF.
    Py_INCREF(pyblob.ptr());
    PyArray_SetBaseObject(reinterpret_cast<PyArrayObject*>(arr_obj),
        pyblob.ptr());
    return arr_obj;
  }
};

bp::object Blob_Reshape(bp::tuple args, bp::dict kwargs) {
  if (bp::len(kwargs) > 0) {
    throw std::runtime_error("Blob.reshape takes no kwargs");
  }
  BlobBase* self = bp::extract<BlobBase*>(args[0]);
  vector<int_tp> shape(bp::len(args) - 1);
  for (int_tp i = 1; i < bp::len(args); ++i) {
    shape[i - 1] = bp::extract<int_tp>(args[i]);
  }
  self->Reshape(shape);
  // We need to explicitly return None to use bp::raw_function.
  return bp::object();
}

bp::object BlobVec_add_blob(bp::tuple args, bp::dict kwargs) {
  typedef vector<shared_ptr<BlobBase > > BlobVec;
  BlobVec* self = bp::extract<BlobVec*>(args[0]);
  vector<int_tp> shape(bp::len(args) - 1);
  for (int_tp i = 1; i < bp::len(args); ++i) {
    shape[i - 1] = bp::extract<int_tp>(args[i]);
  }
  shared_ptr<BlobBase> blob = CreateBlob(Caffe::GetDefaultDevice(),
                                 npy_to_proto_type(bp::extract<NPY_TYPES>(
                                     kwargs.get("dtype", NPY_FLOAT32))));
  blob->Reshape(shape);
  self->push_back(blob);

  // We need to explicitly return None to use bp::raw_function.
  return bp::object();
}

void exception_translator(std::exception ex) {
  std::cout << ex.what() << std::endl;
}

// NOLINT_NEXT_LINE(runtime/references)
variant_proto_types ForwardFromTo_NoGIL(NetBase* net, int_tp start,
                                        int_tp end) {
  variant_proto_types loss;
  Py_BEGIN_ALLOW_THREADS
  switch(net->data_type()) {
    case CAFFE_HALF:
#ifdef USE_HALF
      loss = static_cast<Net<half_fp>*>(net)->ForwardFromTo(start, end);
#endif  // USE_HALF
      break;
    case CAFFE_DOUBLE:
#ifdef USE_DOUBLE
      loss = static_cast<Net<double>*>(net)->ForwardFromTo(start, end);
#endif  // USE_DOUBLE
      break;
    case CAFFE_FLOAT:
    default:
#ifdef USE_SINGLE
      loss = static_cast<Net<float>*>(net)->ForwardFromTo(start, end);
#endif  // USE_SINGLE
      break;
  }
  Py_END_ALLOW_THREADS
  return loss;
}

// NOLINT_NEXT_LINE(runtime/references)
void BackwardFromTo_NoGIL(NetBase* net, int_tp start, int_tp end) {
  Py_BEGIN_ALLOW_THREADS
  switch(net->data_type()) {
    case CAFFE_HALF:
#ifdef USE_HALF
      static_cast<Net<half_fp>*>(net)->BackwardFromTo(start, end);
#endif  // USE_HALF
      break;
    case CAFFE_DOUBLE:
#ifdef USE_DOUBLE
      static_cast<Net<double>*>(net)->BackwardFromTo(start, end);
#endif  // USE_DOUBLE
      break;
    case CAFFE_FLOAT:
    default:
#ifdef USE_SINGLE
      static_cast<Net<float>*>(net)->BackwardFromTo(start, end);
#endif  // USE_SINGLE
      break;
  }
  Py_END_ALLOW_THREADS
}

// NOLINT_NEXT_LINE(runtime/references)
variant_proto_types Step_NoGIL(SolverBase* solver, int_tp iters) {
  variant_proto_types smoothed_loss;
  Py_BEGIN_ALLOW_THREADS
  switch(solver->data_type()) {
    case CAFFE_HALF:
#ifdef USE_HALF
      smoothed_loss = static_cast<Solver<half_fp>*>(solver)->Step(iters);
#endif  // USE_HALF
      break;
    case CAFFE_DOUBLE:
#ifdef USE_DOUBLE
      smoothed_loss = static_cast<Solver<double>*>(solver)->Step(iters);
#endif  // USE_DOUBLE
      break;
    case CAFFE_FLOAT:
    default:
#ifdef USE_SINGLE
      smoothed_loss = static_cast<Solver<float>*>(solver)->Step(iters);
#endif  // USE_SINGLE
      break;
  }
  Py_END_ALLOW_THREADS
  return smoothed_loss;
}

// NOLINT_NEXT_LINE(runtime/references)
void Solver_Solve_NoGIL(SolverBase* solver, const char* resume_file = nullptr) {
  Py_BEGIN_ALLOW_THREADS
  solver->Solve(resume_file);
  Py_END_ALLOW_THREADS
}
BOOST_PYTHON_FUNCTION_OVERLOADS(Solver_SolveOverloads,
                                       Solver_Solve_NoGIL, 1, 2);

class SolverCallback: public SolverBase::Callback {
 protected:
  bp::object on_start_, on_gradients_ready_;

 public:
  SolverCallback(bp::object on_start, bp::object on_gradients_ready)
    : on_start_(on_start), on_gradients_ready_(on_gradients_ready) { }
  virtual void on_gradients_ready() {
    on_gradients_ready_();
  }
  virtual void on_start() {
    on_start_();
  }
};

void Solver_add_callback(SolverBase* solver, bp::object on_start,
  bp::object on_gradients_ready) {
      solver->add_callback(
          new SolverCallback(on_start, on_gradients_ready));
}

// Seems boost cannot call the base method directly
void Solver_add_nccl(SolverBase* solver
#ifdef USE_NCCL
  , NCCL<Dtype>* nccl
#endif
) {
#ifdef USE_NCCL
  solver->add_callback(nccl);
#endif
}

void share_weights(SolverBase* solver, NetBase* net) {
  net->ShareTrainedLayersWith(solver->net_base().get());
}

template<typename Dtype>
class NetCallback: public NetBase::Callback {
 public:
  explicit NetCallback(bp::object run) : run_(run) {}

 protected:
  virtual void run(int layer) {
    run_(layer);
  }
  bp::object run_;
};
void Net_before_forward(NetBase* net, bp::object run) {
  switch(net->data_type()) {
    case CAFFE_HALF:
#ifdef USE_HALF
      static_cast<Net<half_fp>*>(net)->add_before_forward(
          new NetCallback<half_fp>(run));
#endif  // USE_HALF
      break;
    case CAFFE_DOUBLE:
#ifdef USE_DOUBLE
      static_cast<Net<double>*>(net)->add_before_forward(
          new NetCallback<double>(run));
#endif  // USE_DOUBLE
      break;
    case CAFFE_FLOAT:
    default:
#ifdef USE_SINGLE
      static_cast<Net<float>*>(net)->add_before_forward(
          new NetCallback<float>(run));
#endif  // USE_SINGLE
      break;
  }
}
void Net_after_forward(NetBase* net, bp::object run) {
  switch(net->data_type()) {
    case CAFFE_HALF:
#ifdef USE_HALF
      static_cast<Net<half_fp>*>(net)->add_after_forward(
          new NetCallback<half_fp>(run));
#endif  // USE_HALF
      break;
    case CAFFE_DOUBLE:
#ifdef USE_DOUBLE
      static_cast<Net<double>*>(net)->add_after_forward(
          new NetCallback<double>(run));
#endif  // USE_DOUBLE
      break;
    case CAFFE_FLOAT:
    default:
#ifdef USE_SINGLE
      static_cast<Net<float>*>(net)->add_after_forward(
          new NetCallback<float>(run));
#endif  // USE_SINGLE
      break;
  }
}
void Net_before_backward(NetBase* net, bp::object run) {
  switch(net->data_type()) {
    case CAFFE_HALF:
#ifdef USE_HALF
      static_cast<Net<half_fp>*>(net)->add_before_backward(
          new NetCallback<half_fp>(run));
#endif  // USE_HALF
      break;
    case CAFFE_DOUBLE:
#ifdef USE_DOUBLE
      static_cast<Net<double>*>(net)->add_before_backward(
          new NetCallback<double>(run));
#endif  // USE_DOUBLE
      break;
    case CAFFE_FLOAT:
    default:
#ifdef USE_SINGLE
      static_cast<Net<float>*>(net)->add_before_backward(
          new NetCallback<float>(run));
#endif  // USE_SINGLE
      break;
  }
}
void Net_after_backward(NetBase* net, bp::object run) {
  switch(net->data_type()) {
    case CAFFE_HALF:
#ifdef USE_HALF
      static_cast<Net<half_fp>*>(net)->add_after_backward(
          new NetCallback<half_fp>(run));
#endif  // USE_HALF
      break;
    case CAFFE_DOUBLE:
#ifdef USE_DOUBLE
      static_cast<Net<double>*>(net)->add_after_backward(
          new NetCallback<double>(run));
#endif  // USE_DOUBLE
      break;
    case CAFFE_FLOAT:
    default:
#ifdef USE_SINGLE
      static_cast<Net<float>*>(net)->add_after_backward(
          new NetCallback<float>(run));
#endif  // USE_SINGLE
      break;
  }
}

void Net_add_nccl(NetBase* net
#ifdef USE_NCCL
  , NCCL<Dtype>* nccl
#endif
) {
#ifdef USE_NCCL
  net->add_after_backward(nccl);
#endif
}
#ifndef USE_NCCL
template<typename Dtype>
class NCCL {
 public:
  NCCL(shared_ptr<SolverBase > solver, const string& uid) {}
};
#endif

bool HasNCCL() {
#ifdef USE_NCCL
  return true;
#else
  return false;
#endif
}

#ifdef USE_NCCL
bp::object NCCL_New_Uid() {
  string uid = NCCL<Dtype>::new_uid();
#if PY_MAJOR_VERSION >= 3
  // Convert string to bytes so that Python does not
  // try to decode the string using the current locale.

  // Since boost 1.53 boost.python will convert str and bytes
  // to string but will convert string to str. Here we
  // force a bytes object to be returned. When this object
  // is passed back to the NCCL constructor boost.python will
  // correctly convert the bytes to string automatically
  PyObject* py_uid = PyBytes_FromString(uid.c_str());
  return bp::object(bp::handle<>(py_uid));
#else
  // automatic conversion is correct for python 2.
  return bp::object(uid);
#endif
}
#endif


variant_proto_ptr_types Blob_mutable_cpu_data(BlobBase* blob) {
  switch(blob->data_type()) {
    case CAFFE_HALF:
#ifdef USE_HALF
      return static_cast<Blob<half_fp>*>(blob)->mutable_cpu_data();
#endif  // USE_HALF
    case CAFFE_DOUBLE:
#ifdef USE_HALF
      return static_cast<Blob<double>*>(blob)->mutable_cpu_data();
#endif  // USE_HALF
    case CAFFE_INT8_QUANTIZED:
#ifdef USE_HALF
      return static_cast<Blob<uint8_t>*>(blob)->mutable_cpu_data();
#endif  // USE_HALF
    case CAFFE_INT16_QUANTIZED:
#ifdef USE_HALF
      return static_cast<Blob<uint16_t>*>(blob)->mutable_cpu_data();
#endif  // USE_HALF
    case CAFFE_INT32_QUANTIZED:
#ifdef USE_HALF
      return static_cast<Blob<uint32_t>*>(blob)->mutable_cpu_data();
#endif  // USE_HALF
    case CAFFE_INT64_QUANTIZED:
#ifdef USE_HALF
      return static_cast<Blob<uint64_t>*>(blob)->mutable_cpu_data();
#endif  // USE_HALF
    case CAFFE_FLOAT:
    default:
#ifdef USE_HALF
      return static_cast<Blob<float>*>(blob)->mutable_cpu_data();
#endif  // USE_HALF
  }
}

variant_proto_ptr_types Blob_mutable_cpu_diff(BlobBase* blob) {
  switch(blob->data_type()) {
    case CAFFE_HALF:
#ifdef USE_HALF
      return static_cast<Blob<half_fp>*>(blob)->mutable_cpu_diff();
#endif  // USE_HALF
    case CAFFE_DOUBLE:
#ifdef USE_DOUBLE
      return static_cast<Blob<double>*>(blob)->mutable_cpu_diff();
#endif  // USE_DOUBLE
    case CAFFE_INT8_QUANTIZED:
#ifdef USE_INT_QUANT_8
      return static_cast<Blob<uint8_t>*>(blob)->mutable_cpu_diff();
#endif  // USE_INT_QUANT_8
    case CAFFE_INT16_QUANTIZED:
#ifdef USE_INT_QUANT_16
      return static_cast<Blob<uint16_t>*>(blob)->mutable_cpu_diff();
#endif  // USE_INT_QUANT_16
    case CAFFE_INT32_QUANTIZED:
#ifdef USE_INT_QUANT_32
      return static_cast<Blob<uint32_t>*>(blob)->mutable_cpu_diff();
#endif  // USE_INT_QUANT_32
    case CAFFE_INT64_QUANTIZED:
#ifdef USE_INT_QUANT_64
      return static_cast<Blob<uint64_t>*>(blob)->mutable_cpu_diff();
#endif  // USE_INT_QUANT_64
    case CAFFE_FLOAT:
    default:
#ifdef USE_SINGLE
      return static_cast<Blob<float>*>(blob)->mutable_cpu_diff();
#endif  // USE_SINGLE
  }
}


BOOST_PYTHON_MODULE(_caffe) {
  bp::register_exception_translator<std::exception>(&exception_translator);

  // below, we prepend an underscore to methods that will be replaced
  // in Python

  bp::scope().attr("__version__") = AS_STRING(CAFFE_VERSION);

  // Boost variants
  bp::to_python_converter<variant_proto_types,
                          variant_proto_types_to_object>();
  bp::to_python_converter<variant_proto_ptr_types,
                          variant_proto_ptr_types_to_object>();
  bp::to_python_converter<variant_proto_vec_types,
                          variant_proto_vec_types_to_object>();

  bp::implicitly_convertible<half_fp, variant_proto_types>();
  bp::implicitly_convertible<float, variant_proto_types>();
  bp::implicitly_convertible<double, variant_proto_types>();
  bp::implicitly_convertible<uint8_t, variant_proto_types>();
  bp::implicitly_convertible<uint16_t, variant_proto_types>();
  bp::implicitly_convertible<uint32_t, variant_proto_types>();
  bp::implicitly_convertible<uint64_t, variant_proto_types>();

  bp::implicitly_convertible<half_fp*, variant_proto_ptr_types>();
  bp::implicitly_convertible<float*, variant_proto_ptr_types>();
  bp::implicitly_convertible<double*, variant_proto_ptr_types>();
  bp::implicitly_convertible<uint8_t*, variant_proto_ptr_types>();
  bp::implicitly_convertible<uint16_t*, variant_proto_ptr_types>();
  bp::implicitly_convertible<uint32_t*, variant_proto_ptr_types>();
  bp::implicitly_convertible<uint64_t*, variant_proto_ptr_types>();

  bp::implicitly_convertible<vector<half_fp>, variant_proto_vec_types>();
  bp::implicitly_convertible<vector<float>, variant_proto_vec_types>();
  bp::implicitly_convertible<vector<double>, variant_proto_vec_types>();
  bp::implicitly_convertible<vector<uint8_t>, variant_proto_vec_types>();
  bp::implicitly_convertible<vector<uint16_t>, variant_proto_vec_types>();
  bp::implicitly_convertible<vector<uint32_t>, variant_proto_vec_types>();
  bp::implicitly_convertible<vector<uint64_t>, variant_proto_vec_types>();


  // Caffe utility functions
  bp::def("init_log", &InitLog);
  bp::def("init_log", &InitLogLevel);
  #ifndef _MSC_VER
  bp::def("init_log", &InitLogLevelPipe);
  #endif  // _MSC_VER
  bp::def("log", &Log);
  bp::def("has_nccl", &HasNCCL);
  bp::def("set_mode_cpu", &set_mode_cpu);
  bp::def("set_mode_gpu", &set_mode_gpu);
  bp::def("set_random_seed", &set_random_seed);
  bp::def("set_device", &Caffe::SetDevice);
  bp::def("set_devices", &set_devices);
  bp::def("select_device", &select_device);
  bp::def("enumerate_devices", &Caffe::EnumerateDevices);
  bp::def("solver_count", &Caffe::solver_count);
  bp::def("set_solver_count", &Caffe::set_solver_count);
  bp::def("solver_rank", &Caffe::solver_rank);
  bp::def("set_solver_rank", &Caffe::set_solver_rank);
  bp::def("set_multiprocess", &Caffe::set_multiprocess);
  // TODO: Temporary fix, add for all possible types later on
  bp::def("layer_type_list",
          &LayerRegistry<float, float, float>::LayerTypeList);

  bp::class_<NetBase, shared_ptr<NetBase>, boost::noncopyable >("Net",
    bp::no_init)
    // Constructor
    .def("__init__", bp::make_constructor(&Net_Init,
          bp::default_call_policies(), (bp::arg("network_file"), "phase",
            bp::arg("level")=0, bp::arg("stages")=bp::object(),
            bp::arg("weights")=bp::object())))
    .def("_forward", &ForwardFromTo_NoGIL)
    .def("_backward", &BackwardFromTo_NoGIL)
    .def("reshape", &NetBase::Reshape)
    .add_property("quant_mode", &NetBase::quant_mode, &NetBase::set_quant_mode)
    .def("clear_param_diffs", &NetBase::ClearParamDiffs)
    // The cast is to select a particular overload.
    .def("copy_from", static_cast<void (NetBase::*)(const string)>(
        &NetBase::CopyTrainedLayersFrom))
    .def("share_with", &NetBase::ShareTrainedLayersWith)
    .add_property("_blob_loss_weights", &Net_get_blob_loss_weights)
    .def("_bottom_ids", bp::make_function(&NetBase::bottom_ids,
        bp::return_value_policy<bp::copy_const_reference>()))
    .def("_top_ids", bp::make_function(&NetBase::top_ids,
        bp::return_value_policy<bp::copy_const_reference>()))
    .add_property("_blobs", bp::make_function(&NetBase::blobs,
        bp::return_internal_reference<>()))
    .add_property("layers", bp::make_function(&NetBase::layers,
        bp::return_internal_reference<>()))
    .add_property("_blob_names", bp::make_function(&NetBase::blob_names,
        bp::return_value_policy<bp::copy_const_reference>()))
    .add_property("_layer_names", bp::make_function(&NetBase::layer_names,
        bp::return_value_policy<bp::copy_const_reference>()))
    .add_property("_inputs", bp::make_function(&NetBase::input_blob_indices,
        bp::return_value_policy<bp::copy_const_reference>()))
    .add_property("_outputs",
        bp::make_function(&NetBase::output_blob_indices,
        bp::return_value_policy<bp::copy_const_reference>()))
    .def("_set_input_arrays", &Net_SetInputArrays,
        bp::with_custodian_and_ward<1, 3,
        bp::with_custodian_and_ward<1, 4> > ())
    .def("_set_layer_input_arrays", &Net_SetLayerInputArrays,
        bp::with_custodian_and_ward<1, 3,
        bp::with_custodian_and_ward<1, 4> > ())
    .def("save", &Net_Save, Net_SaveOverloads())
    .def("save_hdf5", &Net_SaveHDF5)
    .def("load_hdf5", &Net_LoadHDF5)
    .def("before_forward", &Net_before_forward)
    .def("after_forward", &Net_after_forward)
    .def("before_backward", &Net_before_backward)
    .def("after_backward", &Net_after_backward)
    .def("after_backward", &Net_add_nccl);
  BP_REGISTER_SHARED_PTR_TO_PYTHON_NO_TEMPLATE(NetBase);

  bp::class_<BlobBase, shared_ptr<BlobBase >, boost::noncopyable>(
    "Blob", bp::no_init)
    .add_property("shape",
        bp::make_function(
            static_cast<const vector<int_tp>& (BlobBase::*)() const>(
                &BlobBase::shape),
            bp::return_value_policy<bp::copy_const_reference>()))
    .add_property("num",      &BlobBase::num)
    .add_property("channels", &BlobBase::channels)
    .add_property("height",   &BlobBase::height)
    .add_property("width",    &BlobBase::width)
    .add_property("count",    static_cast<int_tp (BlobBase::*)() const>(
        &BlobBase::count))
    .def("reshape",           bp::raw_function(&Blob_Reshape))
#ifndef CPU_ONLY
    // FIXME
/*    .add_property("_gpu_data_ptr",
        reinterpret_cast<uintptr_t (BlobBase::*)()>(
          &BlobBase::mutable_gpu_data))
    .add_property("_gpu_diff_ptr",
        reinterpret_cast<uintptr_t (BlobBase::*)()>(
          &BlobBase::mutable_gpu_diff))*/
#endif
    .add_property("data", bp::make_function(&Blob_mutable_cpu_data,
                                            NdarrayCallPolicies()))
    .add_property("diff", bp::make_function(&Blob_mutable_cpu_diff,
                                            NdarrayCallPolicies()));
  BP_REGISTER_SHARED_PTR_TO_PYTHON_NO_TEMPLATE(BlobBase);

  bp::class_<LayerBase,
              shared_ptr<PythonLayer<float, float, float> >,
    boost::noncopyable>("Layer", bp::init<const LayerParameter&>())
    .add_property("blobs", &LayerBase::blob_bases)
    .def("setup", &LayerBase::LayerSetUp)
    .def("reshape", static_cast<void (LayerBase::*)
         (const vector<BlobBase*>& bottom, const vector<BlobBase*>& top)>
         (&LayerBase::Reshape))
    .add_property("type", bp::make_function(&LayerBase::type))
    .add_property("layer_param",
                  bp::make_function(&LayerBase::layer_param,
                  bp::return_internal_reference<>()));
  BP_REGISTER_SHARED_PTR_TO_PYTHON_NO_TEMPLATE(LayerBase);

  bp::class_<LayerParameter>("LayerParameter", bp::no_init)
    .add_property("name",          bp::make_function(
                          static_cast<const string& (LayerParameter::*)
                          (void) const>(&LayerParameter::name),
                          bp::return_value_policy<bp::return_by_value>()))
    .add_property("bottom_size",   &LayerParameter::bottom_size)
    .def("bottom",    bp::make_function(
                          static_cast<const string& (LayerParameter::*)
                          (int) const>(&LayerParameter::bottom),  // NOLINT
                          bp::return_value_policy<bp::return_by_value>()))
    .add_property("top_size",      &LayerParameter::top_size)
    .def("top",       bp::make_function(
                          static_cast<const string& (LayerParameter::*)
                          (int) const>(&LayerParameter::top),     // NOLINT
                          bp::return_value_policy<bp::return_by_value>()));

  bp::class_<SolverParameter>("SolverParameter", bp::no_init)
    .add_property("max_iter", &SolverParameter::max_iter)
    .add_property("display", &SolverParameter::display)
    .add_property("layer_wise_reduce", &SolverParameter::layer_wise_reduce)
    .add_property("base_lr", &SolverParameter::base_lr,
           &SolverParameter::set_base_lr);

  bp::class_<SolverBase, shared_ptr<SolverBase>, boost::noncopyable>(
    "Solver", bp::no_init)
    .add_property("net", &SolverBase::net_base)
    .add_property("max_iter", &SolverBase::max_iter)
    .add_property("test_nets", &SolverBase::test_nets_bases)
    .add_property("iter", &SolverBase::iter)
    .add_property("param", bp::make_function(&SolverBase::param,
                      bp::return_value_policy<bp::copy_const_reference>()),
                      &SolverBase::update_solver_param)
    .def("step", &Step_NoGIL)
    .def("add_callback", &Solver_add_callback)
    .def("add_callback", &Solver_add_nccl)
    .def("solve", &Solver_Solve_NoGIL, Solver_SolveOverloads())
    .def("restore", &SolverBase::Restore)
    .def("snapshot", &SolverBase::Snapshot)
    .def("share_weights", &share_weights)
    .def("apply_update", &SolverBase::ApplyUpdate)
    .add_property("param", bp::make_function(&SolverBase::param,
                  bp::return_internal_reference<>()));
  BP_REGISTER_SHARED_PTR_TO_PYTHON_NO_TEMPLATE(SolverBase);

  bp::class_<NetState>("NetState", bp::init<>())
    .add_property("phase", &NetState::phase,
                           &NetState::set_phase)
    .add_property("level", &NetState::level,
                           &NetState::set_level)
    .def("stage_size",     &NetState::stage_size)
    .def("get_stage",      bp::make_function(
                           static_cast<const string& (NetState::*)
                           (int) const>(&NetState::stage),  // NOLINT
                           bp::return_value_policy<bp::return_by_value>()))
    .def("add_stage",      static_cast<void (NetState::*)   // NOLINT
                           (const string&)>(&NetState::add_stage))
    .def("set_stage",      static_cast<void (NetState::*)   // NOLINT
                           (int, const string&)>(&NetState::set_stage))
    .def("clear_stage",    &NetState::clear_stage);

  bp::class_<NetParameter>("NetParameter", bp::init<>())
    .add_property("force_backward", &NetParameter::force_backward,
                                    &NetParameter::set_force_backward)
    .add_property("state",
                     bp::make_function(&NetParameter::state,
                     bp::return_value_policy<bp::copy_const_reference>()),
                     static_cast<void (NetParameter::*)(NetState*)>(
                             &NetParameter::set_allocated_state));


  bp::class_<SolverParameter>("SolverParameter", bp::init<>())
    .add_property("base_lr",   &SolverParameter::base_lr,
                               &SolverParameter::set_base_lr)
    .add_property("max_iter",  &SolverParameter::max_iter,
                               &SolverParameter::set_max_iter)
    .add_property("lr_policy",
                      bp::make_function(&SolverParameter::lr_policy,
                      bp::return_value_policy<bp::copy_const_reference>()),
                      static_cast<void (SolverParameter::*)(const char*)>(
                               &SolverParameter::set_lr_policy))
    .add_property("gamma",     &SolverParameter::gamma,
                               &SolverParameter::set_gamma)
    .add_property("power",     &SolverParameter::power,
                               &SolverParameter::set_power)
    .add_property("momentum",  &SolverParameter::momentum,
                               &SolverParameter::set_momentum)
    .add_property("momentum2", &SolverParameter::momentum2,
                               &SolverParameter::set_momentum2)
    .add_property("delta",     &SolverParameter::delta,
                               &SolverParameter::set_delta)
    .add_property("rms_decay", &SolverParameter::rms_decay,
                               &SolverParameter::set_rms_decay)
    .add_property("weight_decay",
                               &SolverParameter::weight_decay,
                               &SolverParameter::set_weight_decay)
    .add_property("display",   &SolverParameter::display,
                               &SolverParameter::set_display)
    .add_property("regularization_type",
                       bp::make_function(&SolverParameter::regularization_type,
                       bp::return_value_policy<bp::copy_const_reference>()),
                       static_cast<void (SolverParameter::*)(const string&)>(
                               &SolverParameter::set_regularization_type))
    .add_property("stepsize",  &SolverParameter::stepsize,
                               &SolverParameter::set_stepsize)
    .add_property("snapshot",  &SolverParameter::snapshot,
                               &SolverParameter::set_snapshot)
    .add_property("snapshot_format", &SolverParameter::snapshot_format,
                                     &SolverParameter::set_snapshot_format)
    .add_property("snapshot_prefix",
                   bp::make_function(&SolverParameter::snapshot_prefix,
                   bp::return_value_policy<bp::copy_const_reference>()),
                   static_cast<void (SolverParameter::*)(const string&)>(
                           &SolverParameter::set_snapshot_prefix))
    .add_property("type",
                   bp::make_function(&SolverParameter::type,
                   bp::return_value_policy<bp::copy_const_reference>()),
                   static_cast<void (SolverParameter::*)(const string&)>(
                           &SolverParameter::set_type))
    .add_property("net",
                   bp::make_function(&SolverParameter::net,
                   bp::return_value_policy<bp::copy_const_reference>()),
                   static_cast<void (SolverParameter::*)(const string&)>(
                           &SolverParameter::set_net))
    .add_property("train_net",
                   bp::make_function(&SolverParameter::train_net,
                   bp::return_value_policy<bp::copy_const_reference>()),
                   static_cast<void (SolverParameter::*)(const string&)>(
                           &SolverParameter::set_train_net))
    .add_property("net_param",
                   bp::make_function(&SolverParameter::mutable_net_param,
                   bp::return_value_policy<bp::reference_existing_object>()),
                   static_cast<void (SolverParameter::*)(NetParameter*)>(
                           &SolverParameter::set_allocated_net_param))
    .add_property("train_state",
                   bp::make_function(&SolverParameter::mutable_train_state,
                   bp::return_value_policy<bp::reference_existing_object>()),
                   static_cast<void (SolverParameter::*)(NetState*)>(
                           &SolverParameter::set_allocated_train_state));

  bp::enum_<::caffe::SolverParameter_SnapshotFormat>("snapshot_format")
      .value("HDF5", SolverParameter_SnapshotFormat_HDF5)
      .value("BINARYPROTO", SolverParameter_SnapshotFormat_BINARYPROTO);

  bp::enum_<::caffe::DataType>("data_type")
      .value("CAFFE_HALF", CAFFE_HALF)
      .value("CAFFE_FLOAT", CAFFE_FLOAT)
      .value("CAFFE_DOUBLE", CAFFE_DOUBLE)
      .value("CAFFE_INT8_QUANTIZED", CAFFE_INT8_QUANTIZED)
      .value("CAFFE_INT16_QUANTIZED", CAFFE_INT16_QUANTIZED)
      .value("CAFFE_INT32_QUANTIZED", CAFFE_INT32_QUANTIZED)
      .value("CAFFE_INT64_QUANTIZED", CAFFE_INT64_QUANTIZED);


  bp::enum_<::caffe::QuantizerMode>("quantizer_mode")
      .value("CAFFE_QUANT_PASSIVE", CAFFE_QUANT_PASSIVE)
      .value("CAFFE_QUANT_OBSERVE", CAFFE_QUANT_OBSERVE);

#define REGISTER_SOLVERS_TO_PYTHON(Dtype, Name) \
  bp::class_<Solver<Dtype>, bp::bases<SolverBase>, \
    shared_ptr<Solver<Dtype> >, boost::noncopyable>( \
        "Solver" Name, bp::no_init); \
  bp::class_<SGDSolver<Dtype>, bp::bases<Solver<Dtype> >, \
    shared_ptr<SGDSolver<Dtype> >, boost::noncopyable>( \
        "SGDSolver" Name, bp::init<string, Device*>()); \
  bp::class_<NesterovSolver<Dtype>, bp::bases<Solver<Dtype> >, \
    shared_ptr<NesterovSolver<Dtype> >, boost::noncopyable>( \
        "NesterovSolver" Name, bp::init<string, Device*>()); \
  bp::class_<AdaGradSolver<Dtype>, bp::bases<Solver<Dtype> >, \
    shared_ptr<AdaGradSolver<Dtype> >, boost::noncopyable>( \
        "AdaGradSolver" Name, bp::init<string, Device*>()); \
  bp::class_<RMSPropSolver<Dtype>, bp::bases<Solver<Dtype> >, \
    shared_ptr<RMSPropSolver<Dtype> >, boost::noncopyable>( \
        "RMSPropSolver" Name, bp::init<string, Device*>()); \
  bp::class_<AdaDeltaSolver<Dtype>, bp::bases<Solver<Dtype> >, \
    shared_ptr<AdaDeltaSolver<Dtype> >, boost::noncopyable>( \
        "AdaDeltaSolver" Name, bp::init<string, Device*>()); \
  bp::class_<AdamSolver<Dtype>, bp::bases<Solver<Dtype> >, \
    shared_ptr<AdamSolver<Dtype> >, boost::noncopyable>( \
        "AdamSolver" Name, bp::init<string, Device*>());

#ifdef USE_HALF
  REGISTER_SOLVERS_TO_PYTHON(half_fp, "_half");
#endif  // USE_HALF
#ifdef USE_SINGLE
  REGISTER_SOLVERS_TO_PYTHON(float, "_float");
#endif  // USE_SINGLE
#ifdef USE_DOUBLE
  REGISTER_SOLVERS_TO_PYTHON(double, "_double");
#endif  // USE_DOUBLE


  bp::def("get_solver_from_file", &GetSolverFromFile,
      bp::return_value_policy<bp::manage_new_object>());

  bp::def("get_solver", &GetSolver,
      bp::return_value_policy<bp::manage_new_object>());

  // vector wrappers for all the vector types we use
  bp::class_<vector<shared_ptr<BlobBase > > >("BlobVec")
    .def(bp::vector_indexing_suite<vector<shared_ptr<BlobBase > >, true>())
    .def("add_blob", bp::raw_function(&BlobVec_add_blob));
  bp::class_<vector<BlobBase*> >("RawBlobVec")
    .def(bp::vector_indexing_suite<vector<BlobBase*>, true>());
  bp::class_<vector<shared_ptr<LayerBase> > > ("LayerVec")
    .def(bp::vector_indexing_suite<vector<shared_ptr<LayerBase> > , true>());
  bp::class_<vector<string> >("StringVec")
    .def(bp::vector_indexing_suite<vector<string> >());
  bp::class_<vector<int_tp> >("IntTpVec")
    .def(bp::vector_indexing_suite<vector<int_tp> >());
  bp::class_<vector<int> >("IntVec")
    .def(bp::vector_indexing_suite<vector<int> >());

#define REGISTER_DTYPE_VECTORS_TO_PYTHON(Dtype, Name) \
  bp::class_<vector<Dtype> >("DtypeVec" Name) \
    .def(bp::vector_indexing_suite<vector<Dtype> >());

  REGISTER_DTYPE_VECTORS_TO_PYTHON(half_fp, "_half");
  REGISTER_DTYPE_VECTORS_TO_PYTHON(float, "_float");
  REGISTER_DTYPE_VECTORS_TO_PYTHON(double, "_double");
  REGISTER_DTYPE_VECTORS_TO_PYTHON(uint8_t, "_uint8");
  REGISTER_DTYPE_VECTORS_TO_PYTHON(uint16_t, "_uint16");
  REGISTER_DTYPE_VECTORS_TO_PYTHON(uint32_t, "_uint32");
  REGISTER_DTYPE_VECTORS_TO_PYTHON(uint64_t, "_uint64");

  bp::class_<vector<shared_ptr<NetBase> > >("NetVec")
    .def(bp::vector_indexing_suite<vector<shared_ptr<NetBase> >, true>());
  bp::class_<vector<bool> >("BoolVec")
    .def(bp::vector_indexing_suite<vector<bool> >());

  bp::class_<NCCL<float>, shared_ptr<NCCL<float> >,
    boost::noncopyable>("NCCL",
                        bp::init<shared_ptr<Solver<float> >, const string&>())
#ifdef USE_NCCL
    .def("new_uid", NCCL_New_Uid).staticmethod("new_uid")
    .def("bcast", &NCCL<Dtype>::Broadcast)
#endif
    /* NOLINT_NEXT_LINE(whitespace/semicolon) */
  ;
#ifdef USE_NCCL
  BP_REGISTER_SHARED_PTR_TO_PYTHON(NCCL, (Dtype));
#endif  // USE_NCCL

  bp::class_<Timer, shared_ptr<Timer>, boost::noncopyable>(
    "Timer", bp::init<>())
    .def("start", &Timer::Start)
    .def("stop", &Timer::Stop)
    .add_property("ms", &Timer::MilliSeconds);
  BP_REGISTER_SHARED_PTR_TO_PYTHON_NO_TEMPLATE(Timer);

  // boost python expects a void (missing) return value, while import_array
  // returns NULL for python3. import_array1() forces a void return value.
  import_array1();
}

}  // namespace caffe
