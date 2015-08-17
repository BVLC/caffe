#include <Python.h>  // NOLINT(build/include_alpha)

// Produce deprecation warnings (needs to come before arrayobject.h inclusion).
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <boost/make_shared.hpp>
#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/slice.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <numpy/arrayobject.h>

// these need to be included after boost on OS X
#include <algorithm> // NOLINT
#include <string>  // NOLINT(build/include_order)
#include <vector>  // NOLINT(build/include_order)
#include <fstream>  // NOLINT

#include "caffe/array/array.hpp"
#include "caffe/caffe.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/layers/python_layer.hpp"
#include "caffe/sgd_solvers.hpp"

// Temporary solution for numpy < 1.7 versions: old macro, no promises.
// You're strongly advised to upgrade to >= 1.7.
#ifndef NPY_ARRAY_C_CONTIGUOUS
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#define PyArray_SetBaseObject(arr, x) (PyArray_BASE(arr) = (x))
#endif

/* Fix to avoid registration warnings in pycaffe (#3960) */
#define BP_REGISTER_SHARED_PTR_TO_PYTHON(PTR) do { \
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

#if PY_MAJOR_VERSION < 3
#define PyLong_AS_LONG(val) PyInt_AS_LONG(val)
#define PyLong_AsLong(val) PyInt_AsLong(val)
#define PyLong_Check(val) PyInt_Check(val)
#endif

namespace bp = boost::python;

// Define the ellipsis object in boost python
namespace boost { namespace python {
struct ellipsis : public object {
  BOOST_PYTHON_FORWARD_OBJECT_CONSTRUCTORS(ellipsis, object)
};
// bp::numeric::array leads to unexplained segfaults at program exit, instead
// we wrap the numpy array in ndarray
struct ndarray : public object {
  BOOST_PYTHON_FORWARD_OBJECT_CONSTRUCTORS(ndarray, object)
};
namespace converter {
template<> struct object_manager_traits<ellipsis>
  : pytype_object_manager_traits<&PyEllipsis_Type, slice> {};
template <> struct object_manager_traits<ndarray> {
  BOOST_STATIC_CONSTANT(bool, is_specialized = true);
  static bool check(PyObject* x) { return PyArray_Check(x); }
};
}  // namespace converter
}  // namespace python
}  // namespace boost

namespace caffe {

// For Python, for now, we'll just always use float as the type.
typedef float Dtype;
const int NPY_DTYPE = NPY_FLOAT32;

// Selecting mode.
void set_mode_cpu() { Caffe::set_mode(Caffe::CPU); }
void set_mode_gpu() { Caffe::set_mode(Caffe::GPU); }

void InitLog(int level) {
  FLAGS_logtostderr = 1;
  FLAGS_minloglevel = level;
  ::google::InitGoogleLogging("");
  ::google::InstallFailureSignalHandler();
}
void InitLogInfo() {
  InitLog(google::INFO);
}
void Log(const string& s) {
  LOG(INFO) << s;
}

void set_random_seed(unsigned int seed) { Caffe::set_random_seed(seed); }

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

// Net constructor
shared_ptr<Net<Dtype> > Net_Init(string network_file, int phase,
    const int level, const bp::object& stages,
    const bp::object& weights) {
  CheckFile(network_file);

  // Convert stages from list to vector
  vector<string> stages_vector;
  if (!stages.is_none()) {
    for (int i = 0; i < len(stages); i++) {
      stages_vector.push_back(bp::extract<string>(stages[i]));
    }
  }

  // Initialize net
  shared_ptr<Net<Dtype> > net(new Net<Dtype>(network_file,
        static_cast<Phase>(phase), level, &stages_vector));

  // Load weights
  if (!weights.is_none()) {
    std::string weights_file_str = bp::extract<std::string>(weights);
    CheckFile(weights_file_str);
    net->CopyTrainedLayersFrom(weights_file_str);
  }

  return net;
}

// Legacy Net construct-and-load convenience constructor
shared_ptr<Net<Dtype> > Net_Init_Load(
    string param_file, string pretrained_param_file, int phase) {
  LOG(WARNING) << "DEPRECATION WARNING - deprecated use of Python interface";
  LOG(WARNING) << "Use this instead (with the named \"weights\""
    << " parameter):";
  LOG(WARNING) << "Net('" << param_file << "', " << phase
    << ", weights='" << pretrained_param_file << "')";
  CheckFile(param_file);
  CheckFile(pretrained_param_file);

  shared_ptr<Net<Dtype> > net(new Net<Dtype>(param_file,
      static_cast<Phase>(phase)));
  net->CopyTrainedLayersFrom(pretrained_param_file);
  return net;
}

void Net_Save(const Net<Dtype>& net, string filename) {
  NetParameter net_param;
  net.ToProto(&net_param, false);
  WriteProtoToBinaryFile(net_param, filename.c_str());
}

void Net_SaveHDF5(const Net<Dtype>& net, string filename) {
  net.ToHDF5(filename);
}

void Net_LoadHDF5(Net<Dtype>* net, string filename) {
  net->CopyTrainedLayersFromHDF5(filename.c_str());
}

void Net_SetInputArrays(Net<Dtype>* net, bp::object data_obj,
    bp::object labels_obj) {
  // check that this network has an input MemoryDataLayer
  shared_ptr<MemoryDataLayer<Dtype> > md_layer =
    boost::dynamic_pointer_cast<MemoryDataLayer<Dtype> >(net->layers()[0]);
  if (!md_layer) {
    throw std::runtime_error("set_input_arrays may only be called if the"
        " first layer is a MemoryDataLayer");
  }

  // check that we were passed appropriately-sized contiguous memory
  PyArrayObject* data_arr =
      reinterpret_cast<PyArrayObject*>(data_obj.ptr());
  PyArrayObject* labels_arr =
      reinterpret_cast<PyArrayObject*>(labels_obj.ptr());
  CheckContiguousArray(data_arr, "data array", md_layer->channels(),
      md_layer->height(), md_layer->width());
  CheckContiguousArray(labels_arr, "labels array", 1, 1, 1);
  if (PyArray_DIMS(data_arr)[0] != PyArray_DIMS(labels_arr)[0]) {
    throw std::runtime_error("data and labels must have the same first"
        " dimension");
  }
  if (PyArray_DIMS(data_arr)[0] % md_layer->batch_size() != 0) {
    throw std::runtime_error("first dimensions of input arrays must be a"
        " multiple of batch size");
  }

  md_layer->Reset(static_cast<Dtype*>(PyArray_DATA(data_arr)),
      static_cast<Dtype*>(PyArray_DATA(labels_arr)),
      PyArray_DIMS(data_arr)[0]);
}

Solver<Dtype>* GetSolverFromFile(const string& filename) {
  SolverParameter param;
  ReadSolverParamsFromTextFileOrDie(filename, &param);
  return SolverRegistry<Dtype>::CreateSolver(param);
}

struct NdarrayConverterGenerator {
  template <typename T> struct apply;
};

template <>
struct NdarrayConverterGenerator::apply<Dtype*> {
  struct type {
    PyObject* operator() (Dtype* data) const {
      // Just store the data pointer, and add the shape information in postcall.
      return PyArray_SimpleNewFromData(0, NULL, NPY_DTYPE, data);
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
    shared_ptr<Blob<Dtype> > blob =
      bp::extract<shared_ptr<Blob<Dtype> > >(pyblob);
    // Free the temporary pointer-holding array, and construct a new one with
    // the shape information from the blob.
    void* data = PyArray_DATA(reinterpret_cast<PyArrayObject*>(result));
    Py_DECREF(result);
    const int num_axes = blob->num_axes();
    vector<npy_intp> dims(blob->shape().begin(), blob->shape().end());
    PyObject *arr_obj = PyArray_SimpleNewFromData(num_axes, dims.data(),
                                                  NPY_FLOAT32, data);
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
  Blob<Dtype>* self = bp::extract<Blob<Dtype>*>(args[0]);
  vector<int> shape(bp::len(args) - 1);
  for (int i = 1; i < bp::len(args); ++i) {
    shape[i - 1] = bp::extract<int>(args[i]);
  }
  self->Reshape(shape);
  // We need to explicitly return None to use bp::raw_function.
  return bp::object();
}

bp::object BlobVec_add_blob(bp::tuple args, bp::dict kwargs) {
  if (bp::len(kwargs) > 0) {
    throw std::runtime_error("BlobVec.add_blob takes no kwargs");
  }
  typedef vector<shared_ptr<Blob<Dtype> > > BlobVec;
  BlobVec* self = bp::extract<BlobVec*>(args[0]);
  vector<int> shape(bp::len(args) - 1);
  for (int i = 1; i < bp::len(args); ++i) {
    shape[i - 1] = bp::extract<int>(args[i]);
  }
  self->push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  // We need to explicitly return None to use bp::raw_function.
  return bp::object();
}

template<typename Dtype>
class SolverCallback: public Solver<Dtype>::Callback {
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
template<typename Dtype>
void Solver_add_callback(Solver<Dtype> * solver, bp::object on_start,
  bp::object on_gradients_ready) {
  solver->add_callback(new SolverCallback<Dtype>(on_start, on_gradients_ready));
}

// Seems boost cannot call the base method directly
void Solver_add_nccl(SGDSolver<Dtype>* solver
#ifdef USE_NCCL
  , NCCL<Dtype>* nccl
#endif
) {
#ifdef USE_NCCL
  solver->add_callback(nccl);
#endif
}

template<typename Dtype>
class NetCallback: public Net<Dtype>::Callback {
 public:
  explicit NetCallback(bp::object run) : run_(run) {}

 protected:
  virtual void run(int layer) {
    run_(layer);
  }
  bp::object run_;
};
void Net_before_forward(Net<Dtype>* net, bp::object run) {
  net->add_before_forward(new NetCallback<Dtype>(run));
}
void Net_after_forward(Net<Dtype>* net, bp::object run) {
  net->add_after_forward(new NetCallback<Dtype>(run));
}
void Net_before_backward(Net<Dtype>* net, bp::object run) {
  net->add_before_backward(new NetCallback<Dtype>(run));
}
void Net_after_backward(Net<Dtype>* net, bp::object run) {
  net->add_after_backward(new NetCallback<Dtype>(run));
}

void Net_add_nccl(Net<Dtype>* net
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
  NCCL(shared_ptr<Solver<Dtype> > solver, const string& uid) {}
};
#endif

namespace detail {
struct ArrayShape_to_tuple {
  static PyObject *convert(const ArrayShape &p) {
    PyObject *r = PyTuple_New(p.size());
    for (size_t i = 0; i < p.size(); i++)
      PyTuple_SET_ITEM(r, i, PyLong_FromLong(p[i]));
    return r;
  }
  static PyTypeObject const *get_pytype() {
    return &PyTuple_Type;
  }
};
}  // namespace detail
struct ArrayShape_to_tuple {
  ArrayShape_to_tuple() {
    bp::to_python_converter < ArrayShape, detail::ArrayShape_to_tuple
#ifdef BOOST_PYTHON_SUPPORTS_PY_SIGNATURES
    , true
#endif
    > ();
  }
};

struct ArrayShape_from_tuple {
  ArrayShape_from_tuple() {
    bp::converter::registry::push_back(&convertible,
                                       &construct,
                                       bp::type_id<ArrayShape>()
#ifdef BOOST_PYTHON_SUPPORTS_PY_SIGNATURES
                                       , get_pytype
#endif
                                       ); // NOLINT[whitespace/parens]
  }

  static const PyTypeObject *get_pytype() {
    return &PyTuple_Type;
  }

  static void *convertible(PyObject *o) {
    if (!PyTuple_Check(o)) return 0;
    for (int i = 0, S = PyTuple_Size(o); i < S; i++)
      if (!PyLong_Check(PyTuple_GET_ITEM(o, i)))
        return 0;
    return o;
  }

  static void construct(
    PyObject *o,
    boost::python::converter::rvalue_from_python_stage1_data *data) {
    using bp::converter::rvalue_from_python_storage;
    ArrayShape s(PyTuple_Size(o));
    for (int i = 0; i < s.size(); i++)
      s[i] = PyLong_AS_LONG(PyTuple_GET_ITEM(o, i));
    void *storage =
      ((rvalue_from_python_storage<ArrayShape> *) data)->storage.bytes;
    new(storage) ArrayShape(s);
    data->convertible = storage;
  }
};
/**** Python array interface ****/
template<typename T> Array<T> Blob_data_array( Blob<T> & b ) { // NOLINT[runtime/references]
  return Array<T>(b.data(), b.shape());
}
template<typename T> Array<T> Blob_diff_array( Blob<T> & b ) { // NOLINT[runtime/references]
  return Array<T>(b.diff(), b.shape());
}
template<typename T> struct TS {};
template<> struct TS<float> { static const char * value(){ return "f4"; } };
template<> struct TS<double> { static const char * value(){ return "f8"; } };

template<typename T>
bp::dict Array_interface(Array<T> &a) { // NOLINT[runtime/references]
  bp::dict r;
  r["shape"] = bp::tuple(a.shape());
  r["typestr"] = TS<T>::value();
  r["version"] = 3;
  r["data"] = bp::make_tuple((size_t)a.mutable_cpu_data(), 0);
  return r;
}
template<typename T, typename V>
void Array_setitem(Array<T> &a, const bp::ellipsis &, const V &v) { // NOLINT[runtime/references]
  a = v;
}
template<typename T> struct TPE {};
template<> struct TPE<float>{ static int T() { return NPY_FLOAT32; } };
template<> struct TPE<double>{ static int T() { return NPY_FLOAT64; } };
template<typename T>
void Array_setnumpy(Array<T> *a, const bp::ndarray &v, bool resize) {
  // Get the numpy array
  CHECK(PyArray_Check(v.ptr())) << "Wrong type: python array expected";
  PyArrayObject * na = reinterpret_cast<PyArrayObject*>(v.ptr());
  // Resize if needed
  const int D = PyArray_NDIM(na);
  ArrayShape new_shape(D);
  std::copy(PyArray_DIMS(na), PyArray_DIMS(na)+D, new_shape.begin());
  // Create the array
  if (resize && (!a || new_shape != a->shape()))
    *a = Array<T>(new_shape, a->mode());
  else
    CHECK_EQ(new_shape, a->shape()) << "Shape cannot change";
  // Copy the data (create a numpy array and let numpy deal with the convertion)
  PyObject * tmp = PyArray_SimpleNewFromData(D, PyArray_DIMS(na), TPE<T>::T(),
                                             a->mutable_cpu_data());
  PyArray_CopyInto(reinterpret_cast<PyArrayObject*>(tmp), na);
  Py_DECREF(tmp);
}
template<typename T>
void Array_setnumeric(Array<T> &a, const bp::ellipsis &, const bp::ndarray &v) { // NOLINT[runtime/references]
  Array_setnumpy(&a, v, false);
}
template<typename T>
shared_ptr<Array<T> > Array_from_numpy(const bp::ndarray &v) {
  shared_ptr<Array<T> > a(new Array<T>());
  Array_setnumpy(a.get(), v, true);
  return a;
}

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolveOverloads, Solve, 0, 1);

BOOST_PYTHON_MODULE(_caffe) {
  // below, we prepend an underscore to methods that will be replaced
  // in Python

  bp::scope().attr("__version__") = AS_STRING(CAFFE_VERSION);

  // Caffe utility functions
  bp::def("init_log", &InitLog);
  bp::def("init_log", &InitLogInfo);
  bp::def("log", &Log);
  bp::def("set_mode_cpu", &set_mode_cpu);
  bp::def("set_mode_gpu", &set_mode_gpu);
  bp::def("set_random_seed", &set_random_seed);
  bp::def("set_device", &Caffe::SetDevice);
  bp::def("solver_count", &Caffe::solver_count);
  bp::def("set_solver_count", &Caffe::set_solver_count);
  bp::def("solver_rank", &Caffe::solver_rank);
  bp::def("set_solver_rank", &Caffe::set_solver_rank);
  bp::def("set_multiprocess", &Caffe::set_multiprocess);

  bp::def("layer_type_list", &LayerRegistry<Dtype>::LayerTypeList);

  bp::class_<Net<Dtype>, shared_ptr<Net<Dtype> >, boost::noncopyable >("Net",
    bp::no_init)
    // Constructor
    .def("__init__", bp::make_constructor(&Net_Init,
          bp::default_call_policies(), (bp::arg("network_file"), "phase",
            bp::arg("level")=0, bp::arg("stages")=bp::object(),
            bp::arg("weights")=bp::object())))
    // Legacy constructor
    .def("__init__", bp::make_constructor(&Net_Init_Load))
    .def("_forward", &Net<Dtype>::ForwardFromTo)
    .def("_backward", &Net<Dtype>::BackwardFromTo)
    .def("reshape", &Net<Dtype>::Reshape)
    .def("clear_param_diffs", &Net<Dtype>::ClearParamDiffs)
    // The cast is to select a particular overload.
    .def("copy_from", static_cast<void (Net<Dtype>::*)(const string)>(
        &Net<Dtype>::CopyTrainedLayersFrom))
    .def("share_with", &Net<Dtype>::ShareTrainedLayersWith)
    .add_property("_blob_loss_weights", bp::make_function(
        &Net<Dtype>::blob_loss_weights, bp::return_internal_reference<>()))
    .def("_bottom_ids", bp::make_function(&Net<Dtype>::bottom_ids,
        bp::return_value_policy<bp::copy_const_reference>()))
    .def("_top_ids", bp::make_function(&Net<Dtype>::top_ids,
        bp::return_value_policy<bp::copy_const_reference>()))
    .add_property("_blobs", bp::make_function(&Net<Dtype>::blobs,
        bp::return_internal_reference<>()))
    .add_property("layers", bp::make_function(&Net<Dtype>::layers,
        bp::return_internal_reference<>()))
    .add_property("_blob_names", bp::make_function(&Net<Dtype>::blob_names,
        bp::return_value_policy<bp::copy_const_reference>()))
    .add_property("_layer_names", bp::make_function(&Net<Dtype>::layer_names,
        bp::return_value_policy<bp::copy_const_reference>()))
    .add_property("_inputs", bp::make_function(&Net<Dtype>::input_blob_indices,
        bp::return_value_policy<bp::copy_const_reference>()))
    .add_property("_outputs",
        bp::make_function(&Net<Dtype>::output_blob_indices,
        bp::return_value_policy<bp::copy_const_reference>()))
    .def("_set_input_arrays", &Net_SetInputArrays,
        bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >())
    .def("save", &Net_Save)
    .def("save_hdf5", &Net_SaveHDF5)
    .def("load_hdf5", &Net_LoadHDF5)
    .def("before_forward", &Net_before_forward)
    .def("after_forward", &Net_after_forward)
    .def("before_backward", &Net_before_backward)
    .def("after_backward", &Net_after_backward)
    .def("after_backward", &Net_add_nccl);
  BP_REGISTER_SHARED_PTR_TO_PYTHON(Net<Dtype>);

  bp::class_<Blob<Dtype>, shared_ptr<Blob<Dtype> >, boost::noncopyable>(
    "Blob", bp::no_init)
    .add_property("shape",
        bp::make_function(
            static_cast<const vector<int>& (Blob<Dtype>::*)() const>(
                &Blob<Dtype>::shape),
            bp::return_value_policy<bp::copy_const_reference>()))
    .add_property("num",      &Blob<Dtype>::num)
    .add_property("channels", &Blob<Dtype>::channels)
    .add_property("height",   &Blob<Dtype>::height)
    .add_property("width",    &Blob<Dtype>::width)
    .add_property("count",    static_cast<int (Blob<Dtype>::*)() const>(
        &Blob<Dtype>::count))
    .def("reshape",           bp::raw_function(&Blob_Reshape))
    .add_property("data_array", &Blob_data_array<Dtype>)
    .add_property("diff_array", &Blob_diff_array<Dtype>)
    .add_property("data",     bp::make_function(&Blob<Dtype>::mutable_cpu_data,
          NdarrayCallPolicies()))
    .add_property("diff",     bp::make_function(&Blob<Dtype>::mutable_cpu_diff,
          NdarrayCallPolicies()));
  BP_REGISTER_SHARED_PTR_TO_PYTHON(Blob<Dtype>);

  bp::class_<Layer<Dtype>, shared_ptr<PythonLayer<Dtype> >,
    boost::noncopyable>("Layer", bp::init<const LayerParameter&>())
    .add_property("blobs", bp::make_function(&Layer<Dtype>::blobs,
          bp::return_internal_reference<>()))
    .def("setup", &Layer<Dtype>::LayerSetUp)
    .def("reshape", &Layer<Dtype>::Reshape)
    .add_property("type", bp::make_function(&Layer<Dtype>::type));
  BP_REGISTER_SHARED_PTR_TO_PYTHON(Layer<Dtype>);

  bp::class_<SolverParameter>("SolverParameter", bp::no_init)
    .add_property("max_iter", &SolverParameter::max_iter)
    .add_property("display", &SolverParameter::display)
    .add_property("layer_wise_reduce", &SolverParameter::layer_wise_reduce);
  bp::class_<LayerParameter>("LayerParameter", bp::no_init);

  bp::class_<Solver<Dtype>, shared_ptr<Solver<Dtype> >, boost::noncopyable>(
    "Solver", bp::no_init)
    .add_property("net", &Solver<Dtype>::net)
    .add_property("test_nets", bp::make_function(&Solver<Dtype>::test_nets,
          bp::return_internal_reference<>()))
    .add_property("iter", &Solver<Dtype>::iter)
    .def("add_callback", &Solver_add_callback<Dtype>)
    .def("add_callback", &Solver_add_nccl)
    .def("solve", static_cast<void (Solver<Dtype>::*)(const char*)>(
          &Solver<Dtype>::Solve), SolveOverloads())
    .def("step", &Solver<Dtype>::Step)
    .def("restore", &Solver<Dtype>::Restore)
    .def("snapshot", &Solver<Dtype>::Snapshot)
    .add_property("param", bp::make_function(&Solver<Dtype>::param,
              bp::return_value_policy<bp::copy_const_reference>()));
  BP_REGISTER_SHARED_PTR_TO_PYTHON(Solver<Dtype>);

  bp::class_<SGDSolver<Dtype>, bp::bases<Solver<Dtype> >,
    shared_ptr<SGDSolver<Dtype> >, boost::noncopyable>(
        "SGDSolver", bp::init<string>());
  bp::class_<NesterovSolver<Dtype>, bp::bases<Solver<Dtype> >,
    shared_ptr<NesterovSolver<Dtype> >, boost::noncopyable>(
        "NesterovSolver", bp::init<string>());
  bp::class_<AdaGradSolver<Dtype>, bp::bases<Solver<Dtype> >,
    shared_ptr<AdaGradSolver<Dtype> >, boost::noncopyable>(
        "AdaGradSolver", bp::init<string>());
  bp::class_<RMSPropSolver<Dtype>, bp::bases<Solver<Dtype> >,
    shared_ptr<RMSPropSolver<Dtype> >, boost::noncopyable>(
        "RMSPropSolver", bp::init<string>());
  bp::class_<AdaDeltaSolver<Dtype>, bp::bases<Solver<Dtype> >,
    shared_ptr<AdaDeltaSolver<Dtype> >, boost::noncopyable>(
        "AdaDeltaSolver", bp::init<string>());
  bp::class_<AdamSolver<Dtype>, bp::bases<Solver<Dtype> >,
    shared_ptr<AdamSolver<Dtype> >, boost::noncopyable>(
        "AdamSolver", bp::init<string>());

  bp::def("get_solver", &GetSolverFromFile,
      bp::return_value_policy<bp::manage_new_object>());

  // vector wrappers for all the vector types we use
  bp::class_<vector<shared_ptr<Blob<Dtype> > > >("BlobVec")
    .def(bp::vector_indexing_suite<vector<shared_ptr<Blob<Dtype> > >, true>())
    .def("add_blob", bp::raw_function(&BlobVec_add_blob));
  bp::class_<vector<Blob<Dtype>*> >("RawBlobVec")
    .def(bp::vector_indexing_suite<vector<Blob<Dtype>*>, true>());
  bp::class_<vector<shared_ptr<Layer<Dtype> > > >("LayerVec")
    .def(bp::vector_indexing_suite<vector<shared_ptr<Layer<Dtype> > >, true>());
  bp::class_<vector<string> >("StringVec")
    .def(bp::vector_indexing_suite<vector<string> >());
//   bp::class_<vector<int> >("IntVec")
//     .def(bp::vector_indexing_suite<vector<int> >());
  bp::class_<vector<Dtype> >("DtypeVec")
    .def(bp::vector_indexing_suite<vector<Dtype> >());
  bp::class_<vector<shared_ptr<Net<Dtype> > > >("NetVec")
    .def(bp::vector_indexing_suite<vector<shared_ptr<Net<Dtype> > >, true>());
  bp::class_<vector<bool> >("BoolVec")
    .def(bp::vector_indexing_suite<vector<bool> >());

  bp::class_<NCCL<Dtype>, shared_ptr<NCCL<Dtype> >,
    boost::noncopyable>("NCCL",
                        bp::init<shared_ptr<Solver<Dtype> >, const string&>())
#ifdef USE_NCCL
    .def("new_uid", &NCCL<Dtype>::new_uid).staticmethod("new_uid")
    .def("bcast", &NCCL<Dtype>::Broadcast)
#endif
    /* NOLINT_NEXT_LINE(whitespace/semicolon) */
  ;
  BP_REGISTER_SHARED_PTR_TO_PYTHON(NCCL<Dtype>);

  bp::class_<Timer, shared_ptr<Timer>, boost::noncopyable>(
    "Timer", bp::init<>())
    .def("start", &Timer::Start)
    .def("stop", &Timer::Stop)
    .add_property("ms", &Timer::MilliSeconds);
  BP_REGISTER_SHARED_PTR_TO_PYTHON(Timer);

  // Caffe Array
  ArrayShape_to_tuple();
  ArrayShape_from_tuple();

  bp::enum_<ArrayMode>("ArrayMode")
  .value("DEFAULT", AR_DEFAULT)
  .value("CPU", AR_CPU)
  .value("GPU", AR_GPU);

  typedef Expression<Dtype>(ArrayBase<Dtype>::*Binary1)(Dtype) const;
  typedef Expression<Dtype>(ArrayBase<Dtype>::*Binary2)(const ArrayBase<Dtype>&)
    const;
  bp::class_<ArrayBase<Dtype>, boost::noncopyable>("ArrayBase", bp::no_init)
  // Binary operators
  .def(bp::self + bp::self)
  .def(bp::self - bp::self)
  .def(bp::self * bp::self)
  .def(bp::self / bp::self)
  .def(bp::self + float())
  .def(bp::self - float())
  .def(bp::self * float())
  .def(bp::self / float())
  .def(float() + bp::self)
  .def(float() - bp::self)
  .def(float() * bp::self)
  .def(float() / bp::self)
  // Unary operators
  .def(- bp::self)
  // Define all unary and reduction functions
#define DEF_UNARY(F, f) .def(#f, &ArrayBase<Dtype>::f)
  LIST_UNARY(DEF_UNARY)
  LIST_REDUCTION(DEF_UNARY)
  .def("mean", &ArrayBase<Dtype>::mean)
#undef DEF_UNARY
  // Define all binary functions
#define DEF_BINARY(F, f) \
  .def(#f, static_cast<Binary1>(&ArrayBase<Dtype>::f))\
  .def(#f, static_cast<Binary2>(&ArrayBase<Dtype>::f))
  LIST_BINARY(DEF_BINARY)
#undef DEF_BINARY
  // Other functions
  .def("eval", &ArrayBase<Dtype>::eval)
  .add_property("mode", &Array<Dtype>::mode)
  .add_property("effective_mode", &Array<Dtype>::effectiveMode)
  .add_property("shape", bp::make_function(&Array<Dtype>::shape,
    bp::return_value_policy<bp::copy_const_reference>()) );

  bp::class_<Array<Dtype>, bp::bases<ArrayBase<Dtype> > >("Array")
  .def(bp::init<ArrayMode>())
  .def(bp::init<ArrayShape>())
  .def(bp::init<ArrayShape, ArrayMode>())
  .def("__init__", bp::make_constructor(&Array_from_numpy<Dtype>))
  .def("reshape", &Array<Dtype>::reshape)
  .def(bp::self += bp::self)
  .def(bp::self -= bp::self)
  .def(bp::self *= bp::self)
  .def(bp::self /= bp::self)
  .def(bp::self += float())
  .def(bp::self -= float())
  .def(bp::self *= float())
  .def(bp::self /= float())
  .def("__setitem__", &Array_setitem<Dtype, Dtype>)
  .def("__setitem__", &Array_setitem<Dtype, Expression<Dtype> >)
  .def("__setitem__", &Array_setitem<Dtype, Array<Dtype> >)
  .def("__setitem__", &Array_setnumeric<Dtype>)
  .def("__getitem__", static_cast<Array<Dtype> (Array<Dtype>::*)(size_t)>(
    &Array<Dtype>::operator[]))
  .add_property("mode", &Array<Dtype>::mode, &Array<Dtype>::setMode)
  .add_property("__array_interface__", &Array_interface<Dtype>);

  bp::class_ < Expression<Dtype>, bp::bases<ArrayBase<Dtype> > > ("Expression",
                                                                  bp::no_init);

  // boost python expects a void (missing) return value, while import_array
  // returns NULL for python3. import_array1() forces a void return value.
  import_array1();
}

}  // namespace caffe
