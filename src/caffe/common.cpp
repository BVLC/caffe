#if defined(_MSC_VER)
#include <process.h>
#define getpid() _getpid()
#endif

#include <boost/thread.hpp>
#include <glog/logging.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <tuple>
#include <vector>

#include "caffe/common.hpp"

#include "caffe/backend/device.hpp"
#include "caffe/backend/cuda/cuda_device.hpp"
#include "caffe/backend/hip/hip_device.hpp"
#include "caffe/backend/opencl/ocl_device.hpp"
#include "caffe/util/rng.hpp"

#if defined(USE_OPENCL)
  #if defined(USE_CLBLAS)
    #include <clBLAS.h>                     // NOLINT
  #elif defined(USE_CLBLAST)
    #include <clblast.h>                    // NOLINT
  #endif  // USE_CLBLAS or USE_CLBLAST
#endif  // USE_OPENCL

namespace caffe {

// Make sure each thread can have different values.
static boost::thread_specific_ptr<Caffe> thread_instance_;

// Pointer to the global instance of Caffe
static Caffe* global_instance_;
static std::mutex instance_mutex_;

// Device contexts are initialized once and shared on all threads
vector< shared_ptr<Device> > Caffe::devices_;

#ifdef USE_OPENCL
inline vector<viennacl::ocl::platform> get_platforms_safe() {
  vector<viennacl::ocl::platform> ret;
  cl_int err;
  cl_uint num_platforms = 0;
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  VIENNACL_ERR_CHECK(err);

  vector<cl_platform_id> ids(num_platforms);
  err = clGetPlatformIDs(num_platforms, &ids[0], &num_platforms);
  VIENNACL_ERR_CHECK(err);

  for (cl_uint i = 0; i < num_platforms; ++i) {
    ret.push_back(viennacl::ocl::platform(ids[i]));
  }

  return ret;
}
#endif  // USE_OPENCL

Caffe& Caffe::Get() {
  instance_mutex_.lock();
  if (global_instance_ == nullptr) {
    // The first call must be single threaded
    // and defines the global instance
    thread_instance_.reset(new Caffe());
    global_instance_ = thread_instance_.get();
  }
  if (!thread_instance_.get()) {
    // Every thread initially gets a copy of the global initialization.
    // Later, every thread can switch to a different default device
    // or change other aspects of the Caffe object
    thread_instance_.reset(new Caffe(*global_instance_));
  }
  instance_mutex_.unlock();
  return *(thread_instance_.get());
}

// random seeding
int64_t cluster_seedgen(void) {
  int64_t s, seed, pid;
  FILE* f = fopen("/dev/urandom", "rb");
  if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
    fclose(f);
    return seed;
  }

  LOG(INFO)<< "System entropy source not available, "
  "using fallback algorithm to generate seed instead.";
  if (f)
    fclose(f);

  pid = getpid();
  s = time(NULL);
  seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
}

void GlobalInit(int* pargc, char*** pargv) {
  // Google flags.
  ::gflags::ParseCommandLineFlags(pargc, pargv, true);
  // Google logging.
  ::google::InitGoogleLogging(*(pargv)[0]);
  // Provide a backtrace on segfault.

  // Windows port of glogs doesn't have this function built
#if !defined(_MSC_VER)
  ::google::InstallFailureSignalHandler();
#endif
}


Device *Caffe::GetDevice(int id, bool listId) {
  if (Get().devices_.size() == 0) {
    return Get().cpu_device_.get();
  }
  if (listId) {
    return
        id == -1 ?
            Get().default_device_ :
            Get().devices_[id % Get().devices_.size()].get();
  } else {
    for (int i = 0; i < Get().devices_.size(); ++i) {
      Device* device = Get().devices_[i].get();
      if (device->id() == id) {
        return device;
      }
    }
    return GetDefaultDevice();
  }
}

Device *Caffe::GetDefaultDevice() {
  return Get().default_device_;
}

Device *Caffe::GetCPUDevice() {
  return Get().cpu_device_.get();
}

// Copy constructor for thread-local copy
Caffe::Caffe(const Caffe &obj)
    :
#ifdef USE_CUDA
      cublas_handle_(NULL),
      curand_generator_(NULL),
      curand_generator64_(NULL),
#endif  // USE_CUDA
      random_generator_(),
      mode_(Caffe::CPU),
      cpu_device_(new Device()),
      default_device_(cpu_device_.get()),
      solver_count_(1) {
  mode_ = obj.mode_;
  default_device_ = obj.default_device_;
  cpu_device_ = obj.cpu_device_;
  solver_count_ = obj.solver_count_;
}

void Caffe::SelectDevice(int id, bool listId) {
  Caffe::SelectDevice(GetDevice(id, listId));
}

void Caffe::SelectDevice(Device* device_context) {
#ifndef CPU_ONLY
  Get().default_device_ = device_context;

  if (device_context->backend() == Backend::BACKEND_CUDA) {
#ifdef USE_CUDA
    CUDA_CHECK(cudaSetDevice(device_context->id()));

    if (Get().cublas_handle_) {
      CUBLAS_CHECK(cublasDestroy(Get().cublas_handle_));
    }
    if (Get().curand_generator_) {
      CURAND_CHECK(curandDestroyGenerator(Get().curand_generator_));
    }
    if (Get().curand_generator64_) {
      CURAND_CHECK(curandDestroyGenerator(Get().curand_generator64_));
    }
    CUBLAS_CHECK(cublasCreate(&Get().cublas_handle_));

    if (cublasCreate(&(Get().cublas_handle_)) != CUBLAS_STATUS_SUCCESS) {
      LOG(ERROR)<< "Cannot create Cublas handle. Cublas won't be available.";
    }
    // Try to create a curand handler.
    if (curandCreateGenerator(&(Get().curand_generator_),
                              CURAND_RNG_PSEUDO_DEFAULT)
        != CURAND_STATUS_SUCCESS
        || curandSetPseudoRandomGeneratorSeed((Get().curand_generator_),
                                              cluster_seedgen())
            != CURAND_STATUS_SUCCESS) {
      LOG(ERROR)<< "Cannot create Curand generator. Curand won't be available.";
    }
    if (curandCreateGenerator(&(Get().curand_generator64_),
                              CURAND_RNG_QUASI_SOBOL64)
        != CURAND_STATUS_SUCCESS) {
      LOG(ERROR)<< "Cannot create Curand generator. Curand won't be available.";
    }

#endif  // USE_CUDA
  } else if (device_context->backend() == Backend::BACKEND_OPENCL) {
#ifdef USE_OPENCL
#ifdef USE_CLBLAS
    clblasSetup();
#endif  // USE_CLBLAS
#endif  // USE_OPENCL
  }
#endif  // !CPU_ONLY
}

#ifdef CPU_ONLY  // CPU-only Caffe.

Caffe::Caffe() : random_generator_(),
                 mode_(Caffe::CPU),
                 cpu_device_(new Device(-1, -1, Backend::BACKEND_CPU)),
                 default_device_(cpu_device_.get()),
                 solver_count_(1), solver_rank_(0), multiprocess_(false) { }

Caffe::~Caffe() {}

void Caffe::set_random_seed(const size_t seed, Device* device_context) {
  // RNG seed
  Get().random_generator_.reset(new RNG(seed));
}

void Caffe::SetDevices(std::vector<int> device_ids) {
  NO_GPU;
}

void Caffe::SetDevice(const int device_id) {
  NO_GPU;
}

void Caffe::DeviceQuery() {
  NO_GPU;
}


void Caffe::Synchronize(int device_id) {
  NO_GPU;
}

int Caffe::EnumerateDevices(bool silent) {
  NO_GPU;
  return 0;
}

bool Caffe::CheckDevice(const int device_id) {
  NO_GPU;
  return false;
}

int Caffe::FindDevice(const int start_id) {
  NO_GPU;
  return -1;
}

class Caffe::RNG::Generator {
 public:
  Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {}
  explicit Generator(size_t seed) : rng_(new caffe::rng_t(seed)) {}
  caffe::rng_t* rng() {return rng_.get();}
 private:
  shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG() : generator_(new Generator()) {}

Caffe::RNG::RNG(size_t seed) : generator_(new Generator(seed)) {}

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
  generator_ = other.generator_;
  return *this;
}

void* Caffe::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

#else  // Normal GPU + CPU Caffe.

Caffe::Caffe()
    :
#ifdef USE_CUDA
      cublas_handle_(NULL),
      curand_generator_(NULL),
      curand_generator64_(NULL),
#endif  // USE_CUDA
      random_generator_(),
      mode_(Caffe::CPU),
      cpu_device_(new Device()),
      default_device_(cpu_device_.get()),
    solver_count_(1), solver_rank_(0), multiprocess_(false) {
}

Caffe::~Caffe() {
  // Make sure all device contexts and
  // dependent memory blocks are freed properly
  if (this == global_instance_) {
      instance_mutex_.lock();
      global_instance_ = NULL;
      instance_mutex_.unlock();
      devices_.clear();
  }
#ifdef USE_CUDA
  if (cublas_handle_)
    CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  cublas_handle_ = nullptr;
  if (curand_generator_) {
    CURAND_CHECK(curandDestroyGenerator(curand_generator_));
    curand_generator_ = nullptr;
  }
  if (curand_generator64_) {
    CURAND_CHECK(curandDestroyGenerator(curand_generator64_));
    curand_generator64_ = nullptr;
  }
#endif  // USE_CUDA
}

void Caffe::set_random_seed(const size_t seed, Device* device_context) {
  if (device_context->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    // Curand seed
    static bool g_curand_availability_logged = false;
    if (Get().curand_generator_) {
      CURAND_CHECK(
          curandSetPseudoRandomGeneratorSeed(curand_generator(), seed));
      CURAND_CHECK(curandSetGeneratorOffset(curand_generator(), 0));
    } else {
      if (!g_curand_availability_logged) {
        LOG(ERROR)<<
        "Curand not available. Skipping setting the curand seed.";
        g_curand_availability_logged = true;
      }
    }
    if (Get().curand_generator64_) {
      CURAND_CHECK(curandSetGeneratorOffset(curand_generator64(), 0));
    } else {
      if (!g_curand_availability_logged) {
        LOG(ERROR)<<
        "Curand not available. Skipping setting the curand seed.";
        g_curand_availability_logged = true;
      }
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_OPENCL
// TODO: Proper RNG and Seed for OpenCL
#endif  // USE_OPENCL
  }
  // RNG seed
  Get().random_generator_.reset(new RNG(seed));
}

void Caffe::Synchronize(int device_id) {
  if (Caffe::mode() == Brew::GPU) {
    Device * device_context = Caffe::GetDevice(device_id, true);
    if (device_context->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
      cudaDeviceSynchronize();
#endif  // USE_CUDA
    } else {
#ifdef USE_OPENCL
      viennacl::ocl::context &ctx = viennacl::ocl::get_context(
          GetDevice(device_id, true)->id());
      ctx.get_queue().finish();
#endif  // USE_OPENCL
    }
  }
}

int Caffe::EnumerateDevices(bool silent) {
  int cuda_device_count = 0;
  int opencl_device_count = 0;

#ifdef USE_CUDA
  cudaGetDeviceCount(&cuda_device_count);
#endif  // USE_CUDA

#ifdef USE_OPENCL
  typedef vector<viennacl::ocl::platform> platforms_type;
  platforms_type platforms = get_platforms_safe();

  vector<std::tuple<viennacl::ocl::platform,
    viennacl::ocl::device>> platform_devices;

  // Loop through devices
  for (std::size_t platform_id = 0; platform_id < platforms.size();
      ++platform_id) {
    typedef vector<viennacl::ocl::device> devices_type;
    try {
      devices_type devices = platforms[platform_id].devices(CL_DEVICE_TYPE_ALL);
      for (std::size_t device_id = 0; device_id < devices.size(); ++device_id) {
        platform_devices.push_back(
            std::make_tuple(platforms[platform_id], devices[device_id]));
        opencl_device_count++;
      }
    } catch (...) {
      if (!silent) {
        LOG(INFO)<< "OpenCL platform: "
        << platforms[platform_id].info()
        << " does not work correctly.";
      }
    }
  }
#endif  // USE_OPENCL

  if (!silent) {
    LOG(INFO)<< "Total devices: " << cuda_device_count + opencl_device_count;
    LOG(INFO)<< "CUDA devices: " << cuda_device_count;
    LOG(INFO)<< "OpenCL devices: " << opencl_device_count;

    // Display info for all devices
#ifdef USE_CUDA
    for (int i = 0; i < cuda_device_count; ++i) {
      cudaDeviceProp prop;
      CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
      LOG(INFO)<< "Device id:                     "
      << i;
      LOG(INFO)<< "Device backend:                "
      << "CUDA";
      LOG(INFO)<< "Backend details:               "
      << "CUDA";
      LOG(INFO)<< "Device vendor:                 "
      << "NVIDIA Corporation";
      LOG(INFO)<< "Name:                          "
      << prop.name;
      LOG(INFO)<< "Total global memory:           "
      << prop.totalGlobalMem;
    }
#endif  // USE_CUDA

#ifdef USE_OPENCL
    for (int i = 0; i < opencl_device_count; ++i) {
      LOG(INFO)<< "Device id:                     "
      << cuda_device_count + i;
      LOG(INFO)<< "Device backend:                "
      << "OpenCL";
      LOG(INFO)<< "Backend details:               "
      << std::get<0>(platform_devices[i]).info();
      LOG(INFO)<< "Device vendor:                 "
      << std::get<1>(platform_devices[i]).vendor();
      LOG(INFO)<< "Name:                          "
      << std::get<1>(platform_devices[i]).name();
      LOG(INFO)<< "Total global memory:           "
      << std::get<1>(platform_devices[i]).global_mem_size();
    }
#endif  // USE_OPENCL
  }

  return cuda_device_count + opencl_device_count;
}

void Caffe::SetDevices(vector<int> device_ids) {
  int initcount = 0;
  Get().devices_.clear();
  int cuda_device_count = 0;

#ifdef USE_CUDA
  cudaGetDeviceCount(&cuda_device_count);
  for (int i = 0; i < cuda_device_count; ++i) {
    for (int j = 0; j < device_ids.size(); ++j) {
      if (device_ids[j] == i) {
        shared_ptr<Device> dev(
            new CudaDevice(i, initcount));
        Get().devices_.emplace_back(dev);
        dev->Init();
        ++initcount;
      }
    }
  }
#endif  // USE_CUDA

  // Initialize OpenCL devices
#ifdef USE_OPENCL
  int opencl_device_count = 0;

  typedef vector<viennacl::ocl::platform> platforms_type;
  platforms_type platforms = get_platforms_safe();

  vector<std::tuple<viennacl::ocl::platform, viennacl::ocl::device> >
                                                               platform_devices;

  // Loop through devices
  for (int platform_id = 0; platform_id < platforms.size();
      ++platform_id) {
    typedef vector<viennacl::ocl::device> devices_type;
    devices_type devices;
    try {
      devices = platforms[platform_id].devices(
      CL_DEVICE_TYPE_ALL);
    } catch (...) {
      LOG(INFO)<< "OpenCL platform: "
      << platforms[platform_id].info()
      << " does not work correctly.";
    }
    for (int device_id = 0; device_id < devices.size(); ++device_id) {
      platform_devices.push_back(
          std::make_tuple(platforms[platform_id], devices[device_id]));
      // Check if this device is really used
      for (int i = 0; i < device_ids.size(); ++i) {
        int device_id = device_ids[i];
        if (device_id == cuda_device_count + opencl_device_count) {
          // Setup actual context for this device
          viennacl::ocl::setup_context(device_id,
                            std::get<1>(platform_devices[opencl_device_count]));

          shared_ptr<Device> dev(new OclDevice(device_id, initcount));
          Get().devices_.emplace_back(dev);
          dev->Init();
          ++initcount;
        }
      }
      opencl_device_count++;
    }
  }

#endif  // USE_OPENCL

  if (Get().devices_.size() == 0) {
    LOG(FATAL) << "No device could be initialized." << std::endl;
  }

  Get().default_device_ = GetDevice(0, true);
  Caffe::SelectDevice(Get().default_device_);
}

void Caffe::SetDevice(const int device_id) {
  // Fix for compatibility to python and other interfaces that do not
  // know or call SetDevices directly
  if (Get().devices_.size() == 0) {
    // No device has been initialized so far
    Caffe::SetDevices(vector<int> { device_id });
  }

  Get().default_device_ = GetDevice(0, true);
#if defined(USE_OPENCL) && defined(USE_FFT)
  Get().cl_fft_state_.setup();
#endif
}

#ifdef USE_OPENCL
const cl_context& Caffe::GetOpenCLContext(const int id, bool list_id) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(
         GetDevice(id, list_id)->id());
  return ctx.handle().get();
}

const cl_command_queue& Caffe::GetOpenCLQueue(const int id, bool list_id) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(
         GetDevice(id, list_id)->id());
  return ctx.get_queue().handle().get();
}
#endif  // USE_OPENCL

// Should call explicitly for OCL + FFT
void Caffe::TeardownDevice(const int device_id) {
#if defined(USE_OPENCL) &&defined(USE_FFT)
  Get().cl_fft_state_.teardown();
#endif
}

// FIXME: OpenCL equivalent and device abstraction
void Caffe::DeviceQuery() {
  if (Get().default_device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    cudaDeviceProp prop;
    int device;
    if (cudaSuccess != cudaGetDevice(&device)) {
      printf("No cuda device present.\n");
    } else {
      CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
      LOG(INFO)<< "Device id:                     " << device;
      LOG(INFO)<< "Major revision number:         " << prop.major;
      LOG(INFO)<< "Minor revision number:         " << prop.minor;
      LOG(INFO)<< "Name:                          " << prop.name;
      LOG(INFO)<< "Total global memory:           " << prop.totalGlobalMem;
      LOG(INFO)<< "Total shared memory per block: " << prop.sharedMemPerBlock;
      LOG(INFO)<< "Total registers per block:     " << prop.regsPerBlock;
      LOG(INFO)<< "Warp size:                     " << prop.warpSize;
      LOG(INFO)<< "Maximum memory pitch:          " << prop.memPitch;
      LOG(INFO)<< "Maximum threads per block:     " << prop.maxThreadsPerBlock;
      LOG(INFO)<< "Maximum dimension of block:    "
      << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", "
      << prop.maxThreadsDim[2];
      LOG(INFO)<< "Maximum dimension of grid:     "
      << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", "
      << prop.maxGridSize[2];
      LOG(INFO)<< "Clock rate:                    " << prop.clockRate;
      LOG(INFO)<< "Total constant memory:         " << prop.totalConstMem;
      LOG(INFO)<< "Texture alignment:             " << prop.textureAlignment;
      LOG(INFO)<< "Concurrent copy and execution: "
      << (prop.deviceOverlap ? "Yes" : "No");
      LOG(INFO)<< "Number of multiprocessors:     " << prop.multiProcessorCount;
      LOG(INFO)<< "Kernel execution timeout:      "
      << (prop.kernelExecTimeoutEnabled ? "Yes" : "No");
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_OPENCL
    // TODO: Complete OpenCL device information of current device
#endif  // USE_OPENCL
  }

  return;
}

bool Caffe::CheckDevice(const int device_id) {
  // TODO: Find some OpenCL equivalent here

  // This function checks the availability of GPU #device_id.
  // It attempts to create a context on the device by calling cudaFree(0).
  // cudaSetDevice() alone is not sufficient to check the availability.
  // It lazily records device_id, however, does not initialize a
  // context. So it does not know if the host thread has the permission to use
  // the device or not.
  //
  // In a shared environment where the devices are set to EXCLUSIVE_PROCESS
  // or EXCLUSIVE_THREAD mode, cudaSetDevice() returns cudaSuccess
  // even if the device is exclusively occupied by another process or thread.
  // Cuda operations that initialize the context are needed to check
  // the permission. cudaFree(0) is one of those with no side effect,
  // except the context initialization.
  bool r = true;
#ifdef USE_CUDA
  r =
      ((cudaSuccess == cudaSetDevice(device_id))
          && (cudaSuccess == cudaFree(0)));
  // reset any error that may have occurred.
  cudaGetLastError();
#endif  // USE_CUDA
  return r;
}

int Caffe::FindDevice(const int start_id) {
  // TODO: Find some OpenCL equivalent here

  // This function finds the first available device by checking devices with
  // ordinal from start_id to the highest available value. In the
  // EXCLUSIVE_PROCESS or EXCLUSIVE_THREAD mode, if it succeeds, it also
  // claims the device due to the initialization of the context.
#ifdef USE_CUDA
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  for (int i = start_id; i < count; i++) {
    if (CheckDevice(i)) return i;
  }
#endif  // USE_CUDA
  return -1;
}

class Caffe::RNG::Generator {
 public:
  Generator()
      : rng_(new caffe::rng_t(cluster_seedgen())) {
  }
  explicit Generator(size_t seed)
      : rng_(new caffe::rng_t(seed)) {
  }
  caffe::rng_t* rng() {
    return rng_.get();
  }
 private:
  shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG()
    : generator_(new Generator()) {
}

Caffe::RNG::RNG(size_t seed)
    : generator_(new Generator(seed)) {
}

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
  generator_.reset(other.generator_.get());
  return *this;
}

void* Caffe::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

#endif  // CPU_ONLY

}  // namespace caffe
