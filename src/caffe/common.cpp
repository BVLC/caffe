#include <boost/thread.hpp>
#include <glog/logging.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <tuple>
#include <vector>

#include "caffe/common.hpp"

#include "caffe/device.hpp"
#include "caffe/util/rng.hpp"

#if defined(USE_GREENTEA)
  #include "caffe/greentea/cl_kernels.hpp"  // NOLINT
  #if defined(USE_CLBLAS)
    #include <clBLAS.h>                     // NOLINT
  #elif defined(USE_CLBLAST)
    #include <clblast.h>                    // NOLINT
  #endif  // USE_CLBLAS or USE_CLBLAST
#endif  // USE_GREENTEA

namespace caffe {

// Make sure each thread can have different values.
static boost::thread_specific_ptr<Caffe> thread_instance_;

// Pointer to the global instance of Caffe
static Caffe* global_instance_;
static std::atomic<bool> first(true);

// Device contexts are initialized once and shared on all threads
std::vector< shared_ptr<device> > Caffe::devices_;

Caffe& Caffe::Get() {
  if (first.exchange(false)) {
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
  ::google::InstallFailureSignalHandler();
}


device *Caffe::GetDevice(int id, bool listId) {
  if (listId) {
    return
        id == -1 ?
            Get().default_device_ :
            Get().devices_[id % Get().devices_.size()].get();
  } else {
    for (int i = 0; i < Get().devices_.size(); ++i) {
      device* device = Get().devices_[i].get();
      if (device->id() == id) {
        return device;
      }
    }
    return GetDefaultDevice();
  }
}

device *Caffe::GetDefaultDevice() {
  return Get().default_device_;
}

device *Caffe::GetCPUDevice() {
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
      cpu_device_(new device(-1, -1, Backend::BACKEND_CPU)),
      default_device_(cpu_device_.get()),
      solver_count_(1),
      root_solver_(true) {
  mode_ = obj.mode_;
  default_device_ = obj.default_device_;
  cpu_device_ = obj.cpu_device_;
  root_solver_ = obj.root_solver_;
  solver_count_ = obj.solver_count_;
}

void Caffe::SelectDevice(int id, bool listId) {
  Caffe::SelectDevice(GetDevice(id, listId));
}

void Caffe::SelectDevice(device* device_context) {
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
  } else if (device_context->backend() == Backend::BACKEND_OpenCL) {
#ifdef USE_GREENTEA
#ifdef USE_CLBLAS
    clblasSetup();
#endif  // USE_CLBLAS
#endif  // USE_GREENTEA
  }
#endif  // !CPU_ONLY
}

#ifdef CPU_ONLY  // CPU-only Caffe.

Caffe::Caffe() : random_generator_(),
                 mode_(Caffe::CPU),
                 cpu_device_(new device(-1, -1, Backend::BACKEND_CPU)),
                 default_device_(cpu_device_.get()),
                 solver_count_(1),
                 root_solver_(true) {}

Caffe::~Caffe() {}

void Caffe::set_random_seed(const size_t seed, device* device_context) {
  // RNG seed
  Get().random_generator_.reset(new RNG(seed));
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
      cpu_device_(new device(-1, -1, Backend::BACKEND_CPU)),
      default_device_(cpu_device_.get()),
      solver_count_(1), root_solver_(true) {
}

Caffe::~Caffe() {
  // Make sure all device contexts and
  // dependent memory blocks are freed properly
  if (this == global_instance_) {
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

void Caffe::set_random_seed(const size_t seed, device* device_context) {
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
#ifdef USE_GREENTEA
// TODO: Proper RNG and Seed for OpenCL
#endif  // USE_GREENTEA
  }
  // RNG seed
  Get().random_generator_.reset(new RNG(seed));
}

void Caffe::Synchronize(int device_id) {
  if (Caffe::mode() == Brew::GPU) {
    device * device_context = Caffe::GetDevice(device_id, true);
    if (device_context->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
      cudaDeviceSynchronize();
#endif  // USE_CUDA
    } else {
#ifdef USE_GREENTEA
      viennacl::ocl::context &ctx = viennacl::ocl::get_context(
          GetDevice(device_id, true)->id());
      ctx.get_queue().finish();
#endif  // USE_GREENTEA
    }
  }
}

int Caffe::EnumerateDevices(bool silent) {
  int cuda_device_count = 0;
  int greentea_device_count = 0;

#ifdef USE_CUDA
  cudaGetDeviceCount(&cuda_device_count);
#endif  // USE_CUDA

#ifdef USE_GREENTEA
  typedef std::vector<viennacl::ocl::platform> platforms_type;
  platforms_type platforms = viennacl::ocl::get_platforms();

  std::vector<std::tuple<viennacl::ocl::platform,
    viennacl::ocl::device>> platform_devices;

  // Loop through devices
  for (std::size_t platform_id = 0; platform_id < platforms.size();
      ++platform_id) {
    typedef std::vector<viennacl::ocl::device> devices_type;
    try {
      devices_type devices = platforms[platform_id].devices(CL_DEVICE_TYPE_ALL);
      for (std::size_t device_id = 0; device_id < devices.size(); ++device_id) {
        platform_devices.push_back(
            std::make_tuple(platforms[platform_id], devices[device_id]));
        greentea_device_count++;
      }
    } catch (...) {
      if (!silent) {
        LOG(INFO)<< "OpenCL platform: "
        << platforms[platform_id].info()
        << " does not work correctly.";
      }
    }
  }
#endif  // USE_GREENTEA

  if (!silent) {
    LOG(INFO)<< "Total devices: " << cuda_device_count + greentea_device_count;
    LOG(INFO)<< "CUDA devices: " << cuda_device_count;
    LOG(INFO)<< "OpenCL devices: " << greentea_device_count;

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

#ifdef USE_GREENTEA
    for (int i = 0; i < greentea_device_count; ++i) {
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
#endif  // USE_GREENTEA
  }

  return cuda_device_count + greentea_device_count;
}

void Caffe::SetDevices(std::vector<int> device_ids) {
  int initcount = 0;
  Get().devices_.clear();
  int cuda_device_count = 0;
#ifdef USE_CUDA
  cudaGetDeviceCount(&cuda_device_count);
#endif  // USE_CUDA
  for (int i = 0; i < cuda_device_count; ++i) {
    for (int j = 0; j < device_ids.size(); ++j) {
      if (device_ids[j] == i) {
        shared_ptr<device> dev(
            new device(i, initcount, Backend::BACKEND_CUDA));
        Get().devices_.emplace_back(dev);
        dev->Init();
        ++initcount;
      }
    }
  }

  // Initialize GreenTea devices
#ifdef USE_GREENTEA
  int greentea_device_count = 0;

  typedef std::vector<viennacl::ocl::platform> platforms_type;
  platforms_type platforms = viennacl::ocl::get_platforms();

  std::vector< std::tuple<viennacl::ocl::platform,
      viennacl::ocl::device> > platform_devices;

  // Loop through devices
  for (int platform_id = 0; platform_id < platforms.size();
      ++platform_id) {
    typedef std::vector<viennacl::ocl::device> devices_type;
    try {
      devices_type devices = platforms[platform_id].devices(
      CL_DEVICE_TYPE_ALL);
      for (int device_id = 0; device_id < devices.size(); ++device_id) {
        platform_devices.push_back(
            std::make_tuple(platforms[platform_id], devices[device_id]));
        // Check if this device is really used and initialize
        for (int i = 0; i < device_ids.size(); ++i) {
          int device_id = device_ids[i];
          if (device_id == cuda_device_count + greentea_device_count) {
            // Setup actual context and compile kernels for this device
            viennacl::ocl::setup_context(
                device_id,
                std::get<1>(platform_devices[greentea_device_count]));

            shared_ptr<device> dev(
                new device(device_id,
                                  initcount, Backend::BACKEND_OpenCL));
            Get().devices_.emplace_back(dev);
            dev->Init();
            ++initcount;
          }
        }
        greentea_device_count++;
      }
    } catch (...) {
      LOG(INFO)<< "OpenCL platform: "
      << platforms[platform_id].info()
      << " does not work correctly.";
    }
  }
#endif  // USE_GREENTEA

  Get().default_device_ = GetDevice(0, true);
  Caffe::SelectDevice(Get().default_device_);
}

void Caffe::SetDevice(const int device_id) {
  // Fix for compability to python and other interfaces that do not
  // know or call SetDevices directly
  if (Get().devices_.size() == 0) {
    // No device has been initialized so far
    Caffe::SetDevices(std::vector<int> { device_id });
  }

  Get().default_device_ = GetDevice(0, true);
#if defined(USE_GREENTEA) && defined(USE_FFT)
  Get().cl_fft_state_.setup();
#endif
}

// Should call explicitly for OCL + FFT
void Caffe::TeardownDevice(const int device_id) {
#if defined(USE_GREENTEA) &&defined(USE_FFT)
  Get().cl_fft_state_.teardown();
#endif
}

// TODO: Fix this for the new backend
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
#ifdef USE_GREENTEA
    // TODO: Complete OpenCL device information of current device
#endif  // USE_GREENTEA
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

#ifdef USE_CUDA
const char* cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  }
  return "Unknown cublas status";
}

const char* curandGetErrorString(curandStatus_t error) {
  switch (error) {
    case CURAND_STATUS_SUCCESS:
      return "CURAND_STATUS_SUCCESS";
    case CURAND_STATUS_VERSION_MISMATCH:
      return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED:
      return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED:
      return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR:
      return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE:
      return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE:
      return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH:
      return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR:
      return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "Unknown curand status";
}
#endif  // USE_CUDA

#endif  // CPU_ONLY

}  // namespace caffe
