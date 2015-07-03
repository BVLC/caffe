#include <glog/logging.h>
#include <cstdio>
#include <ctime>
#include <tuple>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/device_context.hpp"
#include "caffe/util/rng.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/cl_kernels.hpp"
#ifdef USE_CLBLAS
#include <clBLAS.h>
#endif  // USE_CLBLAS
#endif

namespace caffe {

shared_ptr<Caffe> Caffe::singleton_;

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
  seed = abs(((s * 181) * ((pid - 83) * 359)) % 104729);
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

DeviceContext *Caffe::GetDeviceContext(int id) {
  return id == -1 ? Get().default_device_context_ :
      &(Get().device_contexts_[id]);
}

DeviceContext *Caffe::GetDefaultDeviceContext() {
  return Get().default_device_context_;
}

#ifdef CPU_ONLY  // CPU-only Caffe.

Caffe::Caffe()
: random_generator_(), mode_(Caffe::CPU), default_device_context_(nullptr) {}

Caffe::~Caffe() {}

void Caffe::set_random_seed(const unsigned int seed) {
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
}

void Caffe::EnumerateDevices() {
}

class Caffe::RNG::Generator {
 public:
  Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {}
  explicit Generator(unsigned int seed) : rng_(new caffe::rng_t(seed)) {}
  caffe::rng_t* rng() {return rng_.get();}
 private:
  shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG() : generator_(new Generator()) {}

Caffe::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) {}

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
#endif  // USE_CUDA
      random_generator_(),
      mode_(Caffe::CPU),
      default_device_context_(nullptr) {
  // Try to create a cublas handler, and report an error if failed (but we will
  // keep the program running as one might just want to run CPU code).
#ifdef USE_CUDA
  if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR)<< "Cannot create Cublas handle. Cublas won't be available.";
  }
  // Try to create a curand handler.
  if (curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT)
      != CURAND_STATUS_SUCCESS ||
      curandSetPseudoRandomGeneratorSeed(curand_generator_, cluster_seedgen())
      != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "Cannot create Curand generator. Curand won't be available.";
  }
#endif  // USE_CUDA
}

Caffe::~Caffe() {
  // Make sure all device contexts and
  // dependent memory blocks are freed properly
  device_contexts_.clear();
#ifdef USE_CUDA
  if (cublas_handle_)
    CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  if (curand_generator_) {
    CURAND_CHECK(curandDestroyGenerator(curand_generator_));
  }
#endif  // USE_CUDA
}

void Caffe::set_random_seed(const unsigned int seed) {
  if (Caffe::GetDefaultDeviceContext()->backend() == BACKEND_CUDA) {
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
  DeviceContext * device_context = Caffe::GetDeviceContext(device_id);
  if (device_context->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    cudaDeviceSynchronize();
#endif
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        GetDeviceContext(device_id)->id());
    ctx.get_queue().finish();
#endif
  }
}

void Caffe::EnumerateDevices() {
  int cuda_device_count = 0;
  int greentea_device_count = 0;

#ifdef USE_CUDA
  cudaGetDeviceCount(&cuda_device_count);
#endif

#ifdef USE_GREENTEA
  typedef std::vector<viennacl::ocl::platform> platforms_type;
  platforms_type platforms = viennacl::ocl::get_platforms();

  std::vector<std::tuple<viennacl::ocl::platform,
      viennacl::ocl::device>> platform_devices;

  // Loop through devices
  for (std::size_t platform_id = 0; platform_id < platforms.size();
      ++platform_id) {
    typedef std::vector<viennacl::ocl::device> devices_type;
    devices_type devices = platforms[platform_id].devices(CL_DEVICE_TYPE_ALL);
    for (std::size_t device_id = 0; device_id < devices.size(); ++device_id) {
      platform_devices.push_back(
          std::make_tuple(platforms[platform_id], devices[device_id]));
      greentea_device_count++;
    }
  }
#endif

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

void Caffe::SetDevices(std::vector<int> device_ids) {
  Get().device_contexts_.clear();
#ifdef USE_GREENTEA
  Get().ocl_programs_.clear();
#endif
  int cuda_device_count = 0;
#ifdef USE_CUDA
  cudaGetDeviceCount(&cuda_device_count);
#endif  // USE_CUDA
  for (int i = 0; i < cuda_device_count; ++i) {
    Get().device_contexts_.emplace_back(
        DeviceContext(i, Backend::BACKEND_CUDA));
    for (int j = 0; j < device_ids.size(); ++j) {
      if (device_ids[j] == i) {
        Caffe::GetDeviceContext(i)->Init();
      }
    }
#ifdef USE_GREENTEA
    // Dummy to have same vector size as device contexts
    viennacl::ocl::program program;
    Get().ocl_programs_.push_back(program);
#endif  // USE_GREENTEA
  }

  // Initialize GreenTea devices
#ifdef USE_GREENTEA
  int greentea_device_count = 0;

  typedef std::vector<viennacl::ocl::platform> platforms_type;
  platforms_type platforms = viennacl::ocl::get_platforms();

  std::vector<std::tuple<viennacl::ocl::platform,
    viennacl::ocl::device>> platform_devices;

  // Loop through devices
  for (std::size_t platform_id = 0; platform_id < platforms.size();
      ++platform_id) {
    typedef std::vector<viennacl::ocl::device> devices_type;
    devices_type devices = platforms[platform_id].devices(CL_DEVICE_TYPE_ALL);
    for (std::size_t device_id = 0; device_id < devices.size(); ++device_id) {
      platform_devices.push_back(
          std::make_tuple(platforms[platform_id], devices[device_id]));
      Get().device_contexts_.emplace_back(
          DeviceContext(cuda_device_count + greentea_device_count,
                        Backend::BACKEND_OpenCL));
      // Check if this device is really used and initialize
      bool is_used = false;
      for (int i = 0; i < device_ids.size(); ++i) {
        int device_id = device_ids[i];
        if (device_id == cuda_device_count + greentea_device_count) {
          // Setup actual context and compile kernels for this device
          viennacl::ocl::setup_context(
              device_id, std::get<1>(platform_devices[greentea_device_count]));
          viennacl::ocl::context &ctx = viennacl::ocl::get_context(
              static_cast<uint64_t>(device_id));
          viennacl::ocl::program & program = RegisterKernels(&ctx);
          Get().ocl_programs_.push_back(program);
          // viennacl::ocl::switch_context(device_id);
          // viennacl::ocl::switch_device(std::get<1>
          // (platform_devices[device_id - cuda_device_count]));

          // Add defined number of queues
          for (int q = 0; q < GREENTEA_QUEUE_COUNT - 1; ++q) {
            ctx.add_queue(ctx.current_device());
          }
          Caffe::GetDeviceContext(device_id)->Init();
          is_used = true;
        }
      }
      // Device not used, dummy
      if (!is_used) {
        viennacl::ocl::program program;
        Get().ocl_programs_.push_back(program);
      }
      greentea_device_count++;
    }
  }
#endif  // USE_GREENTEA
}

#ifdef USE_GREENTEA
viennacl::ocl::program & Caffe::GetDeviceProgram(int id) {
  return id == -1 ? Get().default_ocl_program_ : Get().ocl_programs_[id];
}
#endif  // USE_GREENTEA

void Caffe::SetDevice(const int device_id) {
  std::vector<int> devices;
  devices.push_back(device_id);
  Caffe::SetDevices(devices);

  Get().default_device_context_ = GetDeviceContext(device_id);

  if (Get().default_device_context_->backend() == Backend::BACKEND_CUDA) {
#ifdef USE_CUDA
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    if (current_device == device_id) {
      return;
    }
// The call to cudaSetDevice must come before any calls to Get, which
// may perform initialization using the GPU.
    CUDA_CHECK(cudaSetDevice(device_id));
    if (Get().cublas_handle_)
      CUBLAS_CHECK(cublasDestroy(Get().cublas_handle_));
    if (Get().curand_generator_) {
      CURAND_CHECK(curandDestroyGenerator(Get().curand_generator_));
    }
    CUBLAS_CHECK(cublasCreate(&Get().cublas_handle_));
    CURAND_CHECK(
        curandCreateGenerator(&Get().curand_generator_,
                              CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(
        curandSetPseudoRandomGeneratorSeed(Get().curand_generator_,
                                           cluster_seedgen()));
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
#ifdef USE_CLBLAS
    clblasSetup();
#endif  // USE_CLBLAS
#endif  // USE_GREENTEA
  }
}

// TODO: Fix this for the new backend
void Caffe::DeviceQuery() {
  if (Get().default_device_context_->backend() == BACKEND_CUDA) {
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

class Caffe::RNG::Generator {
 public:
  Generator()
      : rng_(new caffe::rng_t(cluster_seedgen())) {
  }
  explicit Generator(unsigned int seed)
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

Caffe::RNG::RNG(unsigned int seed)
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
