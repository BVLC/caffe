#ifdef USE_OCL
#include <clBLAS.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "caffe/util/cl_state.hpp"
#include "caffe/util/device_alternate.hpp"

using std::find;
using std::endl;
using std::map;
using std::make_pair;
using std::pair;
using std::ostream;
using std::string;
using std::vector;

namespace caffe {

const char* clGetErrorString(cl_int error) {
  switch (error) {
  case 0: return "CL_SUCCESS";
  case -1: return "CL_DEVICE_NOT_FOUND";
  case -2: return "CL_DEVICE_NOT_AVAILABLE";
  case -3: return "CL_COMPILER_NOT_AVAILABLE";
  case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case -5: return "CL_OUT_OF_RESOURCES";
  case -6: return "CL_OUT_OF_HOST_MEMORY";
  case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case -8: return "CL_MEM_COPY_OVERLAP";
  case -9: return "CL_IMAGE_FORMAT_MISMATCH";
  case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case -11: return "CL_BUILD_PROGRAM_FAILURE";
  case -12: return "CL_MAP_FAILURE";
  case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case -15: return "CL_COMPILE_PROGRAM_FAILURE";
  case -16: return "CL_LINKER_NOT_AVAILABLE";
  case -17: return "CL_LINK_PROGRAM_FAILURE";
  case -18: return "CL_DEVICE_PARTITION_FAILED";
  case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
  case -30: return "CL_INVALID_VALUE";
  case -31: return "CL_INVALID_DEVICE_TYPE";
  case -32: return "CL_INVALID_PLATFORM";
  case -33: return "CL_INVALID_DEVICE";
  case -34: return "CL_INVALID_CONTEXT";
  case -35: return "CL_INVALID_QUEUE_PROPERTIES";
  case -36: return "CL_INVALID_COMMAND_QUEUE";
  case -37: return "CL_INVALID_HOST_PTR";
  case -38: return "CL_INVALID_MEM_OBJECT";
  case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case -40: return "CL_INVALID_IMAGE_SIZE";
  case -41: return "CL_INVALID_SAMPLER";
  case -42: return "CL_INVALID_BINARY";
  case -43: return "CL_INVALID_BUILD_OPTIONS";
  case -44: return "CL_INVALID_PROGRAM";
  case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
  case -46: return "CL_INVALID_KERNEL_NAME";
  case -47: return "CL_INVALID_KERNEL_DEFINITION";
  case -48: return "CL_INVALID_KERNEL";
  case -49: return "CL_INVALID_ARG_INDEX";
  case -50: return "CL_INVALID_ARG_VALUE";
  case -51: return "CL_INVALID_ARG_SIZE";
  case -52: return "CL_INVALID_KERNEL_ARGS";
  case -53: return "CL_INVALID_WORK_DIMENSION";
  case -54: return "CL_INVALID_WORK_GROUP_SIZE";
  case -55: return "CL_INVALID_WORK_ITEM_SIZE";
  case -56: return "CL_INVALID_GLOBAL_OFFSET";
  case -57: return "CL_INVALID_EVENT_WAIT_LIST";
  case -58: return "CL_INVALID_EVENT";
  case -59: return "CL_INVALID_OPERATION";
  case -60: return "CL_INVALID_GL_OBJECT";
  case -61: return "CL_INVALID_BUFFER_SIZE";
  case -62: return "CL_INVALID_MIP_LEVEL";
  case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
  case -64: return "CL_INVALID_PROPERTY";
  case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
  case -66: return "CL_INVALID_COMPILER_OPTIONS";
  case -67: return "CL_INVALID_LINKER_OPTIONS";
  case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
  case -69: return "CL_INVALID_PIPE_SIZE";
  case -70: return "CL_INVALID_DEVICE_QUEUE";
  case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
  case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
  case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
  case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
  case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
  case -1024: return "clBLAS: Functionality is not implemented";
  case -1023: return "clBLAS: Library is not initialized yet";
  case -1022: return "clBLAS: Matrix A is not a valid memory object";
  case -1021: return "clBLAS: Matrix B is not a valid memory object";
  case -1020: return "clBLAS: Matrix C is not a valid memory object";
  case -1019: return "clBLAS: Vector X is not a valid memory object";
  case -1018: return "clBLAS: Vector Y is not a valid memory object";
  case -1017: return "clBLAS: An input dimension (M:N:K) is invalid";
  case -1016: return "clBLAS: Leading dimension A must not be less than the "
      "size of the first dimension";
  case -1015: return "clBLAS: Leading dimension B must not be less than the "
      "size of the second dimension";
  case -1014: return "clBLAS: Leading dimension C must not be less than the "
      "size of the third dimension";
  case -1013: return "clBLAS: The increment for a vector X must not be 0";
  case -1012: return "clBLAS: The increment for a vector Y must not be 0";
  case -1011: return "clBLAS: The memory object for Matrix A is too small";
  case -1010: return "clBLAS: The memory object for Matrix B is too small";
  case -1009: return "clBLAS: The memory object for Matrix C is too small";
  case -1008: return "clBLAS: The memory object for Vector X is too small";
  case -1007: return "clBLAS: The memory object for Vector Y is too small";
  default: return "Unknown OpenCL error";
  }
}

#ifdef USE_FFT
const char* clfftGetErrorString(clfftStatus status) {
  switch (status) {
  case CLFFT_SUCCESS:
    return "CLFFT_SUCCESS";
  case CLFFT_INVALID_PLAN:
    return "CLFFT_INVALID_PLAN";
  case CLFFT_INVALID_GLOBAL_WORK_SIZE:
    return "CLFFT_INVALID_GLOBAL_WORK_SIZE";
  case CLFFT_INVALID_MIP_LEVEL:
    return "CLFFT_INVALID_MIP_LEVEL";
  case CLFFT_INVALID_BUFFER_SIZE:
    return "CLFFT_INVALID_BUFFER_SIZE";
  case CLFFT_INVALID_GL_OBJECT:
    return "CLFFT_INVALID_GL_OBJECT";
  case CLFFT_INVALID_OPERATION:
    return "CLFFT_INVALID_OPERATION";
  case CLFFT_INVALID_EVENT:
    return "CLFFT_INVALID_EVENT";
  case CLFFT_INVALID_EVENT_WAIT_LIST:
    return "CLFFT_INVALID_EVENT_WAIT_LIST";
  case CLFFT_INVALID_GLOBAL_OFFSET:
    return "CLFFT_INVALID_GLOBAL_OFFSET";
  case CLFFT_INVALID_WORK_ITEM_SIZE:
    return "CLFFT_INVALID_WORK_ITEM_SIZE";
  case CLFFT_INVALID_WORK_GROUP_SIZE:
    return "CLFFT_INVALID_WORK_GROUP_SIZE";
  case CLFFT_INVALID_WORK_DIMENSION:
    return "CLFFT_INVALID_WORK_DIMENSION";
  case CLFFT_INVALID_KERNEL_ARGS:
    return "CLFFT_INVALID_KERNEL_ARGS";
  case CLFFT_INVALID_ARG_SIZE:
    return "CLFFT_INVALID_ARG_SIZE";
  case CLFFT_INVALID_ARG_VALUE:
    return "CLFFT_INVALID_ARG_VALUE";
  case CLFFT_INVALID_ARG_INDEX:
    return "CLFFT_INVALID_ARG_INDEX";
  case CLFFT_INVALID_KERNEL:
    return "CLFFT_INVALID_KERNEL";
  case CLFFT_INVALID_KERNEL_DEFINITION:
    return "CLFFT_INVALID_KERNEL_DEFINITION";
  case CLFFT_INVALID_KERNEL_NAME:
    return "CLFFT_INVALID_KERNEL_NAME";
  case CLFFT_INVALID_PROGRAM_EXECUTABLE:
    return "CLFFT_INVALID_PROGRAM_EXECUTABLE";
  case CLFFT_INVALID_PROGRAM:
    return "CLFFT_INVALID_PROGRAM";
  case CLFFT_INVALID_BUILD_OPTIONS:
    return "CLFFT_INVALID_BUILD_OPTIONS";
  case CLFFT_INVALID_BINARY:
    return "CLFFT_INVALID_BINARY";
  case CLFFT_INVALID_SAMPLER:
    return "CLFFT_INVALID_SAMPLER";
  case CLFFT_INVALID_IMAGE_SIZE:
    return "CLFFT_INVALID_IMAGE_SIZE";
  case CLFFT_INVALID_IMAGE_FORMAT_DESCRIPTOR:
    return "CLFFT_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case CLFFT_INVALID_MEM_OBJECT:
    return "CLFFT_INVALID_MEM_OBJECT";
  case CLFFT_INVALID_HOST_PTR:
    return "CLFFT_INVALID_HOST_PTR";
  case CLFFT_INVALID_COMMAND_QUEUE:
    return "CLFFT_INVALID_COMMAND_QUEUE";
  case CLFFT_INVALID_QUEUE_PROPERTIES:
    return "CLFFT_INVALID_QUEUE_PROPERTIES";
  case CLFFT_INVALID_CONTEXT:
    return "CLFFT_INVALID_CONTEXT";
  case CLFFT_INVALID_DEVICE:
    return "CLFFT_INVALID_DEVICE";
  case CLFFT_INVALID_PLATFORM:
    return "CLFFT_INVALID_PLATFORM";
  case CLFFT_INVALID_DEVICE_TYPE:
    return "CLFFT_INVALID_DEVICE_TYPE";
  case CLFFT_INVALID_VALUE:
    return "CLFFT_INVALID_VALUE";
  case CLFFT_MAP_FAILURE:
    return "CLFFT_MAP_FAILURE";
  case CLFFT_BUILD_PROGRAM_FAILURE:
    return "CLFFT_BUILD_PROGRAM_FAILURE";
  case CLFFT_IMAGE_FORMAT_NOT_SUPPORTED:
    return "CLFFT_IMAGE_FORMAT_NOT_SUPPORTED";
  case CLFFT_IMAGE_FORMAT_MISMATCH:
    return "CLFFT_IMAGE_FORMAT_MISMATCH";
  case CLFFT_MEM_COPY_OVERLAP:
    return "CLFFT_MEM_COPY_OVERLAP";
  case CLFFT_PROFILING_INFO_NOT_AVAILABLE:
    return "CLFFT_PROFILING_INFO_NOT_AVAILABLE";
  case CLFFT_OUT_OF_HOST_MEMORY:
    return "CLFFT_OUT_OF_HOST_MEMORY";
  case CLFFT_OUT_OF_RESOURCES:
    return "CLFFT_OUT_OF_RESOURCES";
  case CLFFT_MEM_OBJECT_ALLOCATION_FAILURE:
    return "CLFFT_MEM_OBJECT_ALLOCATION_FAILURE";
  case CLFFT_COMPILER_NOT_AVAILABLE:
    return "CLFFT_COMPILER_NOT_AVAILABLE";
  case CLFFT_DEVICE_NOT_AVAILABLE:
    return "CLFFT_DEVICE_NOT_AVAILABLE";
  case CLFFT_DEVICE_NOT_FOUND:
    return "CLFFT_DEVICE_NOT_FOUND";
  case CLFFT_BUGCHECK:
    return "CLFFT_BUGCHECK";
  case CLFFT_NOTIMPLEMENTED:
    return "CLFFT_NOTIMPLEMENTED";
  case CLFFT_TRANSPOSED_NOTIMPLEMENTED:
    return "CLFFT_TRANSPOSED_NOTIMPLEMENTED";
  case CLFFT_FILE_NOT_FOUND:
    return "CLFFT_FILE_NOT_FOUND";
  case CLFFT_FILE_CREATE_FAILURE:
    return "CLFFT_FILE_CREATE_FAILURE";
  case CLFFT_VERSION_MISMATCH:
    return "CLFFT_VERSION_MISMATCH";
  case CLFFT_DEVICE_NO_DOUBLE:
    return "CLFFT_DEVICE_NO_DOUBLE";
  case CLFFT_DEVICE_MISMATCH:
    return "CLFFT_DEVICE_MISMATCH";
  default:
    return "CLFFT_UNKNOWN_ERROR";
  }
}
#endif  // USE FFT

extern bool clkernel_submit_program(const char* name,
    const std::vector< std::pair<const char*, const char*> >& program_srcs,
    const char* options, bool no_check);
extern void clkernel_release_program(const char* name);
extern int clkernel_get_num_programs();
extern void clkernel_destroy_kernels();
extern ClKernel& clkernel_get_kernel(const char* name);

struct ClState::Impl {
  explicit Impl(int device_idx) {
    cl_uint num_platforms;
    OCL_CHECK(clGetPlatformIDs(0, NULL, &num_platforms));

    vector<cl_platform_id> platform_ids(num_platforms);
    OCL_CHECK(clGetPlatformIDs(num_platforms, platform_ids.data(), NULL));

    vector<cl_device_id> gpu_device_ids;
    for (vector<cl_platform_id>::iterator it = platform_ids.begin();
         it != platform_ids.end(); ++it) {
      cl_uint num_devices;
      OCL_CHECK(clGetDeviceIDs(*it, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices));

      vector<cl_device_id> device_ids(num_devices);
      OCL_CHECK(clGetDeviceIDs(*it, CL_DEVICE_TYPE_ALL, num_devices,
                               device_ids.data(), NULL));

      gpu_device_ids.insert(gpu_device_ids.end(), device_ids.begin(),
                            device_ids.end());
    }

    if (device_idx >= gpu_device_ids.size()) {
      LOG(ERROR) << "Invalid OCL GPU device index.";
    } else {
      cl_int err_code;

      device_id_ = gpu_device_ids[device_idx];

      context_ =
        clCreateContext(NULL, 1, &device_id_, NULL, NULL, &err_code);
      OCL_CHECK(err_code);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
      command_queue_ = clCreateCommandQueue(context_, device_id_,
          CL_QUEUE_PROFILING_ENABLE, &err_code);
#pragma GCC diagnostic pop
      OCL_CHECK(err_code);

      char extensions[2048];
      OCL_CHECK(clGetDeviceInfo(device_id_, CL_DEVICE_EXTENSIONS,
        sizeof extensions, extensions, NULL));
      fp64_supported_ = strstr(extensions, "cl_khr_fp64") != NULL;

      cl_bool unifiedMem;
      OCL_CHECK(clGetDeviceInfo(device_id_, CL_DEVICE_HOST_UNIFIED_MEMORY,
          sizeof unifiedMem, &unifiedMem, NULL));
      unified_mem_ = unifiedMem;
    }

    init_properties();

    // clBLAS setup
    OCL_CHECK(clblasSetup());

    free_mem_[NULL] = static_cast<size_t>(1) << (sizeof (size_t) * 8 - 1);
  }

  ~Impl() {
    clkernel_destroy_kernels();

    // Finalize work with clBLAS
    clblasTeardown();

    if (!memobjs_.empty())
      LOG(ERROR) << "Cl buffers not released!";

    if (command_queue_ != NULL) {
      clReleaseCommandQueue(command_queue_);
      command_queue_ = NULL;
    }

    if (context_ != NULL) {
      clReleaseContext(context_);
      context_ = NULL;
    }
  }

  const void init_properties() {
    properties_.name = getDeviceInfoString(CL_DEVICE_NAME);
    properties_.vendor_id = getDeviceInfoVal<unsigned int>(CL_DEVICE_VENDOR_ID);
    properties_.vendor = getDeviceInfoString(CL_DEVICE_VENDOR);
    properties_.version = getDeviceInfoString(CL_DEVICE_VERSION);
    properties_.profile = getDeviceInfoString(CL_DEVICE_PROFILE);
    properties_.type = getDeviceInfoVal<cl_device_type>(CL_DEVICE_TYPE);
    properties_.driver_version = getDeviceInfoString(CL_DRIVER_VERSION);
    properties_.extensions = getDeviceInfoMultiLine(CL_DEVICE_EXTENSIONS);
    properties_.available = getDeviceInfoVal<bool>(CL_DEVICE_AVAILABLE);
    properties_.compiler_available = getDeviceInfoVal<bool>(
      CL_DEVICE_COMPILER_AVAILABLE);
    properties_.little_endian = getDeviceInfoVal<bool>(
      CL_DEVICE_ENDIAN_LITTLE);
    properties_.single_fp_config = getDeviceInfoVal<cl_device_fp_config>(
      CL_DEVICE_SINGLE_FP_CONFIG);
    properties_.double_fp_config = getDeviceInfoVal<cl_device_fp_config>(
      CL_DEVICE_DOUBLE_FP_CONFIG);
    properties_.err_correction_support = getDeviceInfoVal<bool>(
      CL_DEVICE_ERROR_CORRECTION_SUPPORT);
    properties_.max_clock_freq = getDeviceInfoVal<unsigned int>(
      CL_DEVICE_MAX_CLOCK_FREQUENCY);
    properties_.profile_timer_res = getDeviceInfoVal<unsigned int>(
      CL_DEVICE_PROFILING_TIMER_RESOLUTION);
    properties_.mem_base_addr_align = getDeviceInfoVal<unsigned int>(
        CL_DEVICE_MEM_BASE_ADDR_ALIGN);
    properties_.local_mem_size = getDeviceInfoVal<unsigned long>(
        CL_DEVICE_LOCAL_MEM_SIZE);
    properties_.device_max_work_group_size = getDeviceInfoVal<unsigned long>(
        CL_DEVICE_MAX_WORK_GROUP_SIZE);
  }

  template<typename T>
  T getDeviceInfoVal(cl_device_info info) {
    T value;
    clGetDeviceInfo(device_id_, info, sizeof value, &value, NULL);
    return value;
  }

  string getDeviceInfoString(cl_device_info info) {
    char value[2048];
    clGetDeviceInfo(device_id_, info, sizeof value, value, NULL);
    return string(value);
  }

  vector<string> getDeviceInfoMultiLine(cl_device_info info,
    char separator=' ') {
    char value[2048];
    clGetDeviceInfo(device_id_, info, sizeof value, value, NULL);

    vector<string> lines;
    std::stringstream ss(value);
    string item;
    while (std::getline(ss, item, separator))
      lines.push_back(string(item));
    return lines;
  }

  vector<string> getDeviceInfoEnumAsMultiLine(cl_device_info info,
    const char* names[]) {
    cl_bitfield value;
    clGetDeviceInfo(device_id_, info, sizeof value, &value, NULL);

    vector<string> lines;
    for (uint64_t i = 0, bit = 1; i < 64; ++i, bit <<= 1) {
      if (value & bit)
        lines.push_back(string(names[i]));
    }
    return lines;
  }

  void print_properties(ostream& out) {
    const char* types[] = {"Default", "CPU", "GPU", "Accelerator", "Custom"};
    const char* fp[] = {"Denorm", "Inf NaN", "Round to Nearest",
                        "Round to Zero", "Round to Inf", "FMA", "Soft Float",
                        "Correctly Rounded Div and Sqrt"};

    out << endl;
    out << "Name:               " << properties_.name << endl;
    out << "Vendor ID:          " << properties_.vendor_id << endl;
    out << "Vendor:             " << properties_.vendor << endl;
    out << "Version:            " << properties_.version << endl;
    out << "Profile:            " << properties_.profile << endl;
    printEnumAsMultiline(out, properties_.type, "Type:               ", types);
    out << "Driver Version:     " << properties_.driver_version << endl;
    printMultiline(out, properties_.extensions, "Extensions:         ");
    printBool(out, properties_.available, "Available:          ");
    printBool(out, properties_.compiler_available, "Compiler Available: ");
    printBool(out, properties_.little_endian, "Little Endian:      ");
    printEnumAsMultiline(out, properties_.single_fp_config,
      "Single FP Caps:     ", fp);
    printEnumAsMultiline(out, properties_.double_fp_config,
      "Double FP Caps:     ", fp);
    printBool(out, properties_.err_correction_support, "Error Correction:   ");
    out << "Max Clock Freq:     " << properties_.max_clock_freq << "MHz"
      << endl;
    out << "Profile Timer Res:  " << properties_.profile_timer_res << "ns"
      << endl;
    out << "Local Mem size:     " << properties_.local_mem_size << endl;
    out << "Device max work group size: " <<
        properties_.device_max_work_group_size << endl;
  }

  void printBool(ostream& out, bool value, const char* title) {
    out << title << (value ? "True" : "False") << endl;
  }

  void printMultiline(ostream& out, const vector<string>& values,
    const char* title) {
    string ttl = title;
    string blanks(ttl.length(), ' ');
    for (vector<string>::const_iterator it = values.begin();
         it != values.end(); ++it) {
      out << ttl << *it << endl;
      ttl = blanks;
    }
  }

  void printEnumAsMultiline(ostream& out, cl_bitfield value, const char* title,
    const char* names[]) {
    string ttl(title);
    string blanks(ttl.length(), ' ');
    for (uint64_t i = 0, bit = 1; i < 64; ++i, bit <<=1) {
      if (value & bit) {
        out << ttl << names[i] << endl;
        ttl = blanks;
      }
    }
    if (ttl != blanks)
      out << ttl << "(none)" << endl;
  }

  void* create_buffer(cl_mem_flags flags, size_t size, void* host_ptr) {
    cl_int errcode;
    if (unified_mem_ && host_ptr == NULL)
      flags |= CL_MEM_ALLOC_HOST_PTR;
    cl_mem memobj = clCreateBuffer(context_, flags, size, host_ptr, &errcode);

    if (errcode == CL_MEM_OBJECT_ALLOCATION_FAILURE) {
      LOG(INFO) << "clCreateBuffer() failed to allocate %d bytes memory "
                << static_cast<int>(size) << " for buffer object...";
    } else if (errcode == CL_OUT_OF_HOST_MEMORY) {
      LOG(INFO) << "clCreateBuffer() failed to allocate %d bytes memory "
                << static_cast<int>(size)
                << " and resources required by OCL on the host...";
    }
    OCL_CHECK(errcode);

    void* buffer = get_memptr_from_freemem(size);
    memobjs_[buffer] = make_pair(memobj, size);

    return buffer;
  }

  void* get_memptr_from_freemem(size_t size) {
    map<void*, size_t>::iterator it = find_best_fit_free_mem(size);

    void* memptr = static_cast<char*>(it->first) - size;

    if (it->second > size)
      free_mem_[memptr] = it->second - size;
    free_mem_.erase(it);

    return memptr;
  }

  map<void*, size_t>::iterator find_best_fit_free_mem(size_t size) {
    map<void*, size_t>::iterator fit = free_mem_.end();
    for (map<void*, size_t>::iterator it = free_mem_.begin();
         it != free_mem_.end(); ++it) {
      if (it->second >= size &&
          (fit == free_mem_.end() || it->second < fit->second))
        fit = it;
    }

    if (fit == free_mem_.end())
      LOG(FATAL) << "Unable to find free memory";

    return fit;
  }

  void destroy_buffer(void* buffer) {
    map<void*, pair<cl_mem, size_t> >::iterator it = memobjs_.find(buffer);
    if (it == memobjs_.end())
      LOG(FATAL) << "Invalid buffer";

    cl_mem mem = it->second.first;
    int size = it->second.second;
    memobjs_.erase(it);
    free_mem_[static_cast<char*>(buffer) + size] = size;

    combine_free_mem();

    OCL_CHECK(clReleaseMemObject(mem));
  }

  void combine_free_mem() {
    for (size_t prevSize = 0; free_mem_.size() != prevSize;) {
      prevSize = free_mem_.size();

      for (map<void*, size_t>::iterator it = free_mem_.begin();
           it != free_mem_.end(); ++it) {
        map<void*, size_t>::iterator it2 = it;
        ++it2;

        if (it2 == free_mem_.end())
          break;

        if (it->first == NULL) {
          if (static_cast<char*>(it2->first) + it->second == NULL) {
            it->second += it2->second;
            free_mem_.erase(it2);
            break;
          }
        } else if (static_cast<char*>(it->first) + it2->second == it2->first) {
          it2->second += it->second;
          free_mem_.erase(it);
          break;
        }
      }
    }
  }

  size_t get_buffer_size(const void* buffer) {
    map<void*, pair<cl_mem, size_t> >::iterator it =
        memobjs_.find(const_cast<void*>(buffer));
    if (it == memobjs_.end())
      LOG(FATAL) << "Invalid buffer object";
    return it->second.second;
  }

  ClMemOff<uint8_t> get_buffer_mem(const void* ptr) {
    const char* cptr = static_cast<const char*>(ptr);
    for (map<void*, pair<cl_mem, size_t> >::iterator it = memobjs_.begin();
         it != memobjs_.end(); ++it) {
      const char* buffer = static_cast<char*>(it->first);
      cl_mem mem = it->second.first;
      int size = it->second.second;

      if (cptr >= buffer && (cptr - buffer) < size)
        return ClMemOff<uint8_t>{mem, static_cast<size_t>(cptr - buffer)};
    }

    return ClMemOff<uint8_t>{NULL, 0};
  }

  ClDeviceProperties properties_;

  cl_device_id device_id_;
  cl_context context_;
  cl_command_queue command_queue_;
  bool fp64_supported_;
  bool unified_mem_;
  map<void*, pair<cl_mem, size_t> > memobjs_;
  map<void*, size_t> free_mem_;
};

ClState::ClState()
  : impl_(NULL) {
}

ClState::~ClState() {
  if (impl_ != NULL)
    delete impl_;
}

bool ClState::is_device_set() { return impl_ != NULL; }

void ClState::set_device(int device_idx) {
  if (impl_ != NULL)
    delete impl_;
  impl_ = new Impl(device_idx);
}



cl_device_id ClState::get_device() {
  if (impl_ == NULL)
    return NULL;
  return impl_->device_id_;
}

cl_context ClState::get_context() {
  if (impl_ == NULL)
    return NULL;
  return impl_->context_;
}

cl_command_queue ClState::get_command_queue() {
  if (impl_ == NULL)
    return NULL;
  return impl_->command_queue_;
}

const ClDeviceProperties& ClState::get_properties() {
  return impl_->properties_;
}

void ClState::print_properties(std::ostream& out) {
  impl_->print_properties(out);
}

bool ClState::fp64_supported() {
  if (impl_ == NULL)
    return false;
  return impl_->fp64_supported_;
}

void* ClState::create_buffer(cl_mem_flags flags, size_t size, void* host_ptr) {
  return impl_->create_buffer(flags, size, host_ptr);
}

void ClState::destroy_buffer(void* buffer) {
  impl_->destroy_buffer(buffer);
}

size_t ClState::get_buffer_size(const void* buffer) {
  return impl_->get_buffer_size(buffer);
}

ClMemOff<uint8_t> ClState::get_buffer_mem(const void* ptr) {
  return impl_->get_buffer_mem(ptr);
}

cl_mem ClState::create_subbuffer(const void* ptr, size_t offset,
    cl_mem_flags flags) {
  ClMemOff<uint8_t> buf = get_buffer_mem(ptr);
  size_t size = get_buffer_size(static_cast<const char*>(ptr) - buf.offset);
  cl_buffer_region bufReg = { offset, size - offset };
  cl_int err;
  cl_mem sub_buf = clCreateSubBuffer(buf.memobj, flags,
      CL_BUFFER_CREATE_TYPE_REGION, &bufReg, &err);
  OCL_CHECK(err);
  return sub_buf;
}

bool ClState::submit_program(const char* name, const char* program_src_start,
    const char* program_src_end, const char* options, bool no_check) {
  std::vector< std::pair<const char*, const char*> > program_srcs;
  program_srcs.push_back(make_pair(program_src_start, program_src_end));
  return clkernel_submit_program(name, program_srcs, options, no_check);
}

bool ClState::submit_program(const char* name,
    const std::vector< std::pair<const char*, const char*> >& program_srcs,
    const char* options, bool no_check) {
  return clkernel_submit_program(name, program_srcs, options, no_check);
}

void ClState::release_program(const char* name) {
  clkernel_release_program(name);
}

int ClState::get_num_programs() { return clkernel_get_num_programs(); }

ClKernel& ClState::get_kernel(const char * name) {
  return clkernel_get_kernel(name);
}

}  // namespace caffe
#endif  // USE_OCL
