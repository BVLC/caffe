#include <memory>

#include "caffe/backend/opencl/ocl_device.hpp"
#include "caffe/backend/opencl/ocl_device_program.hpp"
#include "caffe/backend/opencl/caffe_opencl.hpp"
#include "caffe/backend/opencl/ocl_dev_ptr.hpp"

namespace caffe {

#ifdef USE_OPENCL

OclDevice::OclDevice(uint_tp id, uint_tp list_id) {
  this->current_queue_id_ = 0;
  this->max_local_sizes_ = vector<size_t>(3, 0);
  this->max_group_sizes_ = vector<size_t>(3, 0);
  this->id_ = id;
  this->list_id_ = list_id;
  this->backend_ = BACKEND_OPENCL;
  this->memory_usage_ = 0;
  this->peak_memory_usage_ = 0;
  this->host_unified_ = false;
  this->name_ = "";
}

OclDevice::~OclDevice() {
  buffers_.clear();
}

void OclDevice::Init() {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(id_);
  {
    clGetDeviceInfo(ctx.devices()[0].id(),
                    CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    1 * sizeof(size_t), &max_local_size_, NULL);
  }

  {
    vector<size_t> temp(3);
    clGetDeviceInfo(ctx.devices()[0].id(),
                    CL_DEVICE_MAX_WORK_ITEM_SIZES,
                    3 * sizeof(size_t), &temp[0], NULL);
    max_local_sizes_[0] = std::min(temp[0], max_local_size_);
    max_local_sizes_[1] = std::min(temp[1], max_local_size_);
    max_local_sizes_[2] = std::min(temp[2], max_local_size_);
  }

   max_group_sizes_[0] = SIZE_MAX;
   max_group_sizes_[1] = SIZE_MAX;
   max_group_sizes_[2] = SIZE_MAX;


   {
#ifdef DISABLE_DEVICE_HOST_UNIFIED_MEMORY
    host_unified_ = false;
    LOG(INFO) << "CL_DEVICE_HOST_UNIFIED_MEMORY: disabled";
#else
    cl_bool host_unified;
    clGetDeviceInfo(ctx.devices()[0].id(),
                    CL_DEVICE_HOST_UNIFIED_MEMORY,
                    sizeof(cl_bool), &host_unified, NULL);
    LOG(INFO) << "CL_DEVICE_HOST_UNIFIED_MEMORY: " << host_unified;
    host_unified_ = host_unified
        || ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU;
#endif  // DISABLE_DEVICE_HOST_UNIFIED_MEMORY
   }

  for (int q = 0; q < OPENCL_QUEUE_COUNT - 1; ++q) {
    ctx.add_queue(ctx.devices()[0]);
  }

  {
    cl_uint vec_size;
    clGetDeviceInfo(ctx.devices()[0].id(),
                    CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(cl_uint),
                    &vec_size, NULL);
    this->preferred_vector_widths_[safe_type_name<char>()] = vec_size;
    this->preferred_vector_widths_[safe_type_name<int8_t>()] = vec_size;
    this->preferred_vector_widths_[safe_type_name<uint8_t>()] = vec_size;

    clGetDeviceInfo(ctx.devices()[0].id(),
                    CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(cl_uint),
                    &vec_size, NULL);
    this->preferred_vector_widths_[safe_type_name<int16_t>()] = vec_size;
    this->preferred_vector_widths_[safe_type_name<uint16_t>()] = vec_size;

    clGetDeviceInfo(ctx.devices()[0].id(),
                    CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(cl_uint),
                    &vec_size, NULL);
    this->preferred_vector_widths_[safe_type_name<int32_t>()] = vec_size;
    this->preferred_vector_widths_[safe_type_name<uint32_t>()] = vec_size;

    clGetDeviceInfo(ctx.devices()[0].id(),
                    CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof(cl_uint),
                    &vec_size, NULL);
    this->preferred_vector_widths_[safe_type_name<int64_t>()] = vec_size;
    this->preferred_vector_widths_[safe_type_name<uint64_t>()] = vec_size;

    clGetDeviceInfo(ctx.devices()[0].id(),
                    CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, sizeof(cl_uint),
                    &vec_size, NULL);
    this->preferred_vector_widths_[safe_type_name<half_fp>()] = vec_size;

    clGetDeviceInfo(ctx.devices()[0].id(),
                    CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(cl_uint),
                    &vec_size, NULL);
    this->preferred_vector_widths_[safe_type_name<float>()] = vec_size;

    clGetDeviceInfo(ctx.devices()[0].id(),
                    CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint),
                    &vec_size, NULL);
    this->preferred_vector_widths_[safe_type_name<double>()] = vec_size;

  }

  Device::Init();

  this->CreateMathProgram();
  this->CreateIm2ColProgram();
}

shared_ptr<DeviceProgram> OclDevice::CreateProgram() {
  return make_shared<OclDeviceProgram>(this);
}

void OclDevice::unlock_buffer(int_tp* lock_id) {
  // Make sure the buffer is no longer in use
  // TODO: Only flush related queue(s) (?)
  FinishQueues();
  Device::unlock_buffer(lock_id);
}

string OclDevice::name() {
  if (name_ == "") {
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(id_);

    size_t size;
    size_t max_size = 1024 * 1024;
    clGetDeviceInfo(ctx.devices()[0].id(), CL_DEVICE_NAME,
                    0, NULL, &size);

    // Cap at 1 MB to capture faulty OpenCL implementations (nVidia)
    vector<char> exts(std::min(size, max_size));

    clGetDeviceInfo(ctx.devices()[0].id(), CL_DEVICE_NAME,
                    std::min(size, max_size), &(exts[0]), NULL);

    string extsstr(&(exts[0]));
    std::replace(extsstr.begin(), extsstr.end(), ' ', '_');
    name_ = extsstr;
  }
  return name_;
}

void OclDevice::MallocMemHost(uint_tp size, void** ptr) {
#ifdef USE_MKL
  *ptr = mkl_malloc(size ? size : 1, 64);
#else
#ifdef _MSC_VER
  *ptr = malloc(((size - 1) / CAFFE_MALLOC_CACHE_ALIGN + 1)
                * CAFFE_MALLOC_CACHE_ALIGN);
#else
  CHECK_EQ(0, posix_memalign(ptr, CAFFE_MALLOC_PAGE_ALIGN,
           ((size - 1) / CAFFE_MALLOC_CACHE_ALIGN + 1)
           * CAFFE_MALLOC_CACHE_ALIGN))
              << "Host memory allocation error of size: "
              << size << " b";
#endif  // _MSC_VER
#endif  // USE_MKL
  CHECK(*ptr) << "Host allocation of size " << size << " failed";
}

void OclDevice::FreeMemHost(void* ptr) {
#ifdef USE_MKL
  mkl_free(ptr);
#else
  free(ptr);
#endif  // USE_MKL
}

vptr<void> OclDevice::MallocMemDevice(uint_tp size, void** ptr,
                                               bool zero_copy) {
  CHECK_GT(size, 0) << "Illegal allocation of size 0.";

  cl_mem gpu_ptr;
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());
  cl_int err;
  if (zero_copy) {
    uint_tp zero_copy_size = (size + CAFFE_MALLOC_CACHE_ALIGN - 1)
                            & ~(CAFFE_MALLOC_CACHE_ALIGN - 1);
    this->MallocMemHost(zero_copy_size, ptr);
    gpu_ptr = clCreateBuffer(ctx.handle().get(),
                      CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                      zero_copy_size, *ptr, &err);

    OCL_CHECK(err);

    void *mapped_ptr = clEnqueueMapBuffer(ctx.get_queue().handle().get(),
                                          gpu_ptr, true,
                                          CL_MAP_READ | CL_MAP_WRITE,
                                          0, size, 0, NULL, NULL, NULL);
    CHECK_EQ(mapped_ptr, *ptr)
      << "Device claims it support zero copy"
      << " but failed to create correct user ptr buffer";
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(),
                            gpu_ptr, mapped_ptr, 0, NULL, NULL);
  } else {
    gpu_ptr = clCreateBuffer(ctx.handle().get(),
                                 CL_MEM_READ_WRITE,
                                 size, nullptr, &err);
  }
  CHECK_EQ(0, err) << "OpenCL buffer allocation of size "
                   << size << " failed.";
  return vptr<void>(make_shared<ocl_dev_ptr<void> >(gpu_ptr));
}

void OclDevice::FreeMemDevice(vptr<void> ptr) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());
  ctx.get_queue().finish();
  CHECK_EQ(CL_SUCCESS, clReleaseMemObject(ptr.get_ocl_mem()))
      << "OpenCL memory corruption";
  ctx.get_queue().finish();
}

bool OclDevice::CheckZeroCopy(vptr<const void> gpu_ptr, void* cpu_ptr,
                                       uint_tp size) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());
  void *mapped_ptr = clEnqueueMapBuffer(ctx.get_queue().handle().get(),
                        gpu_ptr.get_ocl_mem(), true, CL_MAP_READ | CL_MAP_WRITE,
                        0, size, 0, NULL, NULL, NULL);
  CHECK_EQ(mapped_ptr, cpu_ptr)
    << "Device claims it support zero copy"
    << " but failed to create correct user ptr buffer";
  bool zero_copy_result = (mapped_ptr == cpu_ptr);
  clEnqueueUnmapMemObject(ctx.get_queue().handle().get(),
                          gpu_ptr.get_ocl_mem(),
                          mapped_ptr, 0, NULL, NULL);
  ctx.get_queue().finish();
  return zero_copy_result;
}


uint_tp OclDevice::num_queues() {
  return OPENCL_QUEUE_COUNT;
}

void OclDevice::get_threads(const vector<size_t>* work_size,
                            vector<size_t>* group,
                            vector<size_t>* local,
                            DeviceKernel* kernel,
                            bool auto_select) {
  CHECK(work_size);
  CHECK(group);
  CHECK(local);
  CHECK(kernel);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(id_);

  // Let the OpenCL implementation choose sizes
  if (auto_select) {
    for(uint_tp i = 0; i < work_size->size(); ++i) {
      local->insert(local->begin() + i, 0);
      group->insert(group->begin() + i, (*work_size)[i]);
    }
    return;
  } else {
    for(uint_tp i = 0; i < work_size->size(); ++i) {
      local->insert(local->begin() + i, 1);
      group->insert(group->begin() + i, 1);
    }
  }

  size_t local_size_multiple_kernel = 1;
  size_t max_local_size_kernel = 1;
  vector<size_t> max_global_sizes_kernel(3, 1);

  // Figure out what the kernel allows according to the OpenCL implementation
  if (OclDeviceKernel* const ocl_dev_kernel
                          = dynamic_cast<OclDeviceKernel*>(kernel)) {
    cl_kernel ocl_kernel = ocl_dev_kernel->get_ocl_kernel().handle().get();
    // Get OpenCL estimate on kernel's allowance for work group sizes
    clGetKernelWorkGroupInfo(ocl_kernel, ctx.devices()[0].id(),
                             CL_KERNEL_GLOBAL_WORK_SIZE,
                             3 * sizeof(size_t),
                             &max_global_sizes_kernel[0],
                             NULL);
    clGetKernelWorkGroupInfo(ocl_kernel, ctx.devices()[0].id(),
                             CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                             sizeof(size_t),
                             &local_size_multiple_kernel,
                             NULL);
    clGetKernelWorkGroupInfo(ocl_kernel, ctx.devices()[0].id(),
                             CL_KERNEL_WORK_GROUP_SIZE,
                             sizeof(size_t),
                             &max_local_size_kernel,
                             NULL);
  }

  bool done = false;
  vector<bool> local_done(work_size->size(), false);
  while (!done) {
    done = true;
    for (uint_tp i = 0; i < work_size->size(); ++i) {
      done = done && local_done[i];
    }
    for (uint_tp i = 0; i < work_size->size(); ++i) {
      if (!done
          && ((*local)[i] <= (*work_size)[i])
          && ((*local)[i] * 2 <= max_local_sizes_[i])) {
        (*local)[i] *= 2;
      } else {
        local_done[i] = true;
      }
      size_t total_local_size = 1;
      for (uint_tp j = 0; j < work_size->size(); ++j) {
        total_local_size *= (*local)[j];
      }
      if (total_local_size > max_local_size_) {
        (*local)[i] /= 2;
        done = true;
      }
    }
  }

  for (uint_tp i = 0; i < work_size->size(); ++i) {
    (*group)[i] = ((*work_size)[i] - 1) / ((*local)[i]) + 1;
  }
}

bool OclDevice::CheckVendor(string vendor) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(id_);
  const viennacl::ocl::device &device = ctx.current_device();

  if (device.vendor().find(vendor) != string::npos) {
      return true;
  }
  return false;
}

bool OclDevice::CheckCapability(DeviceCapability cap) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(id_);

  size_t size;
  size_t max_size = 1024 * 1024;
  clGetDeviceInfo(ctx.devices()[0].id(), CL_DEVICE_EXTENSIONS,
                  0, NULL, &size);

  // Cap at 1 MB to capture faulty OpenCL implementations (nVidia)
  vector<char> exts(std::min(size, max_size));

  clGetDeviceInfo(ctx.devices()[0].id(), CL_DEVICE_EXTENSIONS,
                  std::min(size, max_size), &(exts[0]), NULL);

  string extsstr(&(exts[0]));

  switch(cap) {
    case DEVICE_FP16_SUPPORT:
      return (extsstr.find("cl_khr_fp16") != string::npos);
    case DEVICE_FP32_SUPPORT:
      return true;
    case DEVICE_FP64_SUPPORT:
      return (extsstr.find("cl_khr_fp64") != string::npos)
          || (extsstr.find("cl_amd_fp64") != string::npos);
    case DEVICE_INT32_LOCAL_ATOMICS_SUPPORT:
      return (extsstr.find("cl_khr_local_int32_base_atomics") != string::npos);
    case DEVICE_INT64_LOCAL_ATOMICS_SUPPORT:
      return (extsstr.find("cl_khr_local_int32_base_atomics") != string::npos);
    case DEVICE_INT32_LOCAL_EXTENDED_ATOMICS_SUPPORT:
      return (extsstr.find("cl_khr_local_int32_extended_atomics")
                                                               != string::npos);
    case DEVICE_INT64_LOCAL_EXTENDED_ATOMICS_SUPPORT:
      return (extsstr.find("cl_khr_local_int64_extended_atomics")
                                                               != string::npos);
    case DEVICE_INT32_GLOBAL_ATOMICS_SUPPORT:
      return (extsstr.find("cl_khr_global_int32_base_atomics") != string::npos);
    case DEVICE_INT64_GLOBAL_ATOMICS_SUPPORT:
      return (extsstr.find("cl_khr_global_int64_base_atomics") != string::npos);
    case DEVICE_INT32_GLOBAL_EXTENDED_ATOMICS_SUPPORT:
      return (extsstr.find("cl_khr_global_int32_extended_atomics")
                                                               != string::npos);
    case DEVICE_INT64_GLOBAL_EXTENDED_ATOMICS_SUPPORT:
      return (extsstr.find("cl_khr_global_int64_extended_atomics")
                                                               != string::npos);
    case DEVICE_32_BIT_ADDRESS: {
      cl_uint address_bits;
      clGetDeviceInfo(ctx.devices()[0].id(),
                      CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, sizeof(cl_uint),
                      &address_bits, NULL);
      return std::min(static_cast<int32_t>(address_bits/8),
                      static_cast<int32_t>(sizeof(void*))) == 4;
    }
    case DEVICE_64_BIT_ADDRESS: {
      cl_uint address_bits;
      clGetDeviceInfo(ctx.devices()[0].id(),
                      CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, sizeof(cl_uint),
                      &address_bits, NULL);
      return std::min(static_cast<int32_t>(address_bits/8),
                      static_cast<int32_t>(sizeof(void*))) == 8;
    }
    default:
      return false;
  }
}


bool OclDevice::CheckType(string type) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(id_);
  const viennacl::ocl::device &device = ctx.current_device();

  if (type.compare("GPU") == 0 && device.type() == CL_DEVICE_TYPE_GPU)
    return true;
  if (type.compare("CPU") == 0 && device.type() == CL_DEVICE_TYPE_CPU)
    return true;
  return false;
}

void OclDevice::SwitchQueue(uint_tp id) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(id_);
  ctx.switch_queue(id % num_queues());
  current_queue_id_ = id % num_queues();
}

void OclDevice::FinishQueues() {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(id_);
  for (int i = 0; i < num_queues(); ++i) {
    ctx.switch_queue(i);
    OCL_CHECK(clFinish(ctx.get_queue().handle().get()));
  }
  ctx.switch_queue(0);
  current_queue_id_ = 0;
}

bool OclDevice::is_host_unified() {
  return host_unified_;
}

bool OclDevice::is_beignet() {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(id_);
  return ctx.devices()[0].opencl_c_version().find("beignet")
         != string::npos;
}

void OclDevice::ocl_null_kernel(float arg, cl_event* event) {
  clWaitForEvents(1, event);
  clReleaseEvent(*event);
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(
      Caffe::GetDefaultDevice()->id());
  shared_ptr<OclDeviceKernel> ocl_dev_kernel =
      static_pointer_cast<OclDeviceKernel>(
          this->math_programs_[AUX_DATA_INDEX]
                               ->GetKernel("caffe_gpu_null_kernel"));
  viennacl::ocl::kernel kernel = ocl_dev_kernel->get_ocl_kernel();
  clSetKernelArg(kernel.handle().get(), 0, sizeof(arg), &arg);
  clEnqueueTask(ctx.get_queue().handle().get(), kernel.handle().get(), 0,
                  NULL, event);
  ctx.get_queue().finish();
}

const char* OclDevice::clGetErrorString(cl_int error) {
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
    case -1022: return "clBLAS: Matrix a is not a valid memory object";
    case -1021: return "clBLAS: Matrix b is not a valid memory object";
    case -1020: return "clBLAS: Matrix c is not a valid memory object";
    case -1019: return "clBLAS: Vector x is not a valid memory object";
    case -1018: return "clBLAS: Vector y is not a valid memory object";
    case -1017: return "clBLAS: An input dimension (m:n:k) is invalid";
    case -1016: return "clBLAS: Leading dimension a must not be less than the "
        "size of the first dimension";
    case -1015: return "clBLAS: Leading dimension b must not be less than the "
        "size of the second dimension";
    case -1014: return "clBLAS: Leading dimension c must not be less than the "
        "size of the third dimension";
    case -1013: return "clBLAS: The increment for a vector x must not be 0";
    case -1012: return "clBLAS: The increment for a vector y must not be 0";
    case -1011: return "clBLAS: The memory object for Matrix a is too small";
    case -1010: return "clBLAS: The memory object for Matrix b is too small";
    case -1009: return "clBLAS: The memory object for Matrix c is too small";
    case -1008: return "clBLAS: The memory object for Vector x is too small";
    case -1007: return "clBLAS: The memory object for Vector y is too small";
    default: return "Unknown OpenCL error";
  }
}

#ifdef USE_CLFFT
const char* OclDevice::clfftGetErrorString(clfftStatus status) {
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
#endif  // USE_CLFFT

#endif  // USE_OPENCL

}  // namespace caffe
