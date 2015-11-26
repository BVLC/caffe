#ifndef CAFFE_UTIL_CL_HELPER_H_
#define CAFFE_UTIL_CL_HELPER_H_

#include <CL/cl.h>

#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "caffe/util/cl_kernel.hpp"

namespace caffe {

struct ClDeviceProperties {
  int dev_id;
  std::string name;
  unsigned int vendor_id;
  std::string vendor;
  std::string version;
  std::string profile;
  cl_device_type type;
  std::string driver_version;
  std::vector<std::string> extensions;
  bool available;
  bool compiler_available;
  bool little_endian;
  cl_device_fp_config single_fp_config;
  cl_device_fp_config double_fp_config;
  bool err_correction_support;
  unsigned int max_clock_freq;
  unsigned int profile_timer_res;
  unsigned int mem_base_addr_align;
  unsigned long local_mem_size;
  unsigned int device_max_work_group_size;
};

template<typename T>
struct ClMemOff {
  cl_mem memobj;
  size_t offset;  // offset in elements
};

class ClState {
 public:
  ClState();
  ~ClState();

  bool is_device_set();
  void set_device(int device_idx);
  int get_num_programs();

  cl_device_id get_device();
  cl_context get_context();
  cl_command_queue get_command_queue();

  const ClDeviceProperties& get_properties();
  void print_properties(std::ostream& out);

  bool fp64_supported();

  void* create_buffer(cl_mem_flags flags, size_t size, void* host_ptr);
  void destroy_buffer(void* buffer);
  size_t get_buffer_size(const void* buffer);
  ClMemOff<uint8_t> get_buffer_mem(const void* ptr);
  template<typename T>
  ClMemOff<T> get_buffer_mem(const T* ptr) {
    ClMemOff<uint8_t> m = get_buffer_mem(static_cast<const void*>(ptr));
    ClMemOff<T> mT = {m.memobj, m.offset / sizeof (T)};
    return mT;
  }

  cl_mem create_subbuffer(const void* ptr, size_t offset, cl_mem_flags flags);
  template <typename T> cl_mem create_subbuffer(T* ptr, int offset) {
    return create_subbuffer(ptr, offset * sizeof(T), CL_MEM_READ_WRITE);
  }
  template <typename T> cl_mem create_subbuffer(const T* ptr, int offset) {
    return create_subbuffer(ptr, offset * sizeof(T), CL_MEM_READ_ONLY);
  }

  bool submit_program(const char* name, const char* program_src_start,
      const char* program_src_end, const char* options = NULL,
      bool no_check = false);
  bool submit_program(const char* name,
      const std::vector< std::pair<const char*, const char*> >& program_srcs,
      const char* options = NULL, bool no_check = false);
  void release_program(const char* name);
  ClKernel& get_kernel(const char* name);

 private:
  ClState(const ClState&);

  struct Impl;
  Impl* impl_;
};

}  // namespace caffe

#endif
