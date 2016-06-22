#include <glog/logging.h>

#include <algorithm>
#include <cstdio>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "caffe/greentea/cl_kernels.hpp"
#include "caffe/util/cl_state.hpp"

using std::find;
using std::endl;
using std::map;
using std::make_pair;
using std::pair;
using std::ostream;
using std::string;
using std::vector;

namespace caffe {

struct ClState::Impl {
  explicit Impl() {
  }

  void initialize() {
    free_mem_[NULL] = static_cast<size_t>(1) << (sizeof (size_t) * 8 - 1);
  }

  void* create_buffer(int dev_id, cl_mem_flags flags, size_t size,
                      void* host_ptr, cl_int *errcode) {
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
    cl_mem memobj = clCreateBuffer(ctx.handle().get(), flags,
                                   size, host_ptr, errcode);
    void* buffer = get_memptr_from_freemem(size);
    memobjs_[buffer] = make_pair(memobj, size);
    memdev_[memobj] = dev_id;

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

  size_t get_buffer_size(const void* buffer) {
    map<void*, pair<cl_mem, size_t> >::iterator it =
    memobjs_.find(const_cast<void*>(buffer));

    if (it == memobjs_.end()) {
      LOG(FATAL) << "Invalid buffer object";
    }
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

  int get_mem_dev(cl_mem memobj) {
    return memdev_[memobj];
  }

  void destroy_buffer(void* buffer) {
    map<void*, pair<cl_mem, size_t> >::iterator it1 = memobjs_.find(buffer);
    if (it1 == memobjs_.end())
      LOG(FATAL) << "Invalid buffer";

    cl_mem mem = it1->second.first;
    int size = it1->second.second;
    memobjs_.erase(it1);

    map<cl_mem, int>::iterator it2 = memdev_.find(mem);
    memdev_.erase(it2);

    free_mem_[static_cast<char*>(buffer) + size] = size;

    combine_free_mem();

    clReleaseMemObject(mem);
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


  map<void*, pair<cl_mem, size_t> > memobjs_;
  map<void*, size_t> free_mem_;
  map<cl_mem, int> memdev_;
};

ClState::ClState() {
  impl_ = new Impl();
  impl_->initialize();
}

ClState::~ClState() {
  if (impl_ != NULL)
    delete impl_;
}

void* ClState::create_buffer(int dev_id, cl_mem_flags flags, size_t size,
                             void* host_ptr, cl_int *errcode) {
  return impl_->create_buffer(dev_id, flags, size, host_ptr, errcode);
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

int ClState::get_mem_dev(cl_mem memobj) {
  return impl_->get_mem_dev(memobj);
}

cl_mem ClState::create_subbuffer(const void* ptr, size_t offset,
        cl_mem_flags flags) {
  ClMemOff<uint8_t> buf = get_buffer_mem(ptr);
  size_t size = get_buffer_size(static_cast<const char*>(ptr) - buf.offset);
  cl_buffer_region bufReg = { offset, size - offset };
  cl_int err;
  cl_mem sub_buf = clCreateSubBuffer(buf.memobj, flags,
            CL_BUFFER_CREATE_TYPE_REGION, &bufReg, &err);
  return sub_buf;
}
}  // namespace caffe
