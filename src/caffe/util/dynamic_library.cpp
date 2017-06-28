#include "caffe/util/dynamic_library.hpp"

#include <boost/filesystem.hpp>

#include <stdexcept>
#include <string>
#include <vector>

#ifndef _WIN32
#  include "dlfcn.h"
#  ifdef RTLD_DEEPBIND
#    define CAFFE_DLOPEN_FLAGS (RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND)
#  else
#    define CAFFE_DLOPEN_FLAGS (RTLD_LAZY | RTLD_LOCAL)
#  endif
#else
#  include "Windows.h"
#endif

namespace caffe {

namespace {
#ifndef _WIN32
  void * OpenLibrary(std::string const & name) {
    return ::dlopen(name.c_str(), CAFFE_DLOPEN_FLAGS);
  }

  bool CloseLibrary(void * handle) {
    return ::dlclose(handle) == 0;
  }

  void * FindSymbol(void * handle, std::string const & name) {
    return ::dlsym(handle, name.c_str());
  }
#else
  void * OpenLibrary(std::string const & name) {
    return reinterpret_cast<void *>(::LoadLibrary(name.c_str()));
  }

  bool CloseLibrary(void * handle) {
    return ::FreeLibrary(reinterpret_cast<HMODULE>(handle));
  }

  void * FindSymbol(void * handle, std::string const & name) {
    return ::GetProcAddress(reinterpret_cast<HMODULE>(handle), name.c_str());
  }
#endif
}  // namespace

DynamicLibrary FindLibrary(
  std::string const & name,
  std::vector<std::string> search_path
) {
  for (size_t i = 0; i < search_path.size(); ++i) {
    std::string path =
      (boost::filesystem::path(search_path[i]) / name).string();
    DynamicLibrary library{path};
    if (library.IsValid()) return library;
  }

  return DynamicLibrary{};
}

DynamicLibrary::DynamicLibrary(std::string const & name) : path_(name) {
  handle_ = ::caffe::OpenLibrary(name.c_str());
}

DynamicLibrary::DynamicLibrary(DynamicLibrary && other) {
  handle_ = other.handle_;
  other.handle_ = nullptr;
}

DynamicLibrary::~DynamicLibrary() {
  if (handle_) ::caffe::CloseLibrary(handle_);
}

void * DynamicLibrary::FindSymbol(std::string const & name) const {
  return ::caffe::FindSymbol(handle_, name);
}

}  // namespace caffe
