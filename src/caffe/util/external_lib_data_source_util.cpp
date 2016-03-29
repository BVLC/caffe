#include <string>

#include "caffe/util/external_lib_data_source.hpp"
#include "caffe/util/external_lib_data_source_util.hpp"
#include "glog/logging.h"

using boost::shared_ptr;
using std::string;

#if defined(_MSC_VER)
#include <windows.h>  // NOLINT(build/include_order)

// Windows specific implementation of external library management (dll).
class WindowsDll : public IExternalLib {
 public:
  WindowsDll(const string& ext_lib_path,
    const string& factory_name, const string& ext_lib_param) {
    // Load dynamic link library.
    dll_module_ = LoadLibraryA(ext_lib_path.c_str());
    if (dll_module_ == nullptr) {
      LOG(FATAL) << "Error occurred while loading data source dll";
    }

    // Obtain pointer to factory method.
    ExternalLibDataSourceFactoryMethod factory_method =
      reinterpret_cast<ExternalLibDataSourceFactoryMethod>(
      GetProcAddress(dll_module_, factory_name.c_str()));
    if (factory_method == nullptr) {
      LOG(FATAL) << "Error occurred while accessing factory method in data "
        "source dll";
    }

    // Finally, use factory method to get data source object.
    data_source_ = factory_method(ext_lib_param.c_str());
    if (data_source_ == nullptr) {
      LOG(FATAL) << "Error occurred while creating data source object";
    }
  }

  virtual ~WindowsDll() override {
    // Release data source object.
    if (data_source_ != nullptr) {
      data_source_->Release();
    }

    // Free library handle.
    if (dll_module_ != nullptr) {
      FreeLibrary(dll_module_);
    }
  }

  virtual IExternalLibDataSource* GetDataSource() override {
    return data_source_;
  }

 private:
  HMODULE dll_module_;
  IExternalLibDataSource* data_source_;
};

#elif __linux__
#include <dlfcn.h>  // NOLINT(build/include_order)

// Linux specific implementation of external library management (so).
class LinuxSo : public IExternalLib {
 public:
  LinuxSo(const string& ext_lib_path,
    const string& factory_name, const string& ext_lib_param) {
    // Load dynamic load library.
    handle_ = dlopen(ext_lib_path.c_str(), RTLD_LAZY);
    if (handle_ == NULL) {
      LOG(FATAL) << "Error occurred while loading data source so";
    }

    // Obtain pointer to factory method.
    ExternalLibDataSourceFactoryMethod factory_method =
      reinterpret_cast<ExternalLibDataSourceFactoryMethod>(
      dlsym(handle_, factory_name.c_str()));
    if (factory_method == NULL) {
      LOG(FATAL) << "Error occurred while accessing factory method in data"
        "source so";
    }

    // Finally, use factory method to get data source object.
    data_source_ = factory_method(ext_lib_param.c_str());
    if (data_source_ == NULL) {
      LOG(FATAL) << "Error occurred while creating data source object";
    }
  }

  virtual ~LinuxSo() {
    // Release data source object.
    if (data_source_ != NULL) {
      data_source_->Release();
    }

    // Free library handle.
    if (handle_ != NULL) {
      dlclose(handle_);
    }
  }

  virtual IExternalLibDataSource* GetDataSource() {
    return data_source_;
  }

 private:
  void* handle_;
  IExternalLibDataSource* data_source_;
};

#endif

shared_ptr<IExternalLib> GetDataSourceLibraryWrapper(
  const string& ext_lib_path, const string& factory_name,
  const string& ext_lib_param) {
#if defined(_MSC_VER)
  return shared_ptr<IExternalLib>(new WindowsDll(ext_lib_path, factory_name,
    ext_lib_param));
#elif __linux__
  return shared_ptr<IExternalLib>(new LinuxSo(ext_lib_path, factory_name,
    ext_lib_param));
#else
  LOG(FATAL) << "External library management is not implemented for "
    "current platform!";
  return nullptr;
#endif
}
