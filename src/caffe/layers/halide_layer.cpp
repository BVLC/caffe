#include "caffe/layers/halide_layer.hpp"

#include <dlfcn.h>
#include <HalideRuntime.h>

#include <string>
#include <utility>
#include <vector>



#define CHECK_DL { dlsym_error = dlerror(); \
  if (dlsym_error) { LOG(FATAL) << "Dynamic linking error: " << dlsym_error;} }

namespace caffe {

template <typename Dtype>
void HalideLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  string library(param_.library());

  LOG(INFO) << library;

  halide_lib_handle = dlopen(library.c_str(), RTLD_LAZY);
  if (halide_lib_handle == NULL) {
    LOG(INFO) << "Failed to open dynamic library: " << library;
    LOG(INFO) << "with error: "<< dlerror();
    LOG(INFO) << "ProTip: Check that libHalide.so file is in LD_LIBRARY_PATH";
    LOG(FATAL) << "Halide layer failed to open dynamic library";
  }
  // reset errors
  dlerror();
  const char* dlsym_error;

  // load the symbols
  create_ext = reinterpret_cast<create_t*>( dlsym(halide_lib_handle, "create"));
  CHECK_DL

  destroy_ext = reinterpret_cast<destroy_t*>( dlsym(halide_lib_handle, "destroy") );
  CHECK_DL

  // create an instance of the class
  inst_ext = create_ext();
}

template <typename Dtype>
HalideLayer<Dtype>::~HalideLayer() {
  // destroy the class
  destroy_ext(inst_ext);

  // unload the halide library
  dlclose(halide_lib_handle);
}

template <typename Dtype>
void HalideLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  inst_ext->Reshape(bottom, top);
}

template <typename Dtype>
void HalideLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Now for what we have all been waiting for.
  inst_ext->Forward_gpu(bottom, top);
}

template <typename Dtype>
void HalideLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  inst_ext->Backward_gpu(bottom, propagate_down, top);
}


template <typename Dtype>
void HalideLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void HalideLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
// STUB_GPU(HalideLayer);
#endif

INSTANTIATE_CLASS(HalideLayer);
REGISTER_LAYER_CLASS(Halide);

}  // namespace caffe
