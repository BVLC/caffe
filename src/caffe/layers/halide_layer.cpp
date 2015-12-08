#include "caffe/layers/halide_layer.hpp"
#include <dlfcn.h>





namespace caffe {


template <typename Dtype>
void HalideLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  string library(  this->layer_param_.halide_param().library() );
  LOG(INFO) << library;
  stride_ = 1;

  halide_lib_handle = dlopen(library.c_str(), RTLD_LAZY);
  if ( halide_lib_handle == NULL ) {
    LOG(INFO) << "Failed to open dynamic library: " << library;
    LOG(INFO) << "with error:"<< dlerror();
  }

  // reset errors
  dlerror();

  h_Forward_gpu = (hfunc_t) dlsym(halide_lib_handle, "plip_Forward_gpu");
  const char* dlsym_error = dlerror();
  if (dlsym_error) {
      LOG(FATAL) << "Cannot load symbol create: " << dlsym_error;
  }
  /*
  h_Backward_gpu = (hfunc_t) dlsym(halide_lib_handle, "plip_Backward_gpu");
  dlsym_error = dlerror();
  if (dlsym_error) {
      LOG(FATAL) << "Cannot load symbol destroy: " << dlsym_error;
  }
  */

  /*
  // load the symbols
  create_ext = (create_t*) dlsym(halide_lib_handle, "create");
  const char* dlsym_error = dlerror();
  if (dlsym_error) {
      LOG(FATAL) << "Cannot load symbol create: " << dlsym_error;
  }

  destroy_ext = (destroy_t*) dlsym(halide_lib_handle, "destroy");
  dlsym_error = dlerror();
  if (dlsym_error) {
      LOG(FATAL) << "Cannot load symbol destroy: " << dlsym_error;
  }

  // create an instance of the class
  inst_ext = create_ext();
  */
}

template <typename Dtype>
HalideLayer<Dtype>::~HalideLayer() {
  // destroy the class
  //destroy_ext(inst_ext);

  // unload the halide library
  dlclose(halide_lib_handle);
}

template <typename Dtype>
void HalideLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // XXX update this for stride > 1
  // XXX shape checks
  CHECK_EQ(bottom[0]->shape(0), bottom[0]->shape(1));
  top[0]->ReshapeLike(*bottom[0]);
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


template <typename Dtype>
void HalideLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  bottom_buf_ = HalideWrapBlob(*bottom[0]);
  top_buf_ = HalideWrapBlob(*top[0]);

  // Halide(&bottom_buf_, &top_buf_, stride_);
  h_Forward_gpu(&bottom_buf_, &top_buf_);


  HalideSyncBlob(bottom_buf_, bottom[0]);
  HalideSyncBlob(top_buf_, top[0]);
}

template <typename Dtype>
void HalideLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  bottom_diff_buf_ = HalideWrapBlob(*bottom[0], false);
  top_diff_buf_ = HalideWrapBlob(*top[0], false);
  //plip(&top_diff_buf_, &bottom_diff_buf_);
  HalideSyncBlob(top_diff_buf_, top[0], false);
  HalideSyncBlob(bottom_diff_buf_, bottom[0], false);
}

#ifdef CPU_ONLY
// STUB_GPU(HalideLayer);
#endif

INSTANTIATE_CLASS(HalideLayer);
REGISTER_LAYER_CLASS(Halide);

}  // namespace caffe
