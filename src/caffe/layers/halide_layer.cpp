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
  CHECK_GT(param_.library_size(), 0) << "Must have at least one library";
  string library(param_.library(0));

  LOG(INFO) << library;

  halide_lib_handle = dlopen(library.c_str(), RTLD_LAZY);
  if (halide_lib_handle == NULL) {
    LOG(INFO) << "Failed to open dynamic library: " << library;
    LOG(INFO) << "with error:"<< dlerror();
  }
  // reset errors
  dlerror();
  const char* dlsym_error;

  // Get the metadata

  // Get the halide metadata names
  vector<string> halide_names;
  vector<int> halide_kinds;
  halide_filter_metadata_t* md = reinterpret_cast<halide_filter_metadata_t*>(
        dlsym(halide_lib_handle, "plip_metadata"));
  CHECK_DL;

  int num_arguments = md->num_arguments;
  for (int i = 0; i < num_arguments; i++) {
    halide_names.push_back(md->arguments[i].name);
    halide_kinds.push_back(md->arguments[i].kind);
  }

  // Go over halide names
  caffe_from_halide_.clear();
  for (int i = 0; i < halide_names.size(); i++) {
    int kind = halide_kinds[i];

    if (kind == halide_argument_kind_input_buffer) {
      int bottom_size = bottom.size();
      if (bottom_size == 1) {
        // Ignore bottom names
        caffe_from_halide_.push_back(std::pair<int, int>(0, 0));
      } else {
        // Look through bottom names
        CHECK_EQ(param_.bottom_blob_names_size(), bottom_size);

        string halide_name = halide_names[i];
        bool found = false;
        for (int j = 0; j < bottom_size; j++) {
          if (halide_name == param_.bottom_blob_names(j)) {
            caffe_from_halide_.push_back(std::pair<int, int>(0, j));
            found = true;
            break;
          }
        }
        CHECK(found) << "Halide input param " << halide_name
                     << " not found in proto bottom_blob_names";
      }
    } else if (kind == halide_argument_kind_output_buffer) {
      int top_shape = top.size();
      if (top_shape == 1) {
        // Ignore bottom names
        caffe_from_halide_.push_back(std::pair<int, int>(1, 0));
      } else {
        // Look through bottom names
        CHECK_EQ(param_.top_blob_names_size(), top_shape);

        string halide_name = halide_names[i];
        bool found = false;
        for (int j = 0; j < top_shape; j++) {
          if (halide_name == param_.top_blob_names(j)) {
            caffe_from_halide_.push_back(std::pair<int, int>(1, j));
            found = true;
            break;
          }
        }
        CHECK(found) << "Halide output param " << halide_name
                     << " not found in proto top_blob_names";
      }
    } else {
        LOG(FATAL) << "Invalid halid argument kind only accepting"
                   << "halide_argument_kind_input_buffer and"
                   << "halide_argument_kind_output_buffer";
    }
  }

  // Go over functions
  // int num_functions = param_.function_name_size();
  // CHECK_EQ(param_.library_size(), num_functions )
  //    << "function_name, library size missmatch";

  // CHECK_EQ(param_.function_type_size(), num_functions )
  //    << "function_name, function_type missmatch";

  hv_Forward_gpu = (hvfunc_t) dlsym(halide_lib_handle, "plip_argv");
  CHECK_DL

  /*
  // load the symbols
  create_ext = (create_t*) dlsym(halide_lib_handle, "create");
  CHECK_DL
  destroy_ext = (destroy_t*) dlsym(halide_lib_handle, "destroy");
  CHECK_DL

  // create an instance of the class
  inst_ext = create_ext();
  */
}

template <typename Dtype>
HalideLayer<Dtype>::~HalideLayer() {
  // destroy the class
  // destroy_ext(inst_ext);

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
  std::vector< buffer_t > halide_buffers;

  int n = caffe_from_halide_.size();
  void* args[n];

  for (int i = 0; i < n; i++) {
    int blob_source = caffe_from_halide_[i].first;
    int blob_index = caffe_from_halide_[i].second;
    if (blob_source == 0) {
      halide_buffers.push_back(HalideWrapBlob(*bottom[blob_index]));
      args[i] = &(halide_buffers.back());
    } else if (blob_source == 1) {
      halide_buffers.push_back(HalideWrapBlob(*top[blob_index]));
      args[i] = &(halide_buffers.back());
    } else {
      LOG(FATAL) << "Wrong blob_source value";
    }
  }

  // Now for what we have all been waiting for.
  hv_Forward_gpu(args);

  for (int i = 0; i < n; i++) {
    int blob_source = caffe_from_halide_[i].first;
    int blob_index = caffe_from_halide_[i].second;
    if (blob_source == 0) {
      HalideSyncBlob(
            *(reinterpret_cast<buffer_t*>(args[i])), bottom[blob_index]);
    } else if (blob_source == 1) {
      HalideSyncBlob(
            *(reinterpret_cast<buffer_t*>(args[i])), top[blob_index]);
    } else {
      LOG(FATAL) << "Wrong blob_source value";
    }
  }
}

template <typename Dtype>
void HalideLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  bottom_diff_buf_ = HalideWrapBlob(*bottom[0], false);
  top_diff_buf_ = HalideWrapBlob(*top[0], false);
  // plip(&top_diff_buf_, &bottom_diff_buf_);
  HalideSyncBlob(top_diff_buf_, top[0], false);
  HalideSyncBlob(bottom_diff_buf_, bottom[0], false);
}

#ifdef CPU_ONLY
// STUB_GPU(HalideLayer);
#endif

INSTANTIATE_CLASS(HalideLayer);
REGISTER_LAYER_CLASS(Halide);

}  // namespace caffe
