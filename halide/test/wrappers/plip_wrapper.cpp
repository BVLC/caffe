#include "plip_wrapper.h"


#include "Halide.h"

#include "caffe/util/halide.hpp"
// This is a generated file
#include "plip.h"


using namespace Halide;

/* This Halide Wrapper Layer handles
 *
 */
template <typename Dtype>
int inline PlipWrapper<Dtype>::ExactNumBottomBlobs() { return 1; }

template <typename Dtype>
int inline PlipWrapper<Dtype>::ExactNumTopBlobs() { return 1; }


template <typename Dtype>
void PlipWrapper<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {}

template <typename Dtype>
void PlipWrapper<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  /*
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
  plip_argv(args);

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
  */
  bottom_buf_ = HalideWrapBlob(*bottom[0]);
  top_buf_ = HalideWrapBlob(*top[0]);
  plip(&bottom_buf_, &top_buf_);
  HalideSyncBlob(bottom_buf_, bottom[0]);
  HalideSyncBlob(top_buf_, top[0]);
}


template <typename Dtype>
void PlipWrapper<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  bottom_diff_buf_ = HalideWrapBlob(*bottom[0], false);
  top_diff_buf_ = HalideWrapBlob(*top[0], false);
  plip(&top_diff_buf_, &bottom_diff_buf_);
  HalideSyncBlob(top_diff_buf_, top[0], false);
  HalideSyncBlob(bottom_diff_buf_, bottom[0], false);
}


template <typename Dtype>
void PlipWrapper<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // CHECK_EQ(bottom[0]->shape(0), bottom[0]->shape(1));
  top[0]->ReshapeLike(*bottom[0]);
}


// the class factories
extern "C" DLLInterface<float>* create() {
    return new PlipWrapper<float>;
}

extern "C" void destroy(DLLInterface<float>* p) {
    delete p;
}

