#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#include "caffe/greentea/greentea_im2col.hpp"
#endif

namespace caffe {

template<typename Dtype>
void Im2colLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  if (this->device_context_.backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    for (int n = 0; n < bottom[0]->num(); ++n) {
      im2col_gpu(bottom_data + bottom[0]->offset(n), channels_, height_, width_,
                 kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
                 top_data + top[0]->offset(n));
    }
#endif // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_context_.id());
    viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(
        this->device_context_.id());

    for (int n = 0; n < bottom[0]->num(); ++n) {
      greentea_im2col_gpu<Dtype>(program, ctx, (cl_mem)bottom_data, bottom[0]->offset(n), channels_, height_, width_,
                 kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
                 (cl_mem)top_data, top[0]->offset(n));
    }
#endif // USE_GREENTEA
  }

}

template<typename Dtype>
void Im2colLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  if (this->device_context_.backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    for (int n = 0; n < top[0]->num(); ++n) {
      col2im_gpu(top_diff + top[0]->offset(n), channels_, height_, width_,
                 kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
                 bottom_diff + bottom[0]->offset(n));
    }
#endif // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_context_.id());
    viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(
        this->device_context_.id());

    for (int n = 0; n < top[0]->num(); ++n) {
      greentea_col2im_gpu<Dtype>(program, ctx, (cl_mem) top_diff, top[0]->offset(n),
                          channels_, height_, width_, kernel_h_, kernel_w_,
                          pad_h_, pad_w_, stride_h_, stride_w_,
                          (cl_mem) bottom_diff, bottom[0]->offset(n));
    }
#endif // USE_GREENTEA
  }

}

INSTANTIATE_LAYER_GPU_FUNCS(Im2colLayer);

}  // namespace caffe
