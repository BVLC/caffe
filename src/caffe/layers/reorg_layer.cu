#include "caffe/layers/reorg_layer.hpp"
#include "caffe/util/math_functions.hpp"
#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {
#ifdef USE_CUDA
template <typename Dtype>
__global__ void reorg_kernel(const Dtype *x, int_tp w, int_tp h, int_tp c, int_tp batch, int_tp stride, int_tp forward, Dtype *out)
{
	int_tp size = batch*c*h*w;
	int_tp i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if(i >= size) return;
	int_tp in_index = i;
	int_tp in_w = i%w;
	i = i/w;
	int_tp in_h = i%h;
	i = i/h;
	int_tp in_c = i%c;
	i = i/c;
	int_tp b = i%batch;

	int_tp out_c = c/(stride*stride);

	int_tp c2 = in_c % out_c;
	int_tp offset = in_c / out_c;
	int_tp w2 = in_w*stride + offset % stride;
	int_tp h2 = in_h*stride + offset / stride;
	int_tp out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));

	if(forward)
	{
		out[out_index] = x[in_index];
	}         
	else
	{
		out[in_index] = x[out_index];
	}
}
#endif  // USE_CUDA

template<typename Dtype>
void ReorgLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
									const vector<Blob<Dtype> *> &top) {
	const Dtype *bottom_data = bottom[0]->gpu_data();
	int_tp count = bottom[0]->count();
	Dtype *top_data = top[0]->mutable_gpu_data();

  if (this->device_->backend() == BACKEND_CUDA) {	
#ifdef USE_CUDA
	reorg_kernel<Dtype>
	 CUDA_KERNEL(CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS)(bottom_data, width_, height_,
			  channels_, batch_num_, stride_, reverse_, top_data);
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->template program<Dtype>();

    // Execute kernel
    viennacl::ocl::kernel &oclk_reorg_forward = program.get_kernel(
      CL_KERNEL_SELECT("reorg"));
    viennacl::ocl::enqueue(
      oclk_reorg_forward(count, WrapHandle((cl_mem)bottom_data, &ctx),
	  width_, height_, channels_, batch_num_, stride_, reverse_,
      WrapHandle((cl_mem)top_data, &ctx)),
      ctx.get_queue());
#endif  // USE_GREENTEA
  }
}

template<typename Dtype>
void ReorgLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
									 const vector<Blob<Dtype> *> &bottom) {
  if (!propagate_down[0]) {
    return;
  }
  int_tp count = diff_.count();
  const Dtype *top_diff = diff_.mutable_gpu_diff();
  Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
  if (this->device_->backend() == BACKEND_CUDA) {	
#ifdef USE_CUDA	
	reorg_kernel<Dtype>
	 CUDA_KERNEL(CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS)(top_diff, width_, height_,
			  channels_, batch_num_, stride_, !reverse_, bottom_diff);
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
      viennacl::ocl::context &ctx = viennacl::ocl::get_context(
          this->device_->id());
      viennacl::ocl::program &program = this->device_->template program<Dtype>();
      viennacl::ocl::kernel &oclk_reorg_backward = program.get_kernel(
          CL_KERNEL_SELECT("reorg"));
      viennacl::ocl::enqueue(
          oclk_reorg_backward(count, WrapHandle((cl_mem)top_diff, &ctx),
			      width_, height_, channels_, batch_num_, stride_, int(!reverse_),
			      WrapHandle((cl_mem)bottom_diff, &ctx)),
		  ctx.get_queue());
#endif  // USE_GREENTEA
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ReorgLayer);

}  // namespace caffe
