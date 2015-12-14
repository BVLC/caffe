#include <vector>

#include "caffe/layers/tile_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif


namespace caffe {

#ifdef USE_CUDA
template<typename Dtype>
__global__ void Tile(const int_tp nthreads, const Dtype* bottom_data,
                     const int_tp tile_size, const int_tp num_tiles,
                     const int_tp bottom_tile_axis, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int_tp d = index % tile_size;
    const int_tp b = (index / tile_size / num_tiles) % bottom_tile_axis;
    const int_tp n = index / tile_size / num_tiles / bottom_tile_axis;
    const int_tp bottom_index = (n * bottom_tile_axis + b) * tile_size + d;
    top_data[index] = bottom_data[bottom_index];
  }
}
#endif  // USE_CUDA

template <typename Dtype>
void TileLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int_tp bottom_tile_axis = bottom[0]->shape(axis_);
  const int_tp nthreads = top[0]->count();
  if (this->get_device()->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    Tile<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
    CUDA_KERNEL(CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS)(
        nthreads, bottom_data, inner_dim_, tiles_, bottom_tile_axis, top_data);
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();

    viennacl::ocl::kernel &oclk_tile = program.get_kernel(
        CL_KERNEL_SELECT("tile"));
    viennacl::ocl::enqueue(
        oclk_tile(nthreads, WrapHandle((cl_mem) bottom_data, &ctx), inner_dim_,
                  tiles_, bottom_tile_axis,
                  WrapHandle((cl_mem) top_data, &ctx)),
        ctx.get_queue());
#endif  // USE_GREENTEA
  }
}

#ifdef USE_CUDA
template <typename Dtype>
__global__ void TileBackward(const int_tp nthreads, const Dtype* top_diff,
                             const int_tp tile_size, const int_tp num_tiles,
                             const int_tp bottom_tile_axis,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int_tp d = index % tile_size;
    const int_tp b = (index / tile_size) % bottom_tile_axis;
    const int_tp n = index / tile_size / bottom_tile_axis;
    bottom_diff[index] = 0;
    int_tp top_index = (n * num_tiles * bottom_tile_axis + b) * tile_size + d;
    for (int_tp t = 0; t < num_tiles; ++t) {
      bottom_diff[index] += top_diff[top_index];
      top_index += bottom_tile_axis * tile_size;
    }
  }
}
#endif  // USE_CUDA

template <typename Dtype>
void TileLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int_tp bottom_tile_axis = bottom[0]->shape(axis_);
  const int_tp tile_size = inner_dim_ / bottom_tile_axis;
  const int_tp nthreads = bottom[0]->count();

  if (this->get_device()->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    TileBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
    CUDA_KERNEL(CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS)(
        nthreads, top_diff, tile_size, tiles_, bottom_tile_axis, bottom_diff);
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();

    viennacl::ocl::kernel &oclk_tile = program.get_kernel(
        CL_KERNEL_SELECT("tile_backward"));
    viennacl::ocl::enqueue(
        oclk_tile(nthreads, WrapHandle((cl_mem) top_diff, &ctx), tile_size,
                  tiles_, bottom_tile_axis,
                  WrapHandle((cl_mem) bottom_diff, &ctx)),
        ctx.get_queue());
#endif  // USE_GREENTEA
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TileLayer);

}  // namespace caffe
