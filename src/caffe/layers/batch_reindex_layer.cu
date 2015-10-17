#include <algorithm>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

#ifdef USE_CUDA
template<typename Dtype>
__global__ void BRForward(const int count, const int inner_dim, const Dtype* in,
                          const Dtype* permut, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / (inner_dim);
    int in_n = static_cast<int>(permut[n]);
    out[index] = in[in_n * (inner_dim) + index % (inner_dim)];
  }
}
#endif  // USE_CUDA

template<typename Dtype>
void BatchReindexLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  check_batch_reindex(bottom[0]->shape(0), bottom[1]->count(),
                      bottom[1]->cpu_data());
  if (top[0]->count() == 0) {
    return;
  }
  int threads = top[0]->count();

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    // NOLINT_NEXT_LINE(whitespace/operators)
    BRForward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(threads),
                                 CAFFE_CUDA_NUM_THREADS) (
        top[0]->count(), bottom[0]->count() / bottom[0]->shape(0),
        bottom[0]->gpu_data(), bottom[1]->gpu_data(),
        top[0]->mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(
        this->device_->id());

    viennacl::ocl::kernel &oclk_br = program.get_kernel(
        CL_KERNEL_SELECT("br_forward"));
    viennacl::ocl::enqueue(
        oclk_br(top[0]->count(), bottom[0]->count() / bottom[0]->shape(0),
                WrapHandle((cl_mem) (bottom[0]->gpu_data()), &ctx),
                WrapHandle((cl_mem) (bottom[1]->gpu_data()), &ctx),
                WrapHandle((cl_mem) (top[0]->mutable_gpu_data()), &ctx)),
        ctx.get_queue());
#endif  // USE_GREENTEA
  }
}

#ifdef USE_CUDA
template<typename Dtype>
__global__ void BRBackward(const int count, const int inner_dim,
                           const Dtype* in, const Dtype* top_indexes,
                           const Dtype* begins, const Dtype* counts,
                           Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / (inner_dim);
    out[index] = 0;
    int lower = static_cast<int>(begins[n]);
    int upper = lower + static_cast<int>(counts[n]);
    for (int i = lower; i < upper; ++i) {
      int in_n = static_cast<int>(top_indexes[i]);
      out[index] += in[in_n * (inner_dim) + index % (inner_dim)];
    }
  }
}
#endif  // USE_CUDA

template<typename Dtype>
void BatchReindexLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  CHECK(!propagate_down[1]) << "Cannot backprop to index.";
  if (!propagate_down[0]) {
    return;
  }

  vector<std::pair<int, int> > mapping;
  const Dtype* perm = bottom[1]->cpu_data();
  for (int i = 0; i < bottom[1]->count(); ++i) {
    mapping.push_back(pair<int, int>(static_cast<int>(perm[i]), i));
  }
  std::sort(mapping.begin(), mapping.end(), pair_sort_first());

  // Each element of the bottom diff is potentially the sum of many top diffs.
  // However, we'd like each CUDA thread to handle exactly one output.  Hence,
  // we first pre-compute a list of lists of indices that need to be summed for
  // each output. `top_indexes` holds the data of this list of lists.  The
  // k'th element of `begins` points to the location in `top_indexes` where the
  // list for the k'th example begin, and the k'th element of `counts` is the
  // length of that list.
  vector<int> shape;
  shape.push_back(bottom[1]->count());
  Blob<Dtype> top_indexes(shape);
  shape[0] = bottom[0]->shape(0);
  Blob<Dtype> counts(shape);
  Blob<Dtype> begins(shape);
  Dtype* t_i_data = top_indexes.mutable_cpu_data();
  Dtype* c_data = counts.mutable_cpu_data();
  Dtype* b_data = begins.mutable_cpu_data();
  caffe_set(begins.count(), Dtype(-1), b_data);
  caffe_set(counts.count(), Dtype(0), c_data);
  for (int i = 0; i < mapping.size(); ++i) {
    t_i_data[i] = mapping[i].second;
    if (b_data[mapping[i].first] == -1) {
      b_data[mapping[i].first] = i;
    }
    c_data[mapping[i].first] += 1;
  }

  int threads = bottom[0]->count();

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    // NOLINT_NEXT_LINE(whitespace/operators)
    BRBackward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(threads),
                                  CAFFE_CUDA_NUM_THREADS) (
        bottom[0]->count(), bottom[0]->count() / bottom[0]->shape(0),
        top[0]->gpu_diff(), top_indexes.gpu_data(), begins.gpu_data(),
        counts.gpu_data(), bottom[0]->mutable_gpu_diff());
    CUDA_POST_KERNEL_CHECK;
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(
        this->device_->id());

    viennacl::ocl::kernel &oclk_br = program.get_kernel(
        CL_KERNEL_SELECT("br_backward"));
    viennacl::ocl::enqueue(
        oclk_br(bottom[0]->count(), bottom[0]->count() / bottom[0]->shape(0),
                  WrapHandle((cl_mem)(top[0]->gpu_diff()), &ctx),
                  WrapHandle((cl_mem)(top_indexes.gpu_data()), &ctx),
                  WrapHandle((cl_mem)(begins.gpu_data()), &ctx),
                  WrapHandle((cl_mem)(counts.gpu_data()), &ctx),
                  WrapHandle((cl_mem)(bottom[0]->mutable_gpu_diff()), &ctx)),
        ctx.get_queue());
#endif  // USE_GREENTEA
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BatchReindexLayer);

}  // namespace caffe
