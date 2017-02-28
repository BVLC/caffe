#include <algorithm>
#include <utility>
#include <vector>

#include "caffe/layers/batch_reindex_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

#ifdef USE_CUDA
template<typename Dtype>
__global__ void BRForward(const int_tp count, const int_tp inner_dim,
                          const Dtype* in, const Dtype* permut, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    int_tp n = index / (inner_dim);
    int_tp in_n = static_cast<int_tp>(permut[n]);
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
  if (this->device_->backend() == BACKEND_CUDA) {
    int_tp threads = top[0]->count();
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
    viennacl::ocl::program &program = this->device_->program();

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
__global__ void BRBackward(const int_tp count, const int_tp inner_dim,
                           const Dtype* in, const Dtype* top_indexes,
                           const Dtype* begins, const Dtype* counts,
                           Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    int_tp n = index / (inner_dim);
    out[index] = 0;
    int_tp lower = static_cast<int_tp>(begins[n]);
    int_tp upper = lower + static_cast<int_tp>(counts[n]);
    for (int_tp i = lower; i < upper; ++i) {
      int_tp in_n = static_cast<int_tp>(top_indexes[i]);
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

  vector<std::pair<int_tp, int_tp> > mapping;
  const Dtype* perm = bottom[1]->cpu_data();
  for (int_tp i = 0; i < bottom[1]->count(); ++i) {
    mapping.push_back(pair<int_tp, int_tp>(static_cast<int_tp>(perm[i]), i));
  }
  std::sort(mapping.begin(), mapping.end(), pair_sort_first());

  // Each element of the bottom diff is potentially the sum of many top diffs.
  // However, we'd like each CUDA thread to handle exactly one output.  Hence,
  // we first pre-compute a list of lists of indices that need to be summed for
  // each output. `top_indexes` holds the data of this list of lists.  The
  // k'th element of `begins` points to the location in `top_indexes` where the
  // list for the k'th example begin, and the k'th element of `counts` is the
  // length of that list.
  vector<int_tp> shape;
  shape.push_back(bottom[1]->count());
  Blob<Dtype> top_indexes(shape, this->device_);
  shape[0] = bottom[0]->shape(0);
  Blob<Dtype> counts(shape, this->device_);
  Blob<Dtype> begins(shape, this->device_);
  Dtype* t_i_data = top_indexes.mutable_cpu_data();
  Dtype* c_data = counts.mutable_cpu_data();
  Dtype* b_data = begins.mutable_cpu_data();
  caffe_set(begins.count(), Dtype(-1), b_data);
  caffe_set(counts.count(), Dtype(0), c_data);
  for (int_tp i = 0; i < mapping.size(); ++i) {
    t_i_data[i] = mapping[i].second;
    if (b_data[mapping[i].first] == -1) {
      b_data[mapping[i].first] = i;
    }
    c_data[mapping[i].first] += 1;
  }

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    int_tp threads = bottom[0]->count();
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
    viennacl::ocl::program &program = this->device_->program();

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
