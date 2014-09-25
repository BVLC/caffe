#include <algorithm>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Container>
struct compare_indirect_index_ascend  {
  const Container& container;
  explicit compare_indirect_index_ascend(const Container& container):
  container(container) {
  }
  bool operator()(size_t lindex, size_t rindex) const {
    return container[lindex] < container[rindex];
    }
};

template <typename Container>
struct compare_indirect_index_descend {
  const Container& container;
  explicit compare_indirect_index_descend(const Container& container):
  container(container)  {
  }
  bool operator()(size_t lindex, size_t rindex) const {
    return container[lindex] > container[rindex];
    }
};

#if __cplusplus > 199711L
template <typename Dtype>
std::vector<size_t> kth_element_idxs(const std::vector<Dtype> &v,
  const size_t k, const int sortAscend = 1) {
  std::vector<size_t> idx(v.size());
  for (size_t i = 0; i != idx.size(); ++i) {
      idx[i] = i;
    }
  if (sortAscend) {
      std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                compare_indirect_index_ascend <std::vector<Dtype> > (v));
    } else {
      std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                compare_indirect_index_descend <std::vector<Dtype> > (v));
    }
  return idx;
}
#else
template <typename Dtype>
std::vector<size_t> sort_idxs(const std::vector<Dtype> &v,
                                 const int sortAscend = 1) {
  std::vector<size_t> idx(v.size());
  for (size_t i = 0; i != idx.size(); ++i) {
      idx[i] = i;
    }
  if (sortAscend) {
      std::sort(idx.begin(), idx.end(),
                compare_indirect_index_ascend <std::vector<Dtype> > (v));
  } else {
      std::sort(idx.begin(), idx.end(),
                compare_indirect_index_descend <std::vector<Dtype> > (v));
  }
  return idx;
}
#endif

template <typename Dtype>
void TopKLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  mask_.Reshape(bottom[0]->num(), bottom[0]->channels(),
        bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void TopKLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  if (Caffe::phase() == Caffe::TRAIN) {
  uint_k_ = this->layer_param_.topk_param().k();
  } else {
  uint_k_ = this->layer_param_.topk_param().k() *
            this->layer_param_.topk_param().a();
  }

  DCHECK_GT(uint_k_, 0);
  DCHECK(uint_k_ <= bottom[0]->count() / bottom[0]->num());
  }

  template <typename Dtype>
  void TopKLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    uint* mask = mask_.mutable_cpu_data();

    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const int single_count = bottom[0]->count() / bottom[0]->num();

    caffe_set(count, Dtype(0), top_data);
    caffe_memset(sizeof(uint) * count, 0, mask);

    for (int n = 0; n < num; ++n) {
            std::vector<Dtype> vals;
            vals.assign(bottom_data, bottom_data + single_count);
#if __cplusplus > 199711L
            const std::vector<size_t> idxs = kth_element_idxs(vals, uint_k_, 0);
#else
            const std::vector<size_t> idxs = sort_idxs(vals, 0);
#endif
            for (size_t i = 0; i < uint_k_; ++i) {
                top_data[idxs[i]] =  bottom_data[idxs[i]];
                mask[idxs[i]] = static_cast<uint>(1);
              }
            mask += mask_.offset(1);
            bottom_data += bottom[0]->offset(1);
            top_data += top[0]->offset(1);
          }
      }

  template <typename Dtype>
  void TopKLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0]) {
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        const unsigned int* mask = mask_.cpu_data();
        const int count = bottom[0]->count();
        for (int i = 0; i < count; ++i) {
            bottom_diff[i] = top_diff[i] * mask[i];
          }
      }
  }

  INSTANTIATE_CLASS(TopKLayer);

}  // namespace caffe
