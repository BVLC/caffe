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
void kth_element_idxs(const std::vector<Dtype> &v,
  std::vector<size_t> &idx, const size_t k, const int sortAscend = 1) {
  if (sortAscend) {
      std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                compare_indirect_index_ascend <std::vector<Dtype> > (v));
    } else {
      std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                compare_indirect_index_descend <std::vector<Dtype> > (v));
    }
  return;
}
#else
template <typename Dtype>
void sort_idxs(const std::vector<Dtype> &v, std::vector<size_t> &idx,
                                 const int sortAscend = 1) {
  if (sortAscend) {
      std::sort(idx.begin(), idx.end(),
                compare_indirect_index_ascend <std::vector<Dtype> > (v));
  } else {
      std::sort(idx.begin(), idx.end(),
                compare_indirect_index_descend <std::vector<Dtype> > (v));
  }
  return;
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
  channels4norm =
    this->layer_param_.topk_param().across_channels() ? bottom[0]->channels() : 1;
  DCHECK_GT(uint_k_, 0);
  DCHECK(uint_k_ <= bottom[0]->count() / bottom[0]->num());

  DCHECK_GT(channels4norm, 0);
  DCHECK(channels4norm <= bottom[0]->channels());
  }

  template <typename Dtype>
  void TopKLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    uint* mask = mask_.mutable_cpu_data();

    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();
    const int single_count = bottom[0]->count() / bottom[0]->num();

    caffe_set(count, Dtype(0), top_data);
    caffe_memset(sizeof(uint) * count, 0, mask);

    if (channels4norm > 1) {  // For convolutional layers
        for (int n = 0; n < num; ++n) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                std::vector<Dtype> vals(channels);
                std::vector<size_t> idxs(channels);
                for (int c = 0; c < channels; ++c) {
                    const size_t idx = (c * height + h) * width + w;
                    idxs[c] = idx;
                    vals[c] = bottom_data[idx];
                  }
    #if __cplusplus > 199711L
                kth_element_idxs(vals, idxs, uint_k_, 0);
    #else
                sort_idxs(vals, idxs, 0);
    #endif
                for (size_t i = 0; i < uint_k_; ++i) {
                    top_data[idxs[i]] =  bottom_data[idxs[i]];
                    mask[idxs[i]] = static_cast<uint>(1);
                  }
                }
              }
           mask += mask_.offset(1);
           bottom_data += bottom[0]->offset(1);
           top_data += top[0]->offset(1);
          }
      } else {  // For full-connected layers
    for (int n = 0; n < num; ++n) {
            std::vector<Dtype> vals;
            vals.assign(bottom_data, bottom_data + single_count);
            std::vector<size_t> idxs(vals.size());
            for (size_t i = 0; i != idxs.size(); ++i) {
                idxs[i] = i;
              }
#if __cplusplus > 199711L
            kth_element_idxs(vals, idxs, uint_k_, 0);
#else
            sort_idxs(vals, idxs, 0);
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
