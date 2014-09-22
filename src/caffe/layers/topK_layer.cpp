// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>
#include <boost/type_traits.hpp>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename T>
std::vector<unsigned int> sort_indexes(const std::vector<T> &v, const int sortAscend = 1) {

  std::vector<unsigned int> idx(v.size());
  for (unsigned int i = 0; i != idx.size(); ++i) {
      idx[i] = i;
    }
  if (sortAscend) {
     struct comparator_ascend {
        comparator_ascend(const std::vector<T>* v__) {
          v_ = v__;
        }
        bool operator () (unsigned int i1,unsigned int i2) {
          return (*v_)[i1] < (*v_)[i2];
        }
         const std::vector<T>* v_;
      };
      std::sort(idx.begin(), idx.end(),comparator_ascend(&v));
  }
  else {
      struct comparator_descend {
        comparator_descend(const std::vector<T>* v__) {
          v_ = v__;
        }
        bool operator () (unsigned int i1,unsigned int i2) {
          return (*v_)[i1] < (*v_)[i2];
        }
      const std::vector<T>* v_;
      };
std::sort(idx.begin(), idx.end(),comparator_descend(&v));
  }
  return idx;
}
  
template <typename Dtype>
void TopKLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  if (Caffe::phase() == Caffe::TRAIN) {
  uint_k_ = this->layer_param_.topk_param().k();
  }
  else
  {
  uint_k_ = this->layer_param_.topk_param().k() * this->layer_param_.topk_param().a();
  }

  DCHECK(uint_k_ > 0);
  DCHECK(uint_k_ <= bottom[0]->count() / bottom[0]->num());
  idxs_.Reshape(1, bottom[0]->channels(),
        bottom[0]->height(), bottom[0]->width());
    mask_.Reshape(bottom[0]->num(), bottom[0]->channels(),
        bottom[0]->height(), bottom[0]->width());
  }

  template <typename Dtype>
  void TopKLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    unsigned int* mask = mask_.mutable_cpu_data();

    const int count = bottom[0]->count();
    const int single_count = bottom[0]->count() / bottom[0]->num();

    caffe_set(count, Dtype(0), top_data);

    for (uint i = 0; i < count; ++i) {
        mask[i] = 0;
      }
    const int num = bottom[0]->num();
    for (int n = 0; n < num; ++n) {
            std::vector<Dtype> values;
            values.assign(bottom_data, bottom_data + single_count);

            std::vector<unsigned int> idxs = sort_indexes(values,0);

            for (uint i = 0; i < uint_k_; ++i) {
                top_data[idxs[i]] =  bottom_data[idxs[i]];
                mask[idxs[i]] = 1;
              }
            bottom_data += bottom[0]->offset(1);
            top_data += top[0]->offset(1);
            mask += mask_.offset(1);

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

#ifdef CPU_ONLY
  STUB_GPU(TopKLayer);
#endif

  INSTANTIATE_CLASS(TopKLayer);


}  // namespace caffe
