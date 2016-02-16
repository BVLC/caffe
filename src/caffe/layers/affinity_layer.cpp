#include <boost/pending/disjoint_sets.hpp>

#include <algorithm>
#include <functional>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/layers/affinity_layer.hpp"

namespace caffe {

template<typename Dtype>
void AffinityLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  offsets_.clear();
  offsets_.resize(bottom.size());
  if (this->layer_param().has_affinity_param()) {
    AffinityParameter affinity_param = this->layer_param().affinity_param();
    for (int_tp i = 0; i <
          std::min(static_cast<int_tp>(bottom.size()),
                   static_cast<int_tp>(affinity_param.offset_size())); ++i) {
      offsets_[i] = affinity_param.offset(i);
    }
  }
}

template<typename Dtype>
void AffinityLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  min_index_.clear();
  for (int_tp bidx = 0; bidx < bottom.size(); ++bidx) {
    // 1, #edges, height, width
    top[bidx]->Reshape(1, 2, bottom[bidx]->height(), bottom[bidx]->width());

    shared_ptr<Blob<Dtype> > blob_pointer(
        new Blob<Dtype>(this->get_device()));
    min_index_.push_back(blob_pointer);

    // 1, #edges, height, width
    min_index_[bidx]->Reshape(1, 2, bottom[bidx]->height(),
                              bottom[bidx]->width());
  }
}

template<typename Dtype>
void AffinityLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  for (int_tp bidx = 0; bidx < bottom.size(); ++bidx) {
    const Dtype* bottom_data = bottom[bidx]->cpu_data();
    Dtype* top_data = top[bidx]->mutable_cpu_data();
    Dtype* min_data = min_index_[bidx]->mutable_cpu_data();

    int_tp inner_num = bottom[bidx]->width()
                * bottom[bidx]->height();

    int_tp xmin, ymin;

    // Construct affinity graph
#pragma omp parallel for
    for (int_tp i = 0; i < bottom[bidx]->height() - 1; ++i) {
      for (int_tp j = 0; j < bottom[bidx]->width() - 1; ++j) {
        // Center
        Dtype p0 = bottom_data[offsets_[bidx] * inner_num
                             + i * bottom[bidx]->width() + j];
        // Right
        Dtype p1 = bottom_data[offsets_[bidx] * inner_num
                             + i * bottom[bidx]->width() + (j + 1)];
        // Bottom
        Dtype p2 = bottom_data[offsets_[bidx] * inner_num
                             + (i + 1) * bottom[bidx]->width() + j];

        // Y edge
        top_data[i * bottom[bidx]->width() + j] = std::min(p0, p2);
        ymin = p0 < p2 ? 0 : 1;
        min_data[i * bottom[bidx]->width() + j] = ymin;

        // X edge
        top_data[inner_num
            + i * bottom[bidx]->width() + j] = std::min(p0, p1);
        xmin = p0 < p1 ? 0 : 1;
        min_data[inner_num
            + i * bottom[bidx]->width() + j] = xmin;
      }
    }
  }
}

template<typename Dtype>
void AffinityLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom) {
  for (int_tp bidx = 0; bidx < bottom.size(); ++bidx) {
    if (propagate_down[bidx]) {
      const Dtype* top_diff = top[bidx]->cpu_diff();
      Dtype* bottom_diff = bottom[bidx]->mutable_cpu_diff();
      const Dtype* min_data = min_index_[bidx]->cpu_diff();

      caffe_set(bottom[0]->count(), Dtype(0.0), bottom_diff);

      int_tp inner_num = bottom[bidx]->width()
                  * bottom[bidx]->height();

      // Spread out the affinity losses to pixels
      for (int_tp i = 0; i < bottom[0]->height() - 1; ++i) {
        for (int_tp j = 0; j < bottom[0]->width() - 1; ++j) {
          Dtype ly = top_diff[i * bottom[0]->width() + j];
          Dtype lx = top_diff[inner_num + i * bottom[0]->width() + j];

          int_tp my = min_data[i * bottom[0]->width() + j];
          int_tp mx = min_data[bottom[0]->width()
              * bottom[0]->height() + i * bottom[0]->width() + j];

          // Only propagate to min index contributor of affinity graph
          bottom_diff[offsets_[bidx]
                     * inner_num + i * bottom[0]->width() + (j + mx)] += lx;
          bottom_diff[offsets_[bidx]
                     * inner_num + (i + my) * bottom[0]->width() + j] += ly;
          bottom_diff[((offsets_[bidx] + 1) % 2)
                     * inner_num + i * bottom[0]->width() + (j + mx)] -= lx;
          bottom_diff[((offsets_[bidx] + 1) % 2)
                     * inner_num + (i + my) * bottom[0]->width() + j] -= ly;
        }
      }
    }
  }
}

INSTANTIATE_CLASS(AffinityLayer);
REGISTER_LAYER_CLASS(Affinity);

}  // namespace caffe
