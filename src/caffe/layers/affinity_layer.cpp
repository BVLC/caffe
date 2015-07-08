#include <boost/pending/disjoint_sets.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <functional>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

// #define CAFFE_AFFINITY_DEBUG

namespace caffe {

template<typename Dtype>
void AffinityLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  offsets_.clear();
  offsets_.resize(bottom.size());
  if (this->layer_param().has_affinity_param()) {
    AffinityParameter affinity_param = this->layer_param().affinity_param();
    for (int i = 0; i <
          std::min(static_cast<int>(bottom.size()),
                   static_cast<int>(affinity_param.offset_size())); ++i) {
      offsets_[i] = affinity_param.offset(i);
    }
  }

#ifdef CAFFE_AFFINITY_DEBUG
  cv::namedWindow("prob");
  cv::namedWindow("diff");
#endif
}

template<typename Dtype>
void AffinityLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  min_index_.clear();
  for (int bidx = 0; bidx < bottom.size(); ++bidx) {
    // 1, #edges, height, width
    top[bidx]->Reshape(1, 2, bottom[bidx]->height(), bottom[bidx]->width());

    shared_ptr<Blob<Dtype> > blob_pointer(
        new Blob<Dtype>(this->device_context()));
    min_index_.push_back(blob_pointer);

    // 1, #edges, height, width
    min_index_[bidx]->Reshape(1, 2, bottom[bidx]->height(),
                              bottom[bidx]->width());
  }
}

template<typename Dtype>
void AffinityLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  for (int bidx = 0; bidx < bottom.size(); ++bidx) {
    const Dtype* bottom_data = bottom[bidx]->cpu_data();
    Dtype* top_data = top[bidx]->mutable_cpu_data();
    Dtype* min_data = min_index_[bidx]->mutable_cpu_data();

    int inner_num = bottom[bidx]->width()
                * bottom[bidx]->height();

    int xmin, ymin;

    // Construct affinity graph
#pragma omp parallel for
    for (int i = 0; i < bottom[bidx]->height() - 1; ++i) {
      for (int j = 0; j < bottom[bidx]->width() - 1; ++j) {
        // Center
        Dtype p0 = bottom_data[offsets_[bidx] * inner_num
                             + i * bottom[bidx]->width() + j];
        // Right
        Dtype p1 = bottom_data[offsets_[bidx] * inner_num
                             + i * bottom[bidx]->width() + (j + 1)];
        // Bottom
        Dtype p2 = bottom_data[offsets_[bidx] * inner_num
                             + (i + 1) * bottom[bidx]->width() + j];

        // X edge
        top_data[i * bottom[bidx]->width() + j] = std::min(p0, p1);
        xmin = p0 < p1 ? 0 : 1;
        min_data[i * bottom[bidx]->width() + j] = xmin;

        // Y edge
        top_data[inner_num
            + i * bottom[bidx]->width() + j] = std::min(p0, p2);
        ymin = p0 < p2 ? 0 : 1;
        min_data[inner_num
            + i * bottom[bidx]->width() + j] = ymin;
      }
    }
  }
}

template<typename Dtype>
void AffinityLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom) {
  for (int bidx = 0; bidx < bottom.size(); ++bidx) {
    if (propagate_down[bidx]) {
      const Dtype* top_diff = top[bidx]->cpu_diff();
      Dtype* bottom_diff = bottom[bidx]->mutable_cpu_diff();
      const Dtype* min_data = min_index_[bidx]->cpu_diff();

      caffe_set(bottom[0]->count(), Dtype(0.0), bottom_diff);

      int inner_num = bottom[bidx]->width()
                  * bottom[bidx]->height();

      // Spread out the affinity losses to pixels
      for (int i = 0; i < bottom[0]->height() - 1; ++i) {
        for (int j = 0; j < bottom[0]->width() - 1; ++j) {
          Dtype lx = top_diff[i * bottom[0]->width() + j];
          Dtype ly = top_diff[inner_num + i * bottom[0]->width() + j];

          int mx = min_data[i * bottom[0]->width() + j];
          int my = min_data[bottom[0]->width()
              * bottom[0]->height() + i * bottom[0]->width() + j];

          // Only propagate to min index contributor of affinity graph
          bottom_diff[0 * inner_num + i * bottom[0]->width() + (j + mx)] -= lx;
          bottom_diff[0 * inner_num + (i + my) * bottom[0]->width() + j] -= ly;
          bottom_diff[1 * inner_num + i * bottom[0]->width() + (j + mx)] += lx;
          bottom_diff[1 * inner_num + (i + my) * bottom[0]->width() + j] += ly;
        }
      }
#ifdef CAFFE_AFFINITY_DEBUG
      {
        cv::Mat tmp;

        Dtype* prob_rd = bottom[bidx]->mutable_cpu_data();

        cv::Mat wrapped_prob(bottom[0]->height(), bottom[0]->width(),
                          cv::DataType<Dtype>::type,
                        prob_rd, sizeof(Dtype) * bottom[0]->width());
        cv::imshow("prob", wrapped_prob);

        cv::Mat wrapped_diff(bottom[0]->height(), bottom[0]->width(),
                          cv::DataType<Dtype>::type,
                        bottom_diff, sizeof(Dtype) * bottom[0]->width());

        Dtype sum = std::accumulate(bottom_diff,
                                    bottom_diff
                                    + bottom[0]->height() * bottom[0]->width(),
                                    0.0);

        Dtype mean = sum / (bottom[0]->width()*bottom[0]->height());

        std::vector<Dtype> msd(bottom[0]->height() * bottom[0]->width());
        std::transform(bottom_diff,
                       bottom_diff + (bottom[0]->height()*bottom[0]->width()),
                       msd.begin(), std::bind2nd(std::minus<Dtype>(), mean));

        Dtype sqsum = std::inner_product(msd.begin(),
                                         msd.end(), msd.begin(), 0.0);
        Dtype stdev = std::sqrt(sqsum / (bottom[0]->width()
            * bottom[0]->height()));

        wrapped_diff.convertTo(tmp, CV_32FC1, 1.0 / (2.0 * stdev),
            (stdev - mean) * 1.0 / (2.0 * stdev));

        cv::imshow("diff", tmp);
        cv::waitKey(2);
      }
#endif
    }
  }
}

INSTANTIATE_CLASS(AffinityLayer);
REGISTER_LAYER_CLASS(Affinity);

}  // namespace caffe
