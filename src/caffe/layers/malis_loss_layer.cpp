#include <boost/pending/disjoint_sets.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <map>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

// #define CAFFE_MALIS_DEBUG

namespace caffe {

template<class Dtype>
class MalisAffinityGraphCompare {
 private:
  const Dtype * mEdgeWeightArray;
 public:
  explicit MalisAffinityGraphCompare(const Dtype * EdgeWeightArray) {
    mEdgeWeightArray = EdgeWeightArray;
  }
  bool operator()(const int64_t& ind1, const int64_t& ind2) const {
    return (mEdgeWeightArray[ind1] > mEdgeWeightArray[ind2]);
  }
};

// Derived from https://github.com/srinituraga/malis/blob/master/matlab/malis_loss_mex.cpp
// conn_data:   4d connectivity graph [y * x * z * #edges]
// nhood_data:  graph neighborhood descriptor [3 * #edges]
// seg_data:    true target segmentation [y * x * z]
// pos:         is this a positive example pass [true] or
//              a negative example pass [false] ?
// margin:      sq-sq loss margin [0.3]
template<typename Dtype>
void MalisLossLayer<Dtype>::Malis(const Dtype* conn_data,
                                  const int conn_num_dims,
                                  const int* conn_dims, const int* nhood_data,
                                  const int* nhood_dims, const Dtype* seg_data,
                                  const bool pos,
                                  Dtype* dloss_data, Dtype* loss_out,
                                  Dtype *classerr_out, Dtype *rand_index_out,
                                  Dtype margin, Dtype threshold) {
  if ((nhood_dims[1] != (conn_num_dims - 1))
      || (nhood_dims[0] != conn_dims[conn_num_dims - 1])) {
    LOG(FATAL) << "nhood and conn dimensions don't match";
  }

  /* Cache for speed to access neighbors */
  // nVert stores (x * y * z)
  int64_t nVert = 1;
  for (int64_t i = 0; i < conn_num_dims - 1; ++i) {
    nVert = nVert * conn_dims[i];
  }

  // prodDims stores x, x*y, x*y*z offsets
  std::vector<int64_t> prodDims(conn_num_dims - 1);
  prodDims[0] = 1;
  for (int64_t i = 1; i < conn_num_dims - 1; ++i) {
    prodDims[i] = prodDims[i - 1] * conn_dims[i - 1];
  }

  /* convert n-d offset vectors into linear array offset scalars */
  // nHood is a vector of size #edges
  std::vector<int32_t> nHood(nhood_dims[0]);
  for (int64_t i = 0; i < nhood_dims[0]; ++i) {
    nHood[i] = 0;
    for (int64_t j = 0; j < nhood_dims[1]; ++j) {
      nHood[i] += (int32_t) nhood_data[i + j * nhood_dims[0]] * prodDims[j];
    }
  }

  /* Disjoint sets and sparse overlap vectors */
  std::vector<std::map<int64_t, int64_t> > overlap(nVert);
  std::vector<int64_t> rank(nVert);
  std::vector<int64_t> parent(nVert);
  std::map<int64_t, int64_t> segSizes;
  int64_t nLabeledVert = 0;
  int64_t nPairPos = 0;
  boost::disjoint_sets<int64_t*, int64_t*> dsets(&rank[0], &parent[0]);
  // Loop over all seg data items
  for (int64_t i = 0; i < nVert; ++i) {
    dsets.make_set(i);
    if (0 != seg_data[i]) {
      overlap[i].insert(std::pair<int64_t, int64_t>(seg_data[i], 1));
      ++nLabeledVert;
      ++segSizes[seg_data[i]];
      nPairPos += (segSizes[seg_data[i]] - 1);
    }
  }
  int64_t nPairTot = (nLabeledVert * (nLabeledVert - 1)) / 2;
  int64_t nPairNeg = nPairTot - nPairPos;
  int64_t nPairNorm;
  if (pos) {
    nPairNorm = nPairPos;
  } else {
    nPairNorm = nPairNeg;
  }

  /* Sort all the edges in increasing order of weight */
  std::vector<int64_t> pqueue(
      conn_dims[3] * std::max((conn_dims[0] - 1), 1)
                   * std::max((conn_dims[1] - 1), 1)
                   * std::max((conn_dims[2] - 1), 1));
  int64_t j = 0;
  // Loop over #edges
  for (int64_t d = 0, i = 0; d < conn_dims[3]; ++d) {
    // Loop over Z
    for (int64_t z = 0; z < conn_dims[2]; ++z) {
      // Loop over Y
      for (int64_t y = 0; y < conn_dims[1]; ++y) {
        // Loop over X
        for (int64_t x = 0; x < conn_dims[0]; ++x, ++i) {
          if (x < std::max(conn_dims[0] - 1, 1) &&
              y < std::max(conn_dims[1] - 1, 1) &&
              z < std::max(conn_dims[2] - 1, 1)) {
            pqueue[j++] = i;
          }
        }
      }
    }
  }

  pqueue.resize(j);

  std::sort(pqueue.begin(), pqueue.end(),
       MalisAffinityGraphCompare<Dtype>(conn_data));

  /* Start MST */
  int64_t minEdge;
  int64_t e, v1, v2;
  int64_t set1, set2;
  int64_t nPair = 0;
  double loss = 0, dl = 0;
  int64_t nPairIncorrect = 0;
  std::map<int64_t, int64_t>::iterator it1, it2;

  /* Start Kruskal's */
  for (int64_t i = 0; i < pqueue.size(); ++i) {
    minEdge = pqueue[i];
    // nVert = x * y * z, minEdge in [0, x * y * z * #edges]

    // e: edge dimension (0: X, 1: Y, 2: Z)
    e = minEdge / nVert;

    // v1: node at edge beginning
    v1 = minEdge % nVert;

    // v2: neighborhood node at edge e
    v2 = v1 + nHood[e];

    set1 = dsets.find_set(v1);
    set2 = dsets.find_set(v2);


    if (set1 != set2) {
      dsets.link(set1, set2);

      /* compute the dloss for this MST edge */
      for (it1 = overlap[set1].begin(); it1 != overlap[set1].end(); ++it1) {
        for (it2 = overlap[set2].begin(); it2 != overlap[set2].end(); ++it2) {
          nPair = it1->second * it2->second;

          if (pos && (it1->first == it2->first)) {
            // +ve example pairs
            dl = std::max(Dtype(0.0), threshold + margin - conn_data[minEdge]);
            loss += dl * nPair;
            // Use hinge loss
            dloss_data[minEdge] -= dl * nPair;
            if (conn_data[minEdge] <= threshold) {  // an error
              nPairIncorrect += nPair;
            }

          } else if ((!pos) && (it1->first != it2->first)) {
            // -ve example pairs
            dl = std::max(Dtype(0.0), conn_data[minEdge] - threshold + margin);
            loss += dl * nPair;
            // Use hinge loss
            dloss_data[minEdge] += dl * nPair;
            if (conn_data[minEdge] > threshold) {  // an error
              nPairIncorrect += nPair;
            }
          }
        }
      }

      if (nPairNorm > 0) {
        dloss_data[minEdge] /= nPairNorm;
      } else {
        dloss_data[minEdge] = 0;
      }

      if (dsets.find_set(set1) == set2) {
        std::swap(set1, set2);
      }

      for (it2 = overlap[set2].begin();
          it2 != overlap[set2].end(); ++it2) {
        it1 = overlap[set1].find(it2->first);
        if (it1 == overlap[set1].end()) {
          overlap[set1].insert(pair<int64_t, int64_t>
            (it2->first, it2->second));
        } else {
          it1->second += it2->second;
        }
      }
      overlap[set2].clear();
    }  // end link
  }  // end while

  /* Return items */
  double classerr, randIndex;
  if (nPairNorm > 0) {
    loss /= nPairNorm;
  } else {
    loss = 0;
  }
  *loss_out = loss;
  classerr = static_cast<double>(nPairIncorrect)
      / static_cast<double>(nPairNorm);
  *classerr_out = classerr;
  randIndex = 1.0 - static_cast<double>(nPairIncorrect)
      / static_cast<double>(nPairNorm);
  *rand_index_out = randIndex;
}


template<typename Dtype>
void MalisLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

#ifdef CAFFE_MALIS_DEBUG
  cv::namedWindow("labelled");
  cv::namedWindow("test");
#endif
}

template<typename Dtype>
void MalisLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  if (top.size() >= 2) {
    top[1]->ReshapeLike(*bottom[0]);
  }

  conn_dims_.clear();
  nhood_dims_.clear();
  nhood_data_.clear();

  conn_num_dims_ = 4;
  conn_dims_.push_back(bottom[0]->width());       // X-axis
  conn_dims_.push_back(bottom[0]->height());      // Y-axis
  conn_dims_.push_back(1);                        // Z-axis
  conn_dims_.push_back(2);                        // #edges

  nhood_dims_.push_back(2);                       // #edges
  nhood_dims_.push_back(3);                       // 3 dimensional

  nhood_data_.push_back(1);                       // Edge 1, X
  nhood_data_.push_back(0);                       // Edge 2, X

  nhood_data_.push_back(0);                       // Edge 1, Y
  nhood_data_.push_back(1);                       // Edge 2, Y

  nhood_data_.push_back(0);                       // Edge 1, Z
  nhood_data_.push_back(0);                       // Edge 2, Z

  dloss_pos_.Reshape(
      1, 2, bottom[0]->height(), bottom[0]->width());
  dloss_neg_.Reshape(
      1, 2, bottom[0]->height(), bottom[0]->width());
}

template<typename Dtype>
void MalisLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
#ifdef CAFFE_MALIS_DEBUG
  // This is for debugging only:
  {
    std::vector<int> labels;
    const Dtype* seg_data = bottom[2]->cpu_data();
    for (int i = 0; i < bottom[2]->height() * bottom[2]->width(); ++i) {
      int val = static_cast<int>(seg_data[i]);
      bool found = false;
      for (int j = 0; j < labels.size(); ++j) {
        if (val == labels[j]) {
          found = true;
        }
      }
      if (found == false) {
        labels.push_back(val);
      }
    }

    std::vector<cv::Vec3b> colors;

    for (int i = 0; i < labels.size(); ++i) {
      unsigned char r = 255 * (rand() / (1.0 + RAND_MAX));  // NOLINT
      unsigned char g = 255 * (rand() / (1.0 + RAND_MAX));  // NOLINT
      unsigned char b = 255 * (rand() / (1.0 + RAND_MAX));  // NOLINT

      cv::Vec3b color(r, g, b);
      colors.push_back(color);
    }

    cv::Mat output = cv::Mat::zeros(cv::Size(bottom[1]->height(),
                                             bottom[1]->width()), CV_8UC3);

    const Dtype* imgdata = bottom[2]->cpu_data();

    for (int i = 0; i < bottom[1]->height() * bottom[1]->width(); ++i) {
      int val = imgdata[i];
      if (val == 0) {
        output.at<cv::Vec3b>(i) = cv::Vec3b(0, 0, 0);
        continue;
      }
      for (int j = 0; j < labels.size(); ++j) {
        if (val == labels[j]) {
          output.at<cv::Vec3b>(i) = colors[j];
        }
      }
    }
    cv::imshow("labelled", output);
  }
#endif

  int inner_num = bottom[0]->width() * bottom[0]->height();

  // Predicted affinity
  const Dtype* affinity_prob_x = bottom[0]->cpu_data();
  const Dtype* affinity_prob_y = bottom[0]->cpu_data() + inner_num;

  // Effective affinity
  const Dtype* affinity_x = bottom[1]->cpu_data();
  const Dtype* affinity_y = bottom[1]->cpu_data() + inner_num;

#ifdef CAFFE_MALIS_DEBUG
  {Dtype* prob_rd = bottom[0]->mutable_cpu_data();
  cv::Mat wrapped(bottom[0]->height(), bottom[0]->width(),
                    cv::DataType<Dtype>::type,
                  prob_rd, sizeof(Dtype) * bottom[0]->width());
  cv::imshow("test", wrapped);}
#endif

  // Connection data
  std::vector<Dtype> conn_data_pos(
      2 * bottom[0]->height() * bottom[0]->width());
  std::vector<Dtype> conn_data_neg(
      2 * bottom[0]->height() * bottom[0]->width());

  // Construct positive and negative affinity graph
#pragma omp parallel for
  for (int i = 0; i < bottom[0]->height() - 1; ++i) {
    for (int j = 0; j < bottom[0]->width() - 1; ++j) {
      // X positive
      conn_data_pos[i * bottom[0]->width() + j] = std::min(
          affinity_prob_x[i * bottom[0]->width() + j],
          affinity_x[i * bottom[0]->width() + j]);

      // X negative
      conn_data_neg[i * bottom[0]->width() + j] = std::max(
          affinity_prob_x[i * bottom[0]->width() + j],
          affinity_x[i * bottom[0]->width() + j]);

      // Y positive
      conn_data_pos[inner_num
          + i * bottom[0]->width() + j] = std::min(
          affinity_prob_y[i * bottom[0]->width() + j],
          affinity_y[i * bottom[0]->width() + j]);

      // Y negative
      conn_data_neg[inner_num
          + i * bottom[0]->width() + j] = std::max(
          affinity_prob_y[i * bottom[0]->width() + j],
          affinity_y[i * bottom[0]->width() + j]);
    }
  }

  Dtype loss = 0;

  Dtype loss_out = 0;
  Dtype classerr_out = 0;
  Dtype rand_index_out = 0;

  caffe_set(dloss_neg_.count(), Dtype(0.0), dloss_neg_.mutable_cpu_data());
  caffe_set(dloss_pos_.count(), Dtype(0.0), dloss_pos_.mutable_cpu_data());

  Malis(&conn_data_neg[0], conn_num_dims_, &conn_dims_[0], &nhood_data_[0],
        &nhood_dims_[0], bottom[2]->cpu_data(),
        false, dloss_neg_.mutable_cpu_data(),
        &loss_out, &classerr_out, &rand_index_out, 0.3, 0.5);

  loss += loss_out;

  Malis(&conn_data_pos[0], conn_num_dims_, &conn_dims_[0], &nhood_data_[0],
        &nhood_dims_[0], bottom[2]->cpu_data(),
        true, dloss_pos_.mutable_cpu_data(),
        &loss_out, &classerr_out, &rand_index_out, 0.3, 0.5);

  loss += loss_out;

  top[0]->mutable_cpu_data()[0] = loss;

  if (top.size() == 2) {
    top[1]->ShareData(*(bottom[0]));
  }
}

template<typename Dtype>
void MalisLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    // Diff to propagate to (size w * h * c)
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* dloss_pos_data = dloss_pos_.cpu_data();
    const Dtype* dloss_neg_data = dloss_neg_.cpu_data();

    // Clear the diff
    caffe_set(bottom[0]->count(), Dtype(0.0), bottom_diff);

    int inner_num = bottom[0]->height() * bottom[0]->width();

#pragma omp parallel for
    for (int i = 0; i < bottom[0]->height(); ++i) {
      for (int j = 0; j < bottom[0]->width(); ++j) {
        bottom_diff[i * bottom[0]->width() + j] =
            dloss_pos_data[i * bottom[0]->width() + j] +
            dloss_neg_data[i * bottom[0]->width() + j];
        bottom_diff[inner_num + i * bottom[0]->width() + j] =
            dloss_pos_data[inner_num + i * bottom[0]->width() + j] +
            dloss_neg_data[inner_num + i * bottom[0]->width() + j];
      }
    }
  }
}

INSTANTIATE_CLASS(MalisLossLayer);
REGISTER_LAYER_CLASS(MalisLoss);

}  // namespace caffe
