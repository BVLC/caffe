#include <boost/pending/disjoint_sets.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iterator>
#include <map>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/malis_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"


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
template<typename Dtype>
void MalisLossLayer<Dtype>::Malis(const Dtype* conn_data,
                                  const int_tp conn_num_dims,
                                  const int_tp* conn_dims,
                                  const int_tp* nhood_data,
                                  const int_tp* nhood_dims,
                                  const Dtype* seg_data, const bool pos,
                                  Dtype* dloss_data, Dtype* loss_out,
                                  Dtype *classerr_out, Dtype *rand_index_out) {
  if ((nhood_dims[1] != (conn_num_dims - 1))
      || (nhood_dims[0] != conn_dims[0])) {
    LOG(FATAL) << "nhood and conn dimensions don't match"
        << " (" << nhood_dims[1] << " vs. " << (conn_num_dims - 1)
        << " and " << nhood_dims[0] << " vs. "
        << conn_dims[conn_num_dims - 1] <<")";
  }

  /* Cache for speed to access neighbors */
  // nVert stores (x * y * z)
  int64_t nVert = 1;
  for (int64_t i = 1; i < conn_num_dims; ++i) {
    nVert *= conn_dims[i];
    // std::cout << i << " nVert: " << nVert << std::endl;
  }

  // prodDims stores x, x*y, x*y*z offsets
  std::vector<int64_t> prodDims(conn_num_dims - 1);
  prodDims[conn_num_dims - 2] = 1;
  for (int64_t i = 1; i < conn_num_dims - 1; ++i) {
    prodDims[conn_num_dims - 2 - i] = prodDims[conn_num_dims - 1 - i]
                                      * conn_dims[conn_num_dims - i];
    // std::cout << conn_num_dims - 2 - i << " dims: "
    //   << prodDims[conn_num_dims - 2 - i] << std::endl;
  }

  /* convert n-d offset vectors into linear array offset scalars */
  // nHood is a vector of size #edges

  std::vector<int32_t> nHood(nhood_dims[0]);
  for (int64_t i = 0; i < nhood_dims[0]; ++i) {
    nHood[i] = 0;
    for (int64_t j = 0; j < nhood_dims[1]; ++j) {
      nHood[i] += (int32_t) nhood_data[j + i * nhood_dims[1]] * prodDims[j];
    }
    // std::cout << i << " nHood: " << nHood[i] << std::endl;
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

  int64_t edgeCount = 0;
  // Loop over #edges
  for (int64_t d = 0, i = 0; d < conn_dims[0]; ++d) {
    // Loop over Z
    for (int64_t z = 0; z < conn_dims[1]; ++z) {
      // Loop over Y
      for (int64_t y = 0; y < conn_dims[2]; ++y) {
        // Loop over X
        for (int64_t x = 0; x < conn_dims[3]; ++x, ++i) {
          // Out-of-bounds check:
          if (!((z + nhood_data[d * nhood_dims[1] + 0] < 0)
              ||(z + nhood_data[d * nhood_dims[1] + 0] >= conn_dims[1])
              ||(y + nhood_data[d * nhood_dims[1] + 1] < 0)
              ||(y + nhood_data[d * nhood_dims[1] + 1] >= conn_dims[2])
              ||(x + nhood_data[d * nhood_dims[1] + 2] < 0)
              ||(x + nhood_data[d * nhood_dims[1] + 2] >= conn_dims[3]))) {
            ++edgeCount;
          }
        }
      }
    }
  }

  /* Sort all the edges in increasing order of weight */
  std::vector<int64_t> pqueue(edgeCount);
  int64_t j = 0;
  // Loop over #edges
  for (int64_t d = 0, i = 0; d < conn_dims[0]; ++d) {
    // Loop over Z
    for (int64_t z = 0; z < conn_dims[1]; ++z) {
      // Loop over Y
      for (int64_t y = 0; y < conn_dims[2]; ++y) {
        // Loop over X
        for (int64_t x = 0; x < conn_dims[3]; ++x, ++i) {
          // Out-of-bounds check:
          if (!((z + nhood_data[d * nhood_dims[1] + 0] < 0)
              ||(z + nhood_data[d * nhood_dims[1] + 0] >= conn_dims[1])
              ||(y + nhood_data[d * nhood_dims[1] + 1] < 0)
              ||(y + nhood_data[d * nhood_dims[1] + 1] >= conn_dims[2])
              ||(x + nhood_data[d * nhood_dims[1] + 2] < 0)
              ||(x + nhood_data[d * nhood_dims[1] + 2] >= conn_dims[3]))) {
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

    // e: edge dimension
    e = minEdge / nVert;

    // v1: node at edge beginning
    v1 = minEdge % nVert;

    // v2: neighborhood node at edge e
    v2 = v1 + nHood[e];

    // std::cout << "V1: " << v1 << ", V2: " << v2 << std::endl;

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
            dl = (Dtype(1.0) - conn_data[minEdge]);
            loss += dl * dl * nPair;
            // Use hinge loss
            dloss_data[minEdge] += dl * nPair;
            if (conn_data[minEdge] <= Dtype(0.5)) {  // an error
              nPairIncorrect += nPair;
            }

          } else if ((!pos) && (it1->first != it2->first)) {
            // -ve example pairs
            dl = (-conn_data[minEdge]);
            loss += dl * dl * nPair;
            // Use hinge loss
            dloss_data[minEdge] += dl * nPair;
            if (conn_data[minEdge] > Dtype(0.5)) {  // an error
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

  // std::cout << "nPairIncorrect: " << nPairIncorrect << std::endl;
  // std::cout << "nPairNorm: " << nPairNorm << std::endl;

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

  // Expected inputs:
  // Required (bottom 0 to 2):
  // Bottom 0: Predicted affinity, shaped     (batch size, #edges, (Z), (Y), X)
  // Bottom 1: Ground truth affinity, shaped  (batch size, #edges, (Z), (Y), X)
  // Bottom 2: Segmented ground truth, shaped (batch size, 1,      (Z), (Y), X)

  // Optional (bottom 3):
  // Bottom 3: Edge connectivity, size #edges * 3, shaped (Z,Y,X);(Z,Y,X);...
  // (this means pairs of 3 per edge)
}

template<typename Dtype>
void MalisLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  if (top.size() >= 2) {
    top[1]->ReshapeLike(*bottom[0]);
  }

  // Up to 5 dimensional; supported modes:
  // batch, channels (edges), Z, Y, X    => 3D affinity
  // batch, channels (edges), Y, X       => 2D affinity
  // batch, channels (edges), X          => 1D affinity
  vector<int_tp> shape = bottom[0]->shape();

  conn_dims_.clear();
  nhood_dims_.clear();

  // #edges, Z, Y, X specification (4 dimensions)
  conn_num_dims_ = 4;

  // Channel axis equals number of edges
  nedges_ = shape[1];

  // #edges
  conn_dims_.push_back(nedges_);
  // Z-axis
  conn_dims_.push_back(shape.size() >= 5 ? shape[shape.size() - 3] : 1);
  // Y-axis
  conn_dims_.push_back(shape.size() >= 4 ? shape[shape.size() - 2] : 1);
  // X-axis
  conn_dims_.push_back(shape.size() >= 3 ? shape[shape.size() - 1] : 1);

  // #edges
  nhood_dims_.push_back(nedges_);
  // 3 dimensional (always, to simplify things;
  // can just set unused spatials to 0)
  nhood_dims_.push_back(3);

  affinity_pos_.Reshape(shape);
  affinity_neg_.Reshape(shape);
  dloss_pos_.Reshape(shape);
  dloss_neg_.Reshape(shape);
}

template<typename Dtype>
void MalisLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  // Set up the neighborhood
  nhood_data_.clear();
  if (bottom.size() == 4) {
    // Custom edges
    for (int_tp i = 0; i < nedges_; ++i) {
      // Z edge direction
      nhood_data_.push_back(bottom[3]->cpu_data()[i * 3 + 0]);
      // Y edge direction
      nhood_data_.push_back(bottom[3]->cpu_data()[i * 3 + 1]);
      // X edge direction
      nhood_data_.push_back(bottom[3]->cpu_data()[i * 3 + 2]);
    }
  } else {
    // Dimension primary edges (+Z, +Y, +X) only:
    // 1 edge:    +X          (0,0,1)
    // 2 edges:   +Y, +X      (0,1,0); (0,0,1)
    // 3 edges:   +Z, +Y, +X  (1,0,0); (0,1,0); (0,0,1)
    for (int_tp i = 3 - nedges_; i < 3; ++i) {
      nhood_data_.push_back((i + 3) % 3 == 0 ? 1 : 0);
      nhood_data_.push_back((i + 2) % 3 == 0 ? 1 : 0);
      nhood_data_.push_back((i + 1) % 3 == 0 ? 1 : 0);
    }
  }

  // Predicted affinity
  const Dtype* affinity_prob = bottom[0]->cpu_data();

  // Effective affinity
  const Dtype* affinity = bottom[1]->cpu_data();

  Dtype* affinity_data_pos = affinity_pos_.mutable_cpu_data();
  Dtype* affinity_data_neg = affinity_neg_.mutable_cpu_data();

// Affinity graph must be in the range (0,1)
// square loss (euclidean) is used by MALIS
#pragma omp parallel for
  for (int_tp i = 0; i < bottom[0]->count(); ++i) {
    affinity_data_pos[i] = std::min(affinity_prob[i], affinity[i]);
    affinity_data_neg[i] = std::max(affinity_prob[i], affinity[i]);
  }

  uint_tp batch_offset = 1;
  for (int_tp i = 1; i < bottom[0]->shape().size(); ++i) {
    batch_offset *= bottom[0]->shape()[i];
  }

  uint_tp components_batch_offset = 1;
  uint_tp components_channel_offset = bottom[2]->shape()[1] == 2 ? 1 : 0;
  for (int_tp i = 1; i < bottom[2]->shape().size(); ++i) {
    components_batch_offset *= bottom[2]->shape()[i];
    if (i > 1) {
      components_channel_offset *= bottom[2]->shape()[i];
    }
  }

  Dtype loss = 0;

#pragma omp parallel for reduction(+:loss)
  for (int_tp batch = 0; batch < bottom[0]->shape()[0]; ++batch) {
    Dtype loss_out = 0;
    Dtype classerr_out = 0;
    Dtype rand_index_out = 0;

    caffe_set(dloss_neg_.count(), Dtype(0.0), dloss_neg_.mutable_cpu_data());
    caffe_set(dloss_pos_.count(), Dtype(0.0), dloss_pos_.mutable_cpu_data());

    Malis(&affinity_data_neg[batch_offset * batch], conn_num_dims_,
          &conn_dims_[0], &nhood_data_[0], &nhood_dims_[0],
          bottom[2]->cpu_data() + components_batch_offset * batch, false,
          dloss_neg_.mutable_cpu_data() + batch_offset * batch, &loss_out,
          &classerr_out, &rand_index_out);

    loss += 0.5 * loss_out;
    // std::cout << "NEG: " << loss_out << std::endl;

    Malis(&affinity_data_pos[batch_offset * batch], conn_num_dims_,
          &conn_dims_[0], &nhood_data_[0], &nhood_dims_[0],
          bottom[2]->cpu_data() + components_batch_offset * batch
          + components_channel_offset, true,
          dloss_pos_.mutable_cpu_data() + batch_offset * batch, &loss_out,
          &classerr_out, &rand_index_out);

    loss += 0.5 * loss_out;
    // std::cout << "POS: " << loss_out << std::endl;
  }

  // Normalized loss over batch size
  top[0]->mutable_cpu_data()[0] = loss
      / (static_cast<Dtype>(bottom[0]->shape()[0]));

  if (top.size() == 2) {
    top[1]->ShareData(*(bottom[0]));
  }
}

template<typename Dtype>
void MalisLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* dloss_pos_data = dloss_pos_.cpu_data();
    const Dtype* dloss_neg_data = dloss_neg_.cpu_data();

    // Clear the diff
    caffe_set(bottom[0]->count(), Dtype(0.0), bottom_diff);

#pragma omp parallel for
    for (int_tp i = 0; i < bottom[0]->count(); ++i) {
      bottom_diff[i] = -(dloss_neg_data[i] + dloss_pos_data[i]) / 2.0;
    }
  }
}

INSTANTIATE_CLASS(MalisLossLayer);
REGISTER_LAYER_CLASS(MalisLoss);

}  // namespace caffe
