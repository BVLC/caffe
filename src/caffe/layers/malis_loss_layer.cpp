#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <boost/pending/disjoint_sets.hpp>
#include <vector>
#include <queue>
#include <map>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {

template <class Dtype>
class MalisAffinityGraphCompare{
  private:
  const Dtype * mEdgeWeightArray;
  public:
    MalisAffinityGraphCompare(const Dtype * EdgeWeightArray){
      mEdgeWeightArray = EdgeWeightArray;
    }
    bool operator() (const int ind1, const int ind2) const {
      return (mEdgeWeightArray[ind1] > mEdgeWeightArray[ind2]);
    }
};


// Derived from https://github.com/srinituraga/malis/blob/master/matlab/malis_loss_mex.cpp
// conn_data:   4d connectivity graph [y * x * z * #edges]
// nhood_data:  graph neighborhood descriptor [3 * #edges]
// seg_data:    true target segmentation [y * x * z]
// pos:         is this a positive example pass [true] or a negative example pass [false] ?
// margin:      sq-sq loss margin [0.3]
template <typename Dtype>
void MalisLossLayer<Dtype>::Malis(Dtype* conn_data, int conn_num_dims, int* conn_dims, int conn_num_elements,
                                  Dtype* nhood_data, int nhood_num_dims, int* nhood_dims,
                                  int* seg_data, int seg_num_dims, int* seg_dims, int seg_num_elements,
                                  bool pos, Dtype* dloss_data, Dtype* loss_out, Dtype *classerr_out, Dtype *rand_index_out,
                                  Dtype margin) {
  if (nhood_num_dims != 2) {
    LOG(FATAL)<<"wrong size for nhood";
  }
  if ((nhood_dims[1] != (conn_num_dims - 1))
      || (nhood_dims[0] != conn_dims[conn_num_dims - 1])) {
    LOG(FATAL)<<"nhood and conn dimensions don't match";
  }

  /* Cache for speed to access neighbors */
  int nVert = 1;
  for (int i = 0; i < conn_num_dims - 1; ++i)
    nVert = nVert * conn_dims[i];

  vector<int> prodDims(conn_num_dims - 1);
  prodDims[0] = 1;
  for (int i = 1; i < conn_num_dims - 1; ++i)
    prodDims[i] = prodDims[i - 1] * conn_dims[i - 1];

  /* convert n-d offset vectors into linear array offset scalars */
  vector<int32_t> nHood(nhood_dims[0]);
  for (int i = 0; i < nhood_dims[0]; ++i) {
    nHood[i] = 0;
    for (int j = 0; j < nhood_dims[1]; ++j) {
      nHood[i] += (int32_t) nhood_data[i + j * nhood_dims[0]] * prodDims[j];
    }
  }

  /* Disjoint sets and sparse overlap vectors */
  vector<map<int, int> > overlap(nVert);
  vector<int> rank(nVert);
  vector<int> parent(nVert);
  map<int, int> segSizes;
  int nLabeledVert = 0;
  int nPairPos = 0;
  boost::disjoint_sets<int*, int*> dsets(&rank[0], &parent[0]);
  for (int i = 0; i < nVert; ++i) {
    dsets.make_set(i);
    if (0 != seg_data[i]) {
      overlap[i].insert(pair<int, int>(seg_data[i], 1));
      ++nLabeledVert;
      ++segSizes[seg_data[i]];
      nPairPos += (segSizes[seg_data[i]] - 1);
    }
  }
  int nPairTot = (nLabeledVert * (nLabeledVert - 1)) / 2;
  int nPairNeg = nPairTot - nPairPos;
  int nPairNorm;
  if (pos) {
    nPairNorm = nPairPos;
  } else {
    nPairNorm = nPairNeg;
  }

  /* Sort all the edges in increasing order of weight */
  std::vector<int> pqueue(
      static_cast<int>(3) * (conn_dims[0] - 1) * (conn_dims[1] - 1)
          * (conn_dims[2] - 1));
  int j = 0;
  for (int d = 0, i = 0; d < conn_dims[3]; ++d)
    for (int z = 0; z < conn_dims[2]; ++z)
      for (int y = 0; y < conn_dims[1]; ++y)
        for (int x = 0; x < conn_dims[0]; ++x, ++i) {
          if (x > 0 && y > 0 && z > 0)
            pqueue[j++] = i;
        }
  sort(pqueue.begin(), pqueue.end(), MalisAffinityGraphCompare<Dtype>(conn_data));

  /* Start MST */
  int minEdge;
  int e, v1, v2;
  int set1, set2;
  int nPair = 0;
  double loss = 0, dl = 0;
  int nPairIncorrect = 0;
  map<int, int>::iterator it1, it2;

  /* Start Kruskal's */
  for (int i = 0; i < pqueue.size(); ++i) {
    minEdge = pqueue[i];
    e = minEdge / nVert;
    v1 = minEdge % nVert;
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
            // Sq-Sq loss is used here
            dl = std::max(0.0, 0.5 + margin - conn_data[minEdge]);
            loss += 0.5 * dl * dl * nPair;
            dloss_data[minEdge] += dl * nPair;
            if (conn_data[minEdge] <= 0.5) {  // an error
              nPairIncorrect += nPair;
            }

          } else if ((!pos) && (it1->first != it2->first)) {
            // -ve example pairs
            // Sq-Sq loss is used here
            dl = -std::max(0.0, conn_data[minEdge] - 0.5 + margin);
            loss += 0.5 * dl * dl * nPair;
            dloss_data[minEdge] += dl * nPair;
            if (conn_data[minEdge] > 0.5) {  // an error
              nPairIncorrect += nPair;
            }
          }
        }
      }
      dloss_data[minEdge] /= nPairNorm;
      /* HARD-CODED ALERT!!
       * The derivative of the activation function is also multiplied here.
       * Assumes the logistic nonlinear activation function.
       */
      dloss_data[minEdge] *= conn_data[minEdge] * (1 - conn_data[minEdge]);  // DSigmoid

      /* move the pixel bags of the non-representative to the representative */
      if (dsets.find_set(set1) == set2)  // make set1 the rep to keep and set2 the rep to empty
        std::swap(set1, set2);

      it2 = overlap[set2].begin();
      while (it2 != overlap[set2].end()) {
        it1 = overlap[set1].find(it2->first);
        if (it1 == overlap[set1].end()) {
          overlap[set1].insert(pair<int, int>(it2->first, it2->second));
        } else {
          it1->second += it2->second;
        }
        overlap[set2].erase(it2++);
      }
    }  // end link
  }  // end while

  /* Return items */
  double classerr, randIndex;
  loss /= nPairNorm;
  *loss_out = loss;
  classerr = (double) nPairIncorrect / (double) nPairNorm;
  *classerr_out = classerr;
  randIndex = 1.0 - ((double) nPairIncorrect / (double) nPairNorm);
  *rand_index_out = randIndex;
}


// Derived from http://nghiaho.com/uploads/code/opencv_connected_component/blob.cpp
template <typename Dtype>
void MalisLossLayer<Dtype>::FindBlobs(const cv::Mat &binary,
               std::vector<std::vector<cv::Point2i> > &blobs) {
  blobs.clear();

// Fill the label_image with the blobs
// 0  - background
// 1  - unlabelled foreground
// 2+ - labelled foreground

  cv::Mat label_image;
  binary.convertTo(label_image, CV_32SC1);

  int label_count = 2;  // starts at 2 because 0,1 are used already

  for (int y = 0; y < label_image.rows; y++) {
    int *row = (int*) label_image.ptr(y);
    for (int x = 0; x < label_image.cols; x++) {
      if (row[x] != 1) {
        continue;
      }

      cv::Rect rect;
      cv::floodFill(label_image, cv::Point(x, y), label_count, &rect, 0, 0, 4);

      std::vector<cv::Point2i> blob;

      for (int i = rect.y; i < (rect.y + rect.height); i++) {
        int *row2 = (int*) label_image.ptr(i);
        for (int j = rect.x; j < (rect.x + rect.width); j++) {
          if (row2[j] != label_count) {
            continue;
          }

          blob.push_back(cv::Point2i(j, i));
        }
      }

      blobs.push_back(blob);

      label_count++;
    }
  }
}



template <typename Dtype>
void MalisLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  // Set up the softmax layer
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
}

template <typename Dtype>
void MalisLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void MalisLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  int count = 0;
  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                           Dtype(FLT_MIN)));
      ++count;
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / count;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void MalisLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }

  if (propagate_down[0]) {

    // Diff to propagate to (size w * h * c)
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    // The predictions (size w * h * c)
    const Dtype* prob_data = prob_.cpu_data();

    // Labels (size w * h, c values)
    const Dtype* label = bottom[1]->cpu_data();


    caffe_cpu_copy(prob_.count(), prob_data, bottom_diff);
    int dim = prob_.count() / outer_num_;
    int count = 0;
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
        ++count;
      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(prob_.count(), loss_weight / count, bottom_diff);
  }
}


INSTANTIATE_CLASS(MalisLossLayer);
REGISTER_LAYER_CLASS(MalisLoss);

}  // namespace caffe
