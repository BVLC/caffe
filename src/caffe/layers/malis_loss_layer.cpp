#include <boost/pending/disjoint_sets.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <map>
#include <queue>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<class Dtype>
class MalisAffinityGraphCompare {
 private:
  const Dtype * mEdgeWeightArray;
 public:
  explicit MalisAffinityGraphCompare(const Dtype * EdgeWeightArray) {
    mEdgeWeightArray = EdgeWeightArray;
  }
  bool operator()(const int ind1, const int ind2) const {
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
void MalisLossLayer<Dtype>::Malis(Dtype* conn_data, int conn_num_dims,
                                  int* conn_dims, int* nhood_data,
                                  int* nhood_dims, int* seg_data, bool pos,
                                  Dtype* dloss_data, Dtype* loss_out,
                                  Dtype *classerr_out, Dtype *rand_index_out,
                                  Dtype margin) {
  if ((nhood_dims[1] != (conn_num_dims - 1))
      || (nhood_dims[0] != conn_dims[conn_num_dims - 1])) {
    LOG(FATAL) << "nhood and conn dimensions don't match";
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
  sort(pqueue.begin(), pqueue.end(),
       MalisAffinityGraphCompare<Dtype>(conn_data));

  /* Start MST */
  int minEdge;
  int e, v1, v2;
  int set1, set2;
  int nPair = 0;
  double loss = 0, dl = 0;
  int nPairIncorrect = 0;
  map<int, int>::iterator it1, it2;

  std::cout << "Pqueue size: " << pqueue.size() << std::endl;

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
      // dloss_data[minEdge] *=
      // conn_data[minEdge] * (1 - conn_data[minEdge]);  // DSigmoid
      // Don't pre-multiply derivative, will be done
      // later in the softmax backward
      /* move the pixel bags of the non-representative to the representative */
      // make set1 the rep to keep and set2 the rep to empty
      if (dsets.find_set(set1) == set2)
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
  classerr = static_cast<double>(nPairIncorrect)
      / static_cast<double>(nPairNorm);
  *classerr_out = classerr;
  randIndex = 1.0 - static_cast<double>(nPairIncorrect)
      / static_cast<double>(nPairNorm);
  *rand_index_out = randIndex;
}

// Derived from
// http://nghiaho.com/uploads/code/opencv_connected_component/blob.cpp
template<typename Dtype>
cv::Mat MalisLossLayer<Dtype>::FindBlobs(
    const cv::Mat &input, std::vector<std::vector<cv::Point2i> > *blobs) {
  blobs->clear();

  // Fill the label_image with the blobs

  cv::Mat label_image;
  input.convertTo(label_image, CV_32SC1);

  // Segment into label numbers higher than the original label numbers
  int label_count = prob_.channels();

  for (int y = 0; y < label_image.rows; y++) {
    int *row = reinterpret_cast<int*>(label_image.ptr(y));
    for (int x = 0; x < label_image.cols; x++) {
      // Skip background and already labeled areas
      if (row[x] >= prob_.channels() || row[x] == 0) {
        continue;
      }

      cv::Rect rect;
      cv::floodFill(label_image, cv::Point(x, y), label_count, &rect, 0, 0, 4);

      std::vector<cv::Point2i> blob;

#pragma omp parallel for
      for (int i = rect.y; i < (rect.y + rect.height); i++) {
        int *row2 = reinterpret_cast<int*>(label_image.ptr(i));
        for (int j = rect.x; j < (rect.x + rect.width); j++) {
          if (row2[j] != label_count) {
            continue;
          }
#pragma omp critical
          blob.push_back(cv::Point2i(j, i));
        }
      }

      blobs->push_back(blob);

      label_count++;
    }
  }

  return label_image;
}

template<typename Dtype>
void MalisLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
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

template<typename Dtype>
void MalisLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.softmax_param().axis());
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

  conn_num_dims_ = 4;
  conn_dims_.push_back(bottom[0]->width() - 1);   // X-axis
  conn_dims_.push_back(bottom[0]->height() - 1);  // Y-axis
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
}

template<typename Dtype>
void MalisLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
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
      loss -= log(
          std::max(prob_data[i * dim + label_value * inner_num_ + j],
                   Dtype(FLT_MIN)));
      ++count;
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / count;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template<typename Dtype>
void MalisLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom) {
  std::cout << "Outer dim: " << outer_num_ << std::endl;
  std::cout << "Inner dim: " << inner_num_ << std::endl;

  if (propagate_down[1]) {
    LOG(FATAL)<< this->type()
    << " Layer cannot backpropagate to label inputs.";
  }

  if (propagate_down[0]) {
    // Diff to propagate to (size w * h * c)
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    // The predictions (size w * h * c)
    const Dtype* prob_data = prob_.cpu_data();

    // Labels (size w * h, c values)
    const Dtype* label = bottom[1]->cpu_data();

    cv::namedWindow("labelled");
    cv::namedWindow("prob");
    cv::namedWindow("diff");

    cv::Mat img(bottom[1]->height(), bottom[1]->width(), CV_8SC1);
#pragma omp parallel for
    for (int y = 0; y < bottom[1]->height(); ++y) {
      for (int x = 0; x < bottom[1]->width(); ++x) {
        img.at<unsigned char>(y, x) = label[y * bottom[1]->width() + x];
      }
    }

    std::vector<std::vector<cv::Point2i> > blobs;

    cv::Mat seg = FindBlobs(img, &blobs);

    // This is for debugging only:
    {
      cv::Mat output = cv::Mat::zeros(img.size(), CV_8UC3);
      for (size_t i = 0; i < blobs.size(); i++) {
        unsigned char r = 255 * (rand() / (1.0 + RAND_MAX));  // NOLINT
        unsigned char g = 255 * (rand() / (1.0 + RAND_MAX));  // NOLINT
        unsigned char b = 255 * (rand() / (1.0 + RAND_MAX));  // NOLINT

        for (size_t j = 0; j < blobs[i].size(); j++) {
          int x = blobs[i][j].x;
          int y = blobs[i][j].y;

          output.at<cv::Vec3b>(y, x)[0] = b;
          output.at<cv::Vec3b>(y, x)[1] = g;
          output.at<cv::Vec3b>(y, x)[2] = r;
        }
      }
      cv::imshow("labelled", output);
      cv::waitKey(100);
    }

    Dtype loss_out = 0;
    Dtype classerr_out = 0;
    Dtype rand_index_out = 0;

    std::vector<Dtype> conn_data_pos(
        2 * (bottom[0]->height() - 1) * (bottom[0]->width() - 1));
    std::vector<Dtype> conn_data_neg(
        2 * (bottom[0]->height() - 1) * (bottom[0]->width() - 1));
    std::vector<Dtype> dloss_pos(
        2 * (bottom[0]->height() - 1) * (bottom[0]->width() - 1));
    std::vector<Dtype> dloss_neg(
        2 * (bottom[0]->height() - 1) * (bottom[0]->width() - 1));

    // Construct positive and negative affinity graph
#pragma omp parallel for
    for (int i = 0; i < bottom[0]->height() - 1; ++i) {
      for (int j = 0; j < bottom[0]->width() - 1; ++j) {
        // Center
        Dtype p0 = prob_data[i * bottom[0]->width() + j];
        // Right
        Dtype p1 = prob_data[i * bottom[0]->width() + (j + 1)];
        // Bottom
        Dtype p2 = prob_data[(i + 1) * bottom[0]->width() + j];

        // Center
        Dtype g0 = label[i * bottom[0]->width() + j];
        // Right
        Dtype g1 = label[i * bottom[0]->width() + (j + 1)];
        // Bottom
        Dtype g2 = label[(i + 1) * bottom[0]->width() + j];

        // X positive
        conn_data_pos[i * (bottom[0]->width() - 1) + j] = std::min(
            1.0 - std::fabs(p0 - p1), 1.0 - std::fabs(g0 - g1));

        // X negative
        conn_data_neg[i * (bottom[0]->width() - 1) + j] = std::max(
            1.0 - std::fabs(p0 - p1), 1.0 - std::fabs(g0 - g1));

        // Y positive
        conn_data_pos[(bottom[0]->width() - 1) * (bottom[0]->height() - 1)
            + i * (bottom[0]->width() - 1) + j] = std::min(
            1.0 - std::fabs(p0 - p2), 1.0 - std::fabs(g0 - g2));

        // Y negative
        conn_data_neg[(bottom[0]->width() - 1) * (bottom[0]->height() - 1)
            + i * (bottom[0]->width() - 1) + j] = std::max(
            1.0 - std::fabs(p0 - p2), 1.0 - std::fabs(g0 - g2));
      }
    }

    auto minmax = std::minmax_element(conn_data_neg.begin(),conn_data_neg.end());

    std::cout << "Conndata neg min/max: " <<
        conn_data_neg[minmax.first - conn_data_neg.begin()] << " " <<
        conn_data_neg[minmax.second - conn_data_neg.begin()]  << std::endl;

    minmax = std::minmax_element(dloss_pos.begin(),dloss_pos.end());

    std::cout << "Conndata pos min/max: " <<
        conn_data_pos[minmax.first - conn_data_pos.begin()] << " " <<
        conn_data_pos[minmax.second - conn_data_pos.begin()]  << std::endl;


    std::cout << "Before MALIS 1" << std::endl;

    Malis(&conn_data_pos[0], conn_num_dims_, &conn_dims_[0], &nhood_data_[0],
          &nhood_dims_[0], reinterpret_cast<int*>(seg.ptr(0)),
          true, &dloss_pos[0],
          &loss_out, &classerr_out, &rand_index_out);

    std::cout << "Before MALIS 2" << std::endl;

    Malis(&conn_data_neg[0], conn_num_dims_, &conn_dims_[0], &nhood_data_[0],
          &nhood_dims_[0], reinterpret_cast<int*>(seg.ptr(0)),
          false, &dloss_neg[0],
          &loss_out, &classerr_out, &rand_index_out);


    minmax = std::minmax_element(dloss_neg.begin(),dloss_neg.end());

    std::cout << "DLoss_neg min/max: " <<
        dloss_neg[minmax.first - dloss_neg.begin()] << " " <<
        dloss_neg[minmax.second - dloss_neg.begin()]  << std::endl;

    minmax = std::minmax_element(dloss_pos.begin(),dloss_pos.end());

    std::cout << "DLoss_pos min/max: " <<
        dloss_pos[minmax.first - dloss_pos.begin()] << " " <<
        dloss_pos[minmax.second - dloss_pos.begin()]  << std::endl;

    std::cout << "Before PROB BACK" << std::endl;

    caffe_cpu_copy(prob_.count(), prob_data, bottom_diff);

    std::cout << "Before LOSS BACK" << std::endl;

    // Spread out the losses to pixels
    for (int i = 0; i < bottom[0]->height() - 1; ++i) {
      for (int j = 0; j < bottom[0]->width() - 1; ++j) {
        Dtype lxp = dloss_pos[i * (bottom[0]->width() - 1) + j];
        Dtype lxn = dloss_neg[i * (bottom[0]->width() - 1) + j];

        Dtype lyp = dloss_pos[(bottom[0]->width() - 1)
            * (bottom[0]->height() - 1) + i * (bottom[0]->width() - 1) + j];
        Dtype lyn = dloss_neg[(bottom[0]->width() - 1)
            * (bottom[0]->height() - 1) + i * (bottom[0]->width() - 1) + j];

        // Pick labels
        const int l0 = static_cast<int>
          (label[i * bottom[0]->width() + j]);
        const int l1 = static_cast<int>
          (label[i * bottom[0]->width() + (j + 1)]);
        const int l2 = static_cast<int>
          (label[(i + 1) * bottom[0]->width() + j]);

        // Center
        bottom_diff[l0 * inner_num_ + i * bottom[0]->width() + j] += 0.5
            * (lxp + lxn + lyp + lyn);

        // Right
        bottom_diff[l1 * inner_num_ + i * bottom[0]->width() + (j + 1)] += 0.5
            * (lxp + lxn);

        // Bottom
        bottom_diff[l2 * inner_num_ + (i + 1) * bottom[0]->width() + j] += 0.5
            * (lyp + lyn);
      }
    }

    Dtype* prob_rd = prob_.mutable_cpu_data();

    cv::Mat wrapped_1(bottom[0]->height(), bottom[0]->width(),
                      cv::DataType<Dtype>::type,
                    prob_rd, sizeof(Dtype) * bottom[0]->width());
    cv::imshow("prob", wrapped_1);
    cv::waitKey(100);

    cv::Mat wrapped_2(bottom[0]->height(), bottom[0]->width(),
                      cv::DataType<Dtype>::type,
                    bottom_diff, sizeof(Dtype) * bottom[0]->width());
    cv::imshow("diff", wrapped_2);
    cv::waitKey(100);

    std::cout << "After LOSS BACK" << std::endl;
  }
}

INSTANTIATE_CLASS(MalisLossLayer);
REGISTER_LAYER_CLASS(MalisLoss);

}  // namespace caffe
