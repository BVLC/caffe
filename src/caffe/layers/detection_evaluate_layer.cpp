#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "caffe/layers/detection_evaluate_layer.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

template <typename Dtype>
void DetectionEvaluateLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const DetectionEvaluateParameter& detection_evaluate_param =
      this->layer_param_.detection_evaluate_param();
  background_label_id_ = detection_evaluate_param.background_label_id();
  overlap_threshold_ = detection_evaluate_param.overlap_threshold();
  CHECK_GT(overlap_threshold_, 0.) << "overlap_threshold must be non negative.";
}

template <typename Dtype>
void DetectionEvaluateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), 1);
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[0]->width(), 7);
  CHECK_EQ(bottom[1]->num(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->width(), 7);

  set<int> labels;
  for (int i = 0; i < bottom[1]->height(); ++i) {
    int label = bottom[1]->cpu_data()[i * 7 + 1];
    labels.insert(label);
  }
  // num() and channels() are 1.
  vector<int> top_shape(2, 1);
  top_shape.push_back(labels.size() + bottom[0]->height());
  // Each row is a 5 dimension vector, which stores
  // [image_id, label, confidence, true_pos, false_pos]
  top_shape.push_back(5);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void DetectionEvaluateLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* det_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();

  // Retrieve all detection results.
  map<int, LabelBBox> all_detections;
  GetDetectionResults(det_data, bottom[0]->height(), background_label_id_,
                      &all_detections);

  // Retrieve all ground truth.
  map<int, LabelBBox> all_gt_bboxes;
  GetGroundTruth(gt_data, bottom[1]->height(), background_label_id_,
                 &all_gt_bboxes);

  Dtype* top_data = top[0]->mutable_cpu_data();
  int num_det = 0;
  // Insert number of ground truth for each labe.
  map<int, int> num_pos;
  for (map<int, LabelBBox>::iterator it = all_gt_bboxes.begin();
       it != all_gt_bboxes.end(); ++it) {
    for (LabelBBox::iterator iit = it->second.begin(); iit != it->second.end();
         ++iit) {
      if (num_pos.find(iit->first) == num_pos.end()) {
        num_pos[iit->first] = iit->second.size();
      } else {
        num_pos[iit->first] += iit->second.size();
      }
    }
  }
  for (map<int, int>::iterator it = num_pos.begin(); it != num_pos.end();
       ++it) {
    top_data[num_det * 5] = -1;
    top_data[num_det * 5 + 1] = it->first;
    top_data[num_det * 5 + 2] = it->second;
    top_data[num_det * 5 + 3] = -1;
    top_data[num_det * 5 + 4] = -1;
    ++num_det;
  }

  // Insert detection evaluate status.
  for (map<int, LabelBBox>::iterator it = all_detections.begin();
       it != all_detections.end(); ++it) {
    int image_id = it->first;
    LabelBBox& detections = it->second;
    if (all_gt_bboxes.find(image_id) == all_gt_bboxes.end()) {
      // No ground truth for current image. All detections become false_pos.
      for (LabelBBox::iterator iit = detections.begin();
           iit != detections.end(); ++iit) {
        int label = iit->first;
        const vector<NormalizedBBox>& bboxes = iit->second;
        for (int i = 0; i < bboxes.size(); ++i) {
          top_data[num_det * 5] = image_id;
          top_data[num_det * 5 + 1] = label;
          top_data[num_det * 5 + 2] = bboxes[i].score();
          top_data[num_det * 5 + 3] = 0;
          top_data[num_det * 5 + 4] = 1;
          ++num_det;
        }
      }
    } else {
      const LabelBBox& label_bboxes = all_gt_bboxes.find(image_id)->second;
      for (LabelBBox::iterator iit = detections.begin();
           iit != detections.end(); ++iit) {
        int label = iit->first;
        vector<NormalizedBBox>& bboxes = iit->second;
        if (label_bboxes.find(label) == label_bboxes.end()) {
          // No ground truth for current label. All detections become false_pos.
          for (int i = 0; i < bboxes.size(); ++i) {
            top_data[num_det * 5] = image_id;
            top_data[num_det * 5 + 1] = label;
            top_data[num_det * 5 + 2] = bboxes[i].score();
            top_data[num_det * 5 + 3] = 0;
            top_data[num_det * 5 + 4] = 1;
            ++num_det;
          }
        } else {
          const vector<NormalizedBBox>& gt_bboxes =
              label_bboxes.find(label)->second;
          vector<bool> visited(gt_bboxes.size(), false);
          // Sort detections in descend order based on scores.
          std::sort(bboxes.begin(), bboxes.end(), SortBBoxDescend);
          for (int i = 0; i < bboxes.size(); ++i) {
            top_data[num_det * 5] = image_id;
            top_data[num_det * 5 + 1] = label;
            top_data[num_det * 5 + 2] = bboxes[i].score();
            // Compare with each ground truth bbox.
            float overlap_max = -1;
            int jmax = -1;
            for (int j = 0; j < gt_bboxes.size(); ++j) {
              float overlap = JaccardOverlap(bboxes[i], gt_bboxes[j]);
              if (overlap > overlap_max) {
                overlap_max = overlap;
                jmax = j;
              }
            }
            if (overlap_max >= overlap_threshold_) {
              if (!visited[jmax]) {
                // true positive.
                top_data[num_det * 5 + 3] = 1;
                top_data[num_det * 5 + 4] = 0;
                visited[jmax] = true;
              } else {
                // false positive (multiple detection).
                top_data[num_det * 5 + 3] = 0;
                top_data[num_det * 5 + 4] = 1;
              }
            } else {
              // false positive.
              top_data[num_det * 5 + 3] = 0;
              top_data[num_det * 5 + 4] = 1;
            }
            ++num_det;
          }
        }
      }
    }
  }
}

INSTANTIATE_CLASS(DetectionEvaluateLayer);
REGISTER_LAYER_CLASS(DetectionEvaluate);

}  // namespace caffe
