#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "caffe/layers/fast_rcnn_layers.hpp"

namespace caffe {

template <typename Dtype>
void FasterRcnnDetectionOutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const FasterRcnnDetectionOutputParameter& detection_output_param =
      this->layer_param_.faster_rcnn_detection_output_param();
  CHECK(detection_output_param.has_num_classes()) << "Must specify num_classes";
  num_classes_ = detection_output_param.num_classes();
  share_location_ = false;
  num_loc_classes_ = num_classes_;
  background_label_id_ = detection_output_param.background_label_id();
  code_type_ = PriorBoxParameter_CodeType_CENTER_SIZE_FASTER_RCNN;
  variance_encoded_in_target_ = false;
  keep_top_k_ = -1;
  confidence_threshold_ = detection_output_param.has_confidence_threshold() ?
      detection_output_param.confidence_threshold() : -FLT_MAX;
  // Parameters used in nms.
  nms_threshold_ = detection_output_param.nms_param().nms_threshold();
  CHECK_GE(nms_threshold_, 0.) << "nms_threshold must be non negative.";
  top_k_ = -1;
  eta_ = 2.0;

  bbox_preds_.ReshapeLike(*(bottom[0]));
  bbox_permute_.ReshapeLike(*(bottom[0]));
  conf_permute_.ReshapeLike(*(bottom[1]));
}

template <typename Dtype>
void FasterRcnnDetectionOutputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->num(), bottom[2]->num());
  CHECK_EQ(bottom[2]->channels(), 5);
  if (bbox_preds_.num() != bottom[0]->num() ||
      bbox_preds_.count(1) != bottom[0]->count(1)) {
    bbox_preds_.ReshapeLike(*(bottom[0]));
  }
  if (!share_location_ && (bbox_permute_.num() != bottom[0]->num() ||
      bbox_permute_.count(1) != bottom[0]->count(1))) {
    bbox_permute_.ReshapeLike(*(bottom[0]));
  }
  if (conf_permute_.num() != bottom[1]->num() ||
      conf_permute_.count(1) != bottom[1]->count(1)) {
    conf_permute_.ReshapeLike(*(bottom[1]));
  }
  num_priors_ = bottom[0]->num();
  CHECK_EQ(num_loc_classes_ * 4, bottom[0]->channels());
  CHECK_EQ(num_classes_, bottom[1]->channels());
  // num() and channels() are 1.
  vector<int> top_shape(2, 1);
  // Since the number of bboxes to be kept is unknown before nms, we manually
  // set it to (fake) 1.
  top_shape.push_back(1);
  // Each row is a 7 dimension vector, which stores
  // [image_id, label, confidence, xmin, ymin, xmax, ymax]
  top_shape.push_back(7);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void FasterRcnnDetectionOutputLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* loc_data = bottom[0]->cpu_data();
  const Dtype* conf_data = bottom[1]->cpu_data();
  const Dtype* prior_data = bottom[2]->cpu_data();
  const Dtype* iminfo_data = bottom[3]->cpu_data();
  const int num = 1;

  // Retrieve all location predictions.
  vector<LabelBBox> all_loc_preds;
  GetLocPredictions(loc_data, num, num_priors_, num_loc_classes_,
                    share_location_, &all_loc_preds);

  // Retrieve all confidences.
  vector<map<int, vector<float> > > all_conf_scores;
  GetConfidenceScores(conf_data, num, num_priors_, num_classes_,
                      &all_conf_scores);

  vector<NormalizedBBox> prior_bboxes;
  GetRoiBBoxes(prior_data, num_priors_, &prior_bboxes);

  // Decode all loc predictions to bboxes.
  vector<LabelBBox> all_decode_bboxes;
  vector<vector<float> > prior_variances;
  prior_variances.resize(num_priors_);
  const bool clip_bbox = true;
  float clip_w = iminfo_data[1] - 1.0f;
  float clip_h = iminfo_data[0] - 1.0f;
  DecodeBBoxesAll(all_loc_preds, prior_bboxes, prior_variances, num,
                  share_location_, num_loc_classes_, background_label_id_,
                  code_type_, variance_encoded_in_target_, clip_bbox, clip_w, clip_h,
                  &all_decode_bboxes);

  int num_kept = 0;
  vector<map<int, vector<int> > > all_indices;
  for (int i = 0; i < num; ++i) {
    const LabelBBox& decode_bboxes = all_decode_bboxes[i];
    const map<int, vector<float> >& conf_scores = all_conf_scores[i];
    map<int, vector<int> > indices;
    int num_det = 0;
    for (int c = 0; c < num_classes_; ++c) {
      if (c == background_label_id_) {
        // Ignore background class.
        continue;
      }
      if (conf_scores.find(c) == conf_scores.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find confidence predictions for label " << c;
      }
      const vector<float>& scores = conf_scores.find(c)->second;
      int label = share_location_ ? -1 : c;
      if (decode_bboxes.find(label) == decode_bboxes.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find location predictions for label " << label;
        continue;
      }
      const vector<NormalizedBBox>& bboxes = decode_bboxes.find(label)->second;
      ApplyNMSFast(bboxes, scores, confidence_threshold_, nms_threshold_, eta_,
          top_k_, &(indices[c]));
      num_det += indices[c].size();
    }

    all_indices.push_back(indices);
    num_kept += num_det;
  }

  vector<int> top_shape(2, 1);
  top_shape.push_back(num_kept);
  top_shape.push_back(7);
  Dtype* top_data;
  if (num_kept == 0) {
    LOG(INFO) << "Couldn't find any detections";
    top_shape[2] = num;
    top[0]->Reshape(top_shape);
    top_data = top[0]->mutable_cpu_data();
    caffe_set<Dtype>(top[0]->count(), -1, top_data);
    // Generate fake results per image.
    for (int i = 0; i < num; ++i) {
      top_data[0] = i;
      top_data += 7;
    }
  } else {
    top[0]->Reshape(top_shape);
    top_data = top[0]->mutable_cpu_data();
  }

  int count = 0;
  float scale = iminfo_data[2];
  if (fabs(scale) < 0.00001)
    scale = 1.0f;

  for (int i = 0; i < num; ++i) {
    const map<int, vector<float> >& conf_scores = all_conf_scores[i];
    const LabelBBox& decode_bboxes = all_decode_bboxes[i];
    for (map<int, vector<int> >::iterator it = all_indices[i].begin();
         it != all_indices[i].end(); ++it) {
      int label = it->first;
      if (conf_scores.find(label) == conf_scores.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find confidence predictions for " << label;
        continue;
      }
      const vector<float>& scores = conf_scores.find(label)->second;
      int loc_label = share_location_ ? -1 : label;
      if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find location predictions for " << loc_label;
        continue;
      }
      const vector<NormalizedBBox>& bboxes =
          decode_bboxes.find(loc_label)->second;
      vector<int>& indices = it->second;
      for (int j = 0; j < indices.size(); ++j) {
        int idx = indices[j];
        top_data[count * 7] = i;
        top_data[count * 7 + 1] = label;
        top_data[count * 7 + 2] = scores[idx];
        const NormalizedBBox& bbox = bboxes[idx];
        top_data[count * 7 + 3] = bbox.xmin() / scale;
        top_data[count * 7 + 4] = bbox.ymin() / scale;
        top_data[count * 7 + 5] = bbox.xmax() / scale;
        top_data[count * 7 + 6] = bbox.ymax() / scale;
        ++count;
      }
    }
  }

}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(FasterRcnnDetectionOutputLayer, Forward);
#endif

INSTANTIATE_CLASS(FasterRcnnDetectionOutputLayer);
REGISTER_LAYER_CLASS(FasterRcnnDetectionOutput);

}  // namespace caffe
