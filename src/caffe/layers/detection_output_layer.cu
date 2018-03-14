#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "caffe/layers/detection_output_layer.hpp"

namespace caffe {
	
template <typename Dtype>
void DetectionOutputLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  Forward_const_gpu(bottom, top);
}

template <typename Dtype>
void DetectionOutputLayer<Dtype>::Forward_const_gpu(
    const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) const {
  const Dtype *loc_data = bottom[0]->gpu_data();
  const Dtype *prior_data = bottom[2]->gpu_data();
  const int num = bottom[0]->num();

  int num_priors = bottom[2]->height() / 4;
  CHECK_EQ(num_priors * num_loc_classes_ * 4, bottom[0]->channels())
      << "Number of priors must match number of location predictions.";
  CHECK_EQ(num_priors * num_classes_, bottom[1]->channels())
      << "Number of priors must match number of confidence predictions.";

  // Decode predictions.
  Blob<Dtype> bbox_preds;
  bbox_preds.ReshapeLike(*(bottom[0]));
  Dtype *bbox_data = bbox_preds.mutable_gpu_data();
  const int loc_count = bbox_preds.count();
  const bool clip_bbox = false;
  DecodeBBoxesGPU<Dtype>(loc_count, loc_data, prior_data, code_type_,
                         variance_encoded_in_target_, num_priors,
                         share_location_, num_loc_classes_,
                         background_label_id_, clip_bbox, bbox_data);
  // Retrieve all decoded location predictions.
  const Dtype *bbox_cpu_data;
  if (!share_location_) {
    Blob<Dtype> bbox_permute;
    bbox_permute.ReshapeLike(*(bottom[0]));
    Dtype *bbox_permute_data = bbox_permute.mutable_gpu_data();
    PermuteDataGPU<Dtype>(loc_count, bbox_data, num_loc_classes_, num_priors,
                          4, bbox_permute_data);
    bbox_cpu_data = bbox_permute.cpu_data();
  } else {
    bbox_cpu_data = bbox_preds.cpu_data();
  }

  // Retrieve all confidences.
  Blob<Dtype> conf_permute;
  conf_permute.ReshapeLike(*(bottom[1]));
  Dtype *conf_permutedata = conf_permute.mutable_gpu_data();
  PermuteDataGPU<Dtype>(bottom[1]->count(), bottom[1]->gpu_data(), num_classes_,
                        num_priors, 1, conf_permutedata);
  const Dtype *conf_cpu_data = conf_permute.cpu_data();

  int num_kept = 0;
  vector<map<int, vector<int>>> all_indices;
  for (int i = 0; i < num; ++i) {
    map<int, vector<int>> indices;
    int num_det = 0;
    const int conf_idx = i * num_classes_ * num_priors;
    int bbox_idx;
    if (share_location_) {
      bbox_idx = i * num_priors * 4;
    } else {
      bbox_idx = conf_idx * 4;
    }
    for (int c = 0; c < num_classes_; ++c) {
      if (c == background_label_id_) {
        // Ignore background class.
        continue;
      }
      const Dtype *cur_conf_data = conf_cpu_data + conf_idx + c * num_priors;
      const Dtype *cur_bbox_data = bbox_cpu_data + bbox_idx;
      if (!share_location_) {
        cur_bbox_data += c * num_priors * 4;
      }
      ApplyNMSFast(cur_bbox_data, cur_conf_data, num_priors,
                   confidence_threshold_, nms_threshold_, eta_, top_k_,
                   &(indices[c]));
      num_det += indices[c].size();
    }
    if (keep_top_k_ > -1 && num_det > keep_top_k_) {
      vector<pair<float, pair<int, int>>> score_index_pairs;
      for (map<int, vector<int>>::iterator it = indices.begin();
           it != indices.end(); ++it) {
        int label = it->first;
        const vector<int> &label_indices = it->second;
        for (int j = 0; j < label_indices.size(); ++j) {
          int idx = label_indices[j];
          float score = conf_cpu_data[conf_idx + label * num_priors + idx];
          score_index_pairs.push_back(
              std::make_pair(score, std::make_pair(label, idx)));
        }
      }
      // Keep top k results per image.
      std::partial_sort(score_index_pairs.begin(),score_index_pairs.begin()+keep_top_k_, score_index_pairs.end(),
                SortScorePairDescend<pair<int, int>>);
      score_index_pairs.resize(keep_top_k_);
      // Store the new indices.
      map<int, vector<int>> new_indices;
      for (int j = 0; j < score_index_pairs.size(); ++j) {
        int label = score_index_pairs[j].second.first;
        int idx = score_index_pairs[j].second.second;
        new_indices[label].push_back(idx);
      }
      all_indices.push_back(new_indices);
      num_kept += keep_top_k_;
    } else {
      all_indices.push_back(indices);
      num_kept += num_det;
    }
  }

  vector<int> top_shape(2, 1);
  top_shape.push_back(num_kept);
  top_shape.push_back(7);
  Dtype *top_data;
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
  for (int i = 0; i < num; ++i) {
    const int conf_idx = i * num_classes_ * num_priors;
    int bbox_idx;
    if (share_location_) {
      bbox_idx = i * num_priors * 4;
    } else {
      bbox_idx = conf_idx * 4;
    }
    for (map<int, vector<int>>::iterator it = all_indices[i].begin();
         it != all_indices[i].end(); ++it) {
      int label = it->first;
      vector<int> &indices = it->second;
      const Dtype *cur_conf_data =
          conf_cpu_data + conf_idx + label * num_priors;
      const Dtype *cur_bbox_data = bbox_cpu_data + bbox_idx;
      if (!share_location_) {
        cur_bbox_data += label * num_priors * 4;
      }
      for (int j = 0; j < indices.size(); ++j) {
        int idx = indices[j];
        top_data[count * 7] = i;
        top_data[count * 7 + 1] = label;
        top_data[count * 7 + 2] = cur_conf_data[idx];
        for (int k = 0; k < 4; ++k) {
          top_data[count * 7 + 3 + k] = cur_bbox_data[idx * 4 + k];
        }
        ++count;
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_CONST(DetectionOutputLayer);

} // namespace caffe
