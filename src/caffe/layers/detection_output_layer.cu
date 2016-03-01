#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"

#include "caffe/layers/detection_output_layer.hpp"

namespace caffe {

template <typename Dtype>
void DetectionOutputLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* loc_data = bottom[0]->gpu_data();
  const Dtype* conf_data = bottom[1]->gpu_data();
  const Dtype* prior_data = bottom[2]->gpu_data();
  const int num = bottom[0]->num();

  // Decode predictions.
  Blob<Dtype> bbox_preds;
  bbox_preds.ReshapeLike(*(bottom[0]));
  Dtype* bbox_data = bbox_preds.mutable_gpu_data();
  const int loc_count = bbox_preds.count();
  DecodeBBoxesGPU<Dtype>(loc_count, loc_data, prior_data, code_type_,
      num_priors_, share_location_, num_loc_classes_, background_label_id_,
      bbox_data);

  // Compute overlap between detection results.
  Blob<bool> overlapped(num, num_loc_classes_, num_priors_, num_priors_);
  const int total_bboxes = overlapped.count();
  bool* overlapped_data = overlapped.mutable_gpu_data();
  ComputeOverlappedGPU<Dtype>(total_bboxes, bbox_data, num_priors_,
      num_loc_classes_, nms_threshold_, overlapped_data);
  const bool* overlapped_results = overlapped.cpu_data();

  const Dtype* bbox_cpu_data = bbox_preds.cpu_data();
  vector<LabelBBox> all_decode_bboxes;
  GetLocPredictions(bbox_cpu_data, num, num_priors_, num_loc_classes_,
      share_location_, &all_decode_bboxes);

  // Retrieve all confidences.
  const Dtype* conf_cpu_data = bottom[1]->cpu_data();
  vector<map<int, vector<float> > > all_conf_scores;
  GetConfidenceScores(conf_cpu_data, num, num_priors_, num_classes_,
                      &all_conf_scores);

  int num_kept = 0;
  vector<map<int, vector<int> > > all_indices;
  for (int i = 0; i < num; ++i) {
    // Retrieve kept bbox after nms for each class.
    LabelBBox& decode_bboxes = all_decode_bboxes[i];
    map<int, vector<float> >& conf_scores = all_conf_scores[i];
    map<int, vector<int> > indices;
    int num_det = 0;
    for (int c = 0; c < num_classes_; ++c) {
      if (c == background_label_id_) {
        // Ignore background class.
        if (!share_location_) {
          overlapped_results += num_priors_ * num_priors_;
        }
        continue;
      }
      if (conf_scores.find(c) == conf_scores.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find confidence predictions for label " << c;
      }
      int label = share_location_ ? -1 : c;
      if (decode_bboxes.find(label) == decode_bboxes.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find location predictions for label " << label;
        continue;
      }
      ApplyNMS(decode_bboxes[label], conf_scores[c], overlapped_results, top_k_,
               &(indices[c]));
      num_det += indices[c].size();
      if (!share_location_) {
        overlapped_results += num_priors_ * num_priors_;
      }
    }
    if (share_location_) {
      overlapped_results += num_priors_ * num_priors_;
    }
    if (keep_top_k_ > -1 && num_det > keep_top_k_) {
      vector<pair<float, pair<int, int> > > score_index_pairs;
      for (map<int, vector<int> >::iterator it = indices.begin();
           it != indices.end(); ++it) {
        int label = it->first;
        const vector<int>& label_indices = it->second;
        if (conf_scores.find(label) == conf_scores.end()) {
          // Something bad happened for current label.
          LOG(FATAL) << "Could not find location predictions for " << label;
          continue;
        }
        for (int j = 0; j < label_indices.size(); ++j) {
          int idx = label_indices[j];
          CHECK_LT(idx, conf_scores[label].size());
          score_index_pairs.push_back(std::make_pair(
                  conf_scores[label][idx], std::make_pair(label, idx)));
        }
      }
      // Keep top k results per image.
      std::sort(score_index_pairs.begin(), score_index_pairs.end(),
                SortScorePairDescend<pair<int, int> >);
      score_index_pairs.resize(keep_top_k_);
      // Store the new indices.
      map<int, vector<int> > new_indices;
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
  top[0]->Reshape(top_shape);
  Dtype* top_data = top[0]->mutable_cpu_data();

  int count = 0;
  boost::filesystem::path output_directory(output_directory_);
  for (int i = 0; i < num; ++i) {
    map<int, vector<float> >& conf_scores = all_conf_scores[i];
    const LabelBBox& decode_bboxes = all_decode_bboxes[i];
    for (map<int, vector<int> >::iterator it = all_indices[i].begin();
         it != all_indices[i].end(); ++it) {
      int label = it->first;
      if (conf_scores.find(label) == conf_scores.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find confidence predictions for " << label;
        continue;
      }
      int loc_label = share_location_ ? -1 : label;
      if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find location predictions for " << loc_label;
        continue;
      }
      const vector<NormalizedBBox>& bboxes =
          decode_bboxes.find(loc_label)->second;
      vector<int>& indices = it->second;
      std::ofstream outfile;
      if (need_save_) {
        CHECK(label_to_name_.find(label) != label_to_name_.end())
            << "Cannot find label: " << label << " in the label map.";
        boost::filesystem::path file(
            output_name_prefix_ + label_to_name_[label] + ".txt");
        boost::filesystem::path out_file = output_directory / file;
        outfile.open(out_file.string().c_str(),
                     std::ofstream::out | std::ofstream::app);
        CHECK_LT(name_count_, names_.size());
      }
      for (int j = 0; j < indices.size(); ++j) {
        int idx = indices[j];
        top_data[count * 7] = i;
        top_data[count * 7 + 1] = label;
        top_data[count * 7 + 2] = conf_scores[label][idx];
        NormalizedBBox clip_bbox;
        ClipBBox(bboxes[idx], &clip_bbox);
        top_data[count * 7 + 3] = clip_bbox.xmin();
        top_data[count * 7 + 4] = clip_bbox.ymin();
        top_data[count * 7 + 5] = clip_bbox.xmax();
        top_data[count * 7 + 6] = clip_bbox.ymax();
        if (need_save_) {
          outfile << names_[name_count_];
          outfile << " " << conf_scores[label][idx];
          NormalizedBBox scale_bbox;
          ScaleBBox(clip_bbox, sizes_[name_count_].first,
                    sizes_[name_count_].second, &scale_bbox);
          outfile << " " << static_cast<int>(scale_bbox.xmin());
          outfile << " " << static_cast<int>(scale_bbox.ymin());
          outfile << " " << static_cast<int>(scale_bbox.xmax());
          outfile << " " << static_cast<int>(scale_bbox.ymax());
          outfile << std::endl;
          outfile.flush();
        }
        ++count;
      }
      if (need_save_) {
        outfile.close();
      }
    }
    if (need_save_) {
      ++name_count_;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DetectionOutputLayer);

}  // namespace caffe
