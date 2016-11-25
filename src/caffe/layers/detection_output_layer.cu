#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "caffe/layers/detection_output_layer.hpp"

namespace caffe {

template <typename Dtype>
void DetectionOutputLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* loc_data = bottom[0]->gpu_data();
  const Dtype* prior_data = bottom[2]->gpu_data();
  const int num = bottom[0]->num();

  // Decode predictions.
  Dtype* bbox_data = bbox_preds_.mutable_gpu_data();
  const int loc_count = bbox_preds_.count();
  const bool clip_bbox = false;
  DecodeBBoxesGPU<Dtype>(loc_count, loc_data, prior_data, code_type_,
      variance_encoded_in_target_, num_priors_, share_location_,
      num_loc_classes_, background_label_id_, clip_bbox, bbox_data);
  // Retrieve all decoded location predictions.
  const Dtype* bbox_cpu_data;
  if (!share_location_) {
    Dtype* bbox_permute_data = bbox_permute_.mutable_gpu_data();
    PermuteDataGPU<Dtype>(loc_count, bbox_data, num_loc_classes_, num_priors_,
        4, bbox_permute_data);
    bbox_cpu_data = bbox_permute_.cpu_data();
  } else {
    bbox_cpu_data = bbox_preds_.cpu_data();
  }

  // Retrieve all confidences.
  Dtype* conf_permute_data = conf_permute_.mutable_gpu_data();
  PermuteDataGPU<Dtype>(bottom[1]->count(), bottom[1]->gpu_data(),
      num_classes_, num_priors_, 1, conf_permute_data);
  const Dtype* conf_cpu_data = conf_permute_.cpu_data();

  int num_kept = 0;
  vector<map<int, vector<int> > > all_indices;
  for (int i = 0; i < num; ++i) {
    map<int, vector<int> > indices;
    int num_det = 0;
    const int conf_idx = i * num_classes_ * num_priors_;
    int bbox_idx;
    if (share_location_) {
      bbox_idx = i * num_priors_ * 4;
    } else {
      bbox_idx = conf_idx * 4;
    }
    for (int c = 0; c < num_classes_; ++c) {
      if (c == background_label_id_) {
        // Ignore background class.
        continue;
      }
      const Dtype* cur_conf_data = conf_cpu_data + conf_idx + c * num_priors_;
      const Dtype* cur_bbox_data = bbox_cpu_data + bbox_idx;
      if (!share_location_) {
        cur_bbox_data += c * num_priors_ * 4;
      }
      ApplyNMSFast(cur_bbox_data, cur_conf_data, num_priors_,
          confidence_threshold_, nms_threshold_, eta_, top_k_, &(indices[c]));
      num_det += indices[c].size();
    }
    if (keep_top_k_ > -1 && num_det > keep_top_k_) {
      vector<pair<float, pair<int, int> > > score_index_pairs;
      for (map<int, vector<int> >::iterator it = indices.begin();
           it != indices.end(); ++it) {
        int label = it->first;
        const vector<int>& label_indices = it->second;
        for (int j = 0; j < label_indices.size(); ++j) {
          int idx = label_indices[j];
          float score = conf_cpu_data[conf_idx + label * num_priors_ + idx];
          score_index_pairs.push_back(std::make_pair(
                  score, std::make_pair(label, idx)));
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
  boost::filesystem::path output_directory(output_directory_);
  for (int i = 0; i < num; ++i) {
    const int conf_idx = i * num_classes_ * num_priors_;
    int bbox_idx;
    if (share_location_) {
      bbox_idx = i * num_priors_ * 4;
    } else {
      bbox_idx = conf_idx * 4;
    }
    for (map<int, vector<int> >::iterator it = all_indices[i].begin();
         it != all_indices[i].end(); ++it) {
      int label = it->first;
      vector<int>& indices = it->second;
      if (need_save_) {
        CHECK(label_to_name_.find(label) != label_to_name_.end())
          << "Cannot find label: " << label << " in the label map.";
        CHECK_LT(name_count_, names_.size());
      }
      const Dtype* cur_conf_data =
        conf_cpu_data + conf_idx + label * num_priors_;
      const Dtype* cur_bbox_data = bbox_cpu_data + bbox_idx;
      if (!share_location_) {
        cur_bbox_data += label * num_priors_ * 4;
      }
      for (int j = 0; j < indices.size(); ++j) {
        int idx = indices[j];
        top_data[count * 7] = i;
        top_data[count * 7 + 1] = label;
        top_data[count * 7 + 2] = cur_conf_data[idx];
        for (int k = 0; k < 4; ++k) {
          top_data[count * 7 + 3 + k] = cur_bbox_data[idx * 4 + k];
        }
        if (need_save_) {
          // Generate output bbox.
          NormalizedBBox bbox;
          bbox.set_xmin(top_data[count * 7 + 3]);
          bbox.set_ymin(top_data[count * 7 + 4]);
          bbox.set_xmax(top_data[count * 7 + 5]);
          bbox.set_ymax(top_data[count * 7 + 6]);
          NormalizedBBox out_bbox;
          OutputBBox(bbox, sizes_[name_count_], has_resize_, resize_param_,
                     &out_bbox);
          float score = top_data[count * 7 + 2];
          float xmin = out_bbox.xmin();
          float ymin = out_bbox.ymin();
          float xmax = out_bbox.xmax();
          float ymax = out_bbox.ymax();
          ptree pt_xmin, pt_ymin, pt_width, pt_height;
          pt_xmin.put<float>("", round(xmin * 100) / 100.);
          pt_ymin.put<float>("", round(ymin * 100) / 100.);
          pt_width.put<float>("", round((xmax - xmin) * 100) / 100.);
          pt_height.put<float>("", round((ymax - ymin) * 100) / 100.);

          ptree cur_bbox;
          cur_bbox.push_back(std::make_pair("", pt_xmin));
          cur_bbox.push_back(std::make_pair("", pt_ymin));
          cur_bbox.push_back(std::make_pair("", pt_width));
          cur_bbox.push_back(std::make_pair("", pt_height));

          ptree cur_det;
          cur_det.put("image_id", names_[name_count_]);
          if (output_format_ == "ILSVRC") {
            cur_det.put<int>("category_id", label);
          } else {
            cur_det.put("category_id", label_to_name_[label].c_str());
          }
          cur_det.add_child("bbox", cur_bbox);
          cur_det.put<float>("score", score);

          detections_.push_back(std::make_pair("", cur_det));
        }
        ++count;
      }
    }
    if (need_save_) {
      ++name_count_;
      if (name_count_ % num_test_image_ == 0) {
        if (output_format_ == "VOC") {
          map<string, std::ofstream*> outfiles;
          for (int c = 0; c < num_classes_; ++c) {
            if (c == background_label_id_) {
              continue;
            }
            string label_name = label_to_name_[c];
            boost::filesystem::path file(
                output_name_prefix_ + label_name + ".txt");
            boost::filesystem::path out_file = output_directory / file;
            outfiles[label_name] = new std::ofstream(out_file.string().c_str(),
                std::ofstream::out);
          }
          BOOST_FOREACH(ptree::value_type &det, detections_.get_child("")) {
            ptree pt = det.second;
            string label_name = pt.get<string>("category_id");
            if (outfiles.find(label_name) == outfiles.end()) {
              std::cout << "Cannot find " << label_name << std::endl;
              continue;
            }
            string image_name = pt.get<string>("image_id");
            float score = pt.get<float>("score");
            vector<int> bbox;
            BOOST_FOREACH(ptree::value_type &elem, pt.get_child("bbox")) {
              bbox.push_back(static_cast<int>(elem.second.get_value<float>()));
            }
            *(outfiles[label_name]) << image_name;
            *(outfiles[label_name]) << " " << score;
            *(outfiles[label_name]) << " " << bbox[0] << " " << bbox[1];
            *(outfiles[label_name]) << " " << bbox[0] + bbox[2];
            *(outfiles[label_name]) << " " << bbox[1] + bbox[3];
            *(outfiles[label_name]) << std::endl;
          }
          for (int c = 0; c < num_classes_; ++c) {
            if (c == background_label_id_) {
              continue;
            }
            string label_name = label_to_name_[c];
            outfiles[label_name]->flush();
            outfiles[label_name]->close();
            delete outfiles[label_name];
          }
        } else if (output_format_ == "COCO") {
          boost::filesystem::path output_directory(output_directory_);
          boost::filesystem::path file(output_name_prefix_ + ".json");
          boost::filesystem::path out_file = output_directory / file;
          std::ofstream outfile;
          outfile.open(out_file.string().c_str(), std::ofstream::out);

          boost::regex exp("\"(null|true|false|-?[0-9]+(\\.[0-9]+)?)\"");
          ptree output;
          output.add_child("detections", detections_);
          std::stringstream ss;
          write_json(ss, output);
          std::string rv = boost::regex_replace(ss.str(), exp, "$1");
          outfile << rv.substr(rv.find("["), rv.rfind("]") - rv.find("["))
              << std::endl << "]" << std::endl;
        } else if (output_format_ == "ILSVRC") {
          boost::filesystem::path output_directory(output_directory_);
          boost::filesystem::path file(output_name_prefix_ + ".txt");
          boost::filesystem::path out_file = output_directory / file;
          std::ofstream outfile;
          outfile.open(out_file.string().c_str(), std::ofstream::out);

          BOOST_FOREACH(ptree::value_type &det, detections_.get_child("")) {
            ptree pt = det.second;
            int label = pt.get<int>("category_id");
            string image_name = pt.get<string>("image_id");
            float score = pt.get<float>("score");
            vector<int> bbox;
            BOOST_FOREACH(ptree::value_type &elem, pt.get_child("bbox")) {
              bbox.push_back(static_cast<int>(elem.second.get_value<float>()));
            }
            outfile << image_name << " " << label << " " << score;
            outfile << " " << bbox[0] << " " << bbox[1];
            outfile << " " << bbox[0] + bbox[2];
            outfile << " " << bbox[1] + bbox[3];
            outfile << std::endl;
          }
        }
        name_count_ = 0;
        detections_.clear();
      }
    }
  }
  if (visualize_) {
#ifdef USE_OPENCV
    vector<cv::Mat> cv_imgs;
    this->data_transformer_->TransformInv(bottom[3], &cv_imgs);
    vector<cv::Scalar> colors = GetColors(label_to_display_name_.size());
    VisualizeBBox(cv_imgs, top[0], visualize_threshold_, colors,
        label_to_display_name_, save_file_);
#endif  // USE_OPENCV
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DetectionOutputLayer);

}  // namespace caffe
