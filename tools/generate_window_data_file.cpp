// Copyright 2014 BVLC and contributors.

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "caffe/caffe.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc != 5) {
    LOG(ERROR) << "Makes a window file that can be used by the caffe"
        " WindowDataLayer for finetuning.\n"
        "  Usage: generate_window_data_file"
        "  bounding_boxes_ground_truth_file  output_file  spatial_bins"
        "  stride_size_ratio\n"
        "The input ground truth bounding boxes file format contains repeated"
        " blocks of:\n"
        "  # image_index\n"
        "  img_path\n"
        "  channels\n"
        "  height\n"
        "  width\n"
        "  num_bounding_boxes\n"
        "  class_index x1 y1 x2 y2\n"
        "  <... num_bounding_boxes-1 more bounding boxes follow ...>\n";
        "The output window file format contains repeated blocks of:\n"
        "  # image_index\n"
        "  img_path\n"
        "  channels\n"
        "  height\n"
        "  width\n"
        "  num_windows\n"
        "  class_index x1 y1 x2 y2\n"
        "  <... num_windows-1 more windows follow ...>\n";
    return 1;
  }

  std::ifstream input_file(argv[1]);
  CHECK(input_file.good()) << "Failed to open bounding boxes ground truth file "
      << argv[1];

  std::ofstream output_file(argv[2]);
  CHECK(output_file.good()) << "Failed to open output file "
      << argv[2];

  ROIGeneratorParameter roi_param;
  SlidingWindowROIGeneratorParameter* param = roi_param.mutable_sliding_window_param();
  vector<string> spatial_bins;
  string spaital_bins_str = argv[3];
  boost::split(spatial_bins, spaital_bins_str, boost::is_any_of(","));
  for (size_t i = 0; i < spatial_bins.size(); ++i) {
    param->add_spatial_bin(boost::lexical_cast<int>(spatial_bins[i]));
  }
  param->set_stride_size_ratio(boost::lexical_cast<float>(argv[4]));
  shared_ptr<ROIGenerator<float> > roi_generator(
      new SlidingWindowROIGenerator<float>(roi_param));

  string hashtag;
  int image_index;
  std::string image_path;
  int channels;
  int height;
  int width;
  int num_bounding_boxes;
  int class_index;
  int x1;
  int y1;
  int x2;
  int y2;
  std::vector<int> class_indices;
  std::vector<Rect> bounding_boxes;
  std::vector<Rect> candidate_bboxes;
  std::vector<float> bbox_areas;
  Rect overlapped_rect;
  float overlapped_area;
  std::vector<int> overlapped_class_indices;
  std::vector<float> overlapped_ratios;
  std::vector<int> overlapped_candidate_indices;
  if (!(input_file >> hashtag >> image_index)) {
    LOG(FATAL) << "Bounding boxes ground truth file is empty" << argv[2];
  }
  do {
    CHECK_EQ(hashtag, "#");
    // read image path
    string image_path;
    input_file >> image_path;
    // read image dimensions
    vector<int> image_size(3);
    input_file >> channels >> height >> width >> num_bounding_boxes;
    bounding_boxes.clear();
    for (int i = 0; i < num_bounding_boxes; ++i) {
      input_file >> class_index >> x1 >> y1 >> x2 >> y2;
      class_indices.push_back(class_index);
      Rect bbox(x1, y1, x2, y2);
      bounding_boxes.push_back(bbox);
    }
    Blob<float> dummy(1, 1, height, width);
    roi_generator->generate(dummy, &candidate_bboxes);
    for (size_t i = 0; i < bounding_boxes.size(); ++i) {
      bbox_areas.push_back(bounding_boxes[i].area());
    }
    for (size_t i = 0; i < candidate_bboxes.size(); ++i) {
      for (size_t j = 0; j < bounding_boxes.size(); ++j) {
        overlapped_rect = candidate_bboxes[i].intersect(bounding_boxes[j]);
        overlapped_area = overlapped_rect.area();
        if (overlapped_area > 0) {
          overlapped_class_indices.push_back(class_indices[j]);
          overlapped_ratios.push_back(overlapped_area / bbox_areas[j]);
          overlapped_candidate_indices.push_back(i);
        }
      }
    }
    if (overlapped_candidate_indices.size() > 0) {
      output_file << "# " << image_index << std::endl << image_path << std::endl
          << channels << std::endl << height << std::endl << width << std::endl
          << overlapped_candidate_indices.size() << std::endl;
      for (size_t i = 0; i < overlapped_candidate_indices.size(); ++i) {
        output_file << overlapped_class_indices[i] << std::endl
            << overlapped_ratios[i] << std::endl
            << overlapped_candidate_indices[i] << std::endl;
      }
    }
  } while (input_file >> hashtag >> image_index);
  input_file.close();
  output_file.close();
  return 0;
}
