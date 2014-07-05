// Copyright 2014 BVLC and contributors.

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "caffe/caffe.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc != 3) {
    LOG(ERROR) << "Usage: generate_window_data_file"
        "  bounding_boxes_ground_truth_file  output_file";
    return 1;
  }

  std::ifstream input_file(argv[1]);
  CHECK(input_file.good()) << "Failed to open bounding boxes ground truth file "
      << argv[2];

  std::ifstream output_file(argv[2]);
  CHECK(output_file.good()) << "Failed to open output file "
      << argv[3];

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
  std::vector<float> bbox_areas;
  cv::Mat image;
  shared_ptr<ROIGenerator> roi_generator(new SlidingWindowROIGenerator());
  Rect overlapped_rect;
  float overlapped_area;
  std::vector<int> overlapped_class_indices;
  std::vector<float> overlapped_ratios;
  std::vector<int> overlapped_candidate_indices;
  if (!(input_file >> hashtag >> image_index)) {
    LOG(FATAL) << "Bounding boxes ground truth file is empty" << argv[2];
  }
  // caffe/layers/window_data_layer.cpp
  // window_file format
  // repeated:
  //    # image_index
  //    img_path (abs path)
  //    channels
  //    height
  //    width
  //    num_windows
  //    class_index overlap x1 y1 x2 y2
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
    roi_generator->generate(height, width, &candidate_bboxes);
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
      output_file << "# " << image_index << std::endl << file_path << std::endl
          << channels << std::endl << height << std::endl << width << std::endl
          << overlapped_bboxes_indices.size() << std::endl;
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
