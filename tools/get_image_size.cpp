/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// This program retrieves the sizes of a set of images.
// Usage:
//   get_image_size [FLAGS] ROOTFOLDER/ LISTFILE OUTFILE
//
// where ROOTFOLDER is the root folder that holds all the images and
// annotations, and LISTFILE should be a list of files as well as their labels
// or label files.
// For classification task, the file should be in the format as
//   imgfolder1/img1.JPEG 7
//   ....
// For detection task, the file should be in the format as
//   imgfolder1/img1.JPEG annofolder1/anno1.xml
//   ....

#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

DEFINE_string(name_id_file, "",
              "A file which maps image_name to image_id.");

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Get sizes of a set of images.\n"
        "Usage:\n"
        "    get_image_size ROOTFOLDER/ LISTFILE OUTFILE\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/get_image_size");
    return 1;
  }

  std::ifstream infile(argv[2]);
  if (!infile.good()) {
    LOG(FATAL) << "Failed to open file: " << argv[2];
  }
  std::vector<std::pair<std::string, std::string> > lines;
  std::string filename, label;
  while (infile >> filename >> label) {
    lines.push_back(std::make_pair(filename, label));
  }
  infile.close();
  LOG(INFO) << "A total of " << lines.size() << " images.";

  const string name_id_file = FLAGS_name_id_file;
  std::map<string, int> map_name_id;
  if (!name_id_file.empty()) {
    std::ifstream nameidfile(name_id_file.c_str());
    if (!nameidfile.good()) {
      LOG(FATAL) << "Failed to open name_id_file: " << name_id_file;
    }
    std::string name;
    int id;
    while (nameidfile >> name >> id) {
      CHECK(map_name_id.find(name) == map_name_id.end());
      map_name_id[name] = id;
    }
    CHECK_EQ(map_name_id.size(), lines.size());
  }

  // Storing to outfile
  boost::filesystem::path root_folder(argv[1]);
  std::ofstream outfile(argv[3]);
  if (!outfile.good()) {
    LOG(FATAL) << "Failed to open file: " << argv[3];
  }
  int height, width;
  int count = 0;
  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    boost::filesystem::path img_file = root_folder / lines[line_id].first;
    GetImageSize(img_file.string(), &height, &width);
    std::string img_name = img_file.stem().string();
    if (map_name_id.size() == 0) {
      outfile << img_name << " " << height << " " << width << std::endl;
    } else {
      CHECK(map_name_id.find(img_name) != map_name_id.end());
      int img_id = map_name_id.find(img_name)->second;
      outfile << img_id << " " << height << " " << width << std::endl;
    }

    if (++count % 1000 == 0) {
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    LOG(INFO) << "Processed " << count << " files.";
  }
  outfile.flush();
  outfile.close();
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
