// This program retrieves the sizes of a set of images.
// Usage:
//   get_image_size ROOTFOLDER LISTFILE OUTFILE
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
#include <string>
#include <utility>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

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
    outfile << img_file.stem().string() << " " << height << " " << width
        << std::endl;

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
