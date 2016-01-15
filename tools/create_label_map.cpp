// This program reads in pairs label names and optionally ids and display names
// and store them in LabelMap proto buffer.
// Usage:
//   create_label_map [FLAGS] MAPFILE OUTFILE
// where MAPFILE is a list of label names and optionally label ids and
// displaynames, and OUTFILE stores the information in LabelMap proto.
// Example:
//   ./build/tools/create_label_map --delimiter=" " --include_background=true
//   data/VOC2007/map.txt data/VOC2007/labelmap_voc.prototxt
// The format of MAPFILE is like following:
//   class1 [1] [someclass1]
//   ...
// The format of OUTFILE is like following:
//   item {
//     name: "class1"
//     label: 1
//     display_name: "someclass1"
//   }
//   ...

#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

DEFINE_bool(include_background, false,
    "When this option is on, include none_of_the_above as class 0.");
DEFINE_string(delimiter, " ",
    "The delimiter used to separate fields in label_map_file.");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Read in pairs label names and optionally ids and "
        "display names and store them in LabelMap proto buffer.\n"
        "Usage:\n"
        "    create_label_map [FLAGS] MAPFILE OUTFILE\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/create_label_map");
    return 1;
  }

  const bool include_background = FLAGS_include_background;
  const string delimiter = FLAGS_delimiter;

  const string& map_file = argv[1];
  LabelMap label_map;
  ReadLabelFileToLabelMap(map_file, include_background, delimiter, &label_map);

  WriteProtoToTextFile(label_map, argv[2]);
}
