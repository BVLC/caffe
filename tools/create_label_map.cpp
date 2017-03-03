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

  return 0;
}
