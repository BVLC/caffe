// Copyright 2014 Jeff Donahue
//
// This is a script to upgrade "V0" network prototxts to the new format.
// Usage:
//    upgrade_net_proto v0_net_proto_file_in net_proto_file_out

#include <cstring>
#include <iostream>  // NOLINT(readability/streams)
#include <fstream>  // NOLINT(readability/streams)

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

using std::ofstream;

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc != 3) {
    LOG(ERROR) << "Usage: "
        << "upgrade_net_proto v0_net_proto_file_in net_proto_file_out";
    return 0;
  }

  bool success = true;
  // First, check whether the input file is already in the new format.
  NetParameter upgraded_net_param;
  if (ReadProtoFromTextFile(argv[1], &upgraded_net_param)) {
    LOG(ERROR) << "File already in new proto format: " << argv[1];
  } else {
    V0NetParameter v0_net_param;
    CHECK(ReadProtoFromTextFile(argv[1], &v0_net_param))
        << "Failed to parse input V0NetParameter file: " << argv[1];
    success = UpgradeV0Net(v0_net_param, &upgraded_net_param);
    if (!success) {
      LOG(ERROR) << "Encountered one or more problems upgrading net param "
          << "proto; see above.";
    }
  }
  // TODO(jdonahue): figure out why WriteProtoToTextFile doesn't work
  // (no file is created).
  // WriteProtoToTextFile(upgraded_net_param, argv[2]);
  ofstream output_proto;
  output_proto.open(argv[2]);
  output_proto << upgraded_net_param.DebugString();
  output_proto.close();
  LOG(ERROR) << "Wrote upgraded NetParameter proto to " << argv[2];
  return !success;
}
