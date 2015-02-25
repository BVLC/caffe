// This is a script to upgrade "V0" network prototxts to the new format.
// Usage:
//    upgrade_net_proto_text v0_net_proto_file_in net_proto_file_out

#include <cstring>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

using std::ofstream;

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc != 3) {
    LOG(ERROR) << "Usage: "
        << "upgrade_net_proto_text v0_net_proto_file_in net_proto_file_out";
    return 1;
  }

  NetParameter net_param;
  if (!ReadProtoFromTextFile(argv[1], &net_param)) {
    LOG(ERROR) << "Failed to parse input text file as NetParameter: "
               << argv[1];
    return 2;
  }
  bool need_upgrade = NetNeedsUpgrade(net_param);
  bool need_data_upgrade = NetNeedsDataUpgrade(net_param);
  bool success = true;
  if (need_upgrade) {
    NetParameter v0_net_param(net_param);
    success = UpgradeV0Net(v0_net_param, &net_param);
  } else {
    LOG(ERROR) << "File already in V1 proto format: " << argv[1];
  }

  if (need_data_upgrade) {
    UpgradeNetDataTransformation(&net_param);
  }

  // Convert to a NetParameterPrettyPrint to print fields in desired
  // order.
  NetParameterPrettyPrint net_param_pretty;
  NetParameterToPrettyPrint(net_param, &net_param_pretty);

  // Save new format prototxt.
  WriteProtoToTextFile(net_param_pretty, argv[2]);

  LOG(ERROR) << "Wrote upgraded NetParameter text proto to " << argv[2];
  return !success;
}
