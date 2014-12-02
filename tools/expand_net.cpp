// This is a script expand templates layers to generate a network proto_file
// Usage:
//    expand_net net_proto_file_in net_proto_file_out

#include <cstring>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)

#include "caffe/caffe.hpp"
#include "caffe/util/expand_templates.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

using std::ofstream;

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc != 3) {
    LOG(ERROR) << "Usage: "
        << "expand_net net_proto_file_in net_proto_file_out";
    return 1;
  }

  NetParameter in_param;
  ReadNetParamsFromTextFileOrDie(argv[1], &in_param);

  NetParameter expanded(in_param);
  ExpandTemplatesNet(in_param, &expanded);

  // Convert to a NetParameterPrettyPrint to print fields in desired
  // order.
  NetParameterPrettyPrint net_param_pretty;
  NetParameterToPrettyPrint(expanded, &net_param_pretty);

  // Save new format prototxt.
  WriteProtoToTextFile(net_param_pretty, argv[2]);

  LOG(INFO) << "Wrote expanded NetParameter text proto to " << argv[2];
  return 0;
}
