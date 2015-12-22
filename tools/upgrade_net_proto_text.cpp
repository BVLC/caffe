// This is a script to upgrade "V0" network prototxts to the new format.
// Usage:
//    upgrade_net_proto_text v0_net_proto_file_in net_proto_file_out

#include <cstring>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>

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
  string input_filename(argv[1]);
  if (!ReadProtoFromTextFile(input_filename, &net_param)) {
    LOG(ERROR) << "Failed to parse input text file as NetParameter: "
               << input_filename;
    return 2;
  }
  bool need_upgrade = NetNeedsUpgrade(net_param);
  bool need_data_upgrade = NetNeedsDataUpgrade(net_param);
  bool success = true;
  if (need_upgrade) {
    success = UpgradeNetAsNeeded(input_filename, &net_param);
    if (!success) {
      LOG(ERROR) << "Encountered error(s) while upgrading prototxt; "
                 << "see details above.";
    }
  } else {
    LOG(ERROR) << "File already in latest proto format: " << input_filename;
  }

  if (need_data_upgrade) {
    UpgradeNetDataTransformation(&net_param);
  }
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp

  // Convert to a NetParameterPrettyPrint to print fields in desired
  // order.
  NetParameterPrettyPrint net_param_pretty;
  NetParameterToPrettyPrint(net_param, &net_param_pretty);
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod/post-rebase-error-fix

  // Save new format prototxt.
  WriteProtoToTextFile(net_param, argv[2]);

  LOG(ERROR) << "Wrote upgraded NetParameter text proto to " << argv[2];
  return !success;
}
