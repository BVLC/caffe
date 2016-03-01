// This is a script to upgrade old solver prototxts to the new format.
// Usage:
//    upgrade_solver_proto_text old_solver_proto_file_in solver_proto_file_out

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
  FLAGS_alsologtostderr = 1;  // Print output to stderr (while still logging)
  ::google::InitGoogleLogging(argv[0]);
  if (argc != 3) {
    LOG(ERROR) << "Usage: upgrade_solver_proto_text "
        << "old_solver_proto_file_in solver_proto_file_out";
    return 1;
  }

  SolverParameter solver_param;
  string input_filename(argv[1]);
  if (!ReadProtoFromTextFile(input_filename, &solver_param)) {
    LOG(ERROR) << "Failed to parse input text file as SolverParameter: "
               << input_filename;
    return 2;
  }
  bool need_upgrade = SolverNeedsTypeUpgrade(solver_param);
  bool success = true;
  if (need_upgrade) {
    success = UpgradeSolverAsNeeded(input_filename, &solver_param);
    if (!success) {
      LOG(ERROR) << "Encountered error(s) while upgrading prototxt; "
                 << "see details above.";
    }
  } else {
    LOG(ERROR) << "File already in latest proto format: " << input_filename;
  }

  // Save new format prototxt.
  WriteProtoToTextFile(solver_param, argv[2]);

  LOG(INFO) << "Wrote upgraded SolverParameter text proto to " << argv[2];
  return !success;
}
