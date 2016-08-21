
#include <string>

#include "caffe/caffe.hpp"
#include "caffe/multi_node/ps_node.hpp"

using namespace caffe; // NOLINT (build namespace)

DEFINE_int32(ps_threads, 1, "number of parameter server threads");

DEFINE_string(ip, "127.0.0.1", "the ip of the id and model server");
DEFINE_int32(id_port, 1955, "the tcp port of ID server");
DEFINE_int32(model_port, 1957, "the tcp port of model server");

DEFINE_string(request, "models/bvlc_alexnet/ps.prototxt", \
"the location of the model request configuration file");
// DEFINE_string(request, "examples/cifar10/ps.prototxt",
// "the location of the model request configuration file");

int main(int argc, char** argv) {
  google::InstallFailureSignalHandler();
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  string id_server_addr = "tcp://";
  id_server_addr += FLAGS_ip;
  id_server_addr += ":";
  id_server_addr += boost::lexical_cast<string>(FLAGS_id_port);

  string model_server_addr = "tcp://";
  model_server_addr += FLAGS_ip;
  model_server_addr += ":";
  model_server_addr += boost::lexical_cast<string>(FLAGS_model_port);

  NodeEnv::set_model_server(model_server_addr);
  NodeEnv::set_id_server(id_server_addr);
  NodeEnv::set_request_file(FLAGS_request);
  NodeEnv::set_node_role(PARAM_SERVER);

  LOG(INFO) << "node id: " << NodeEnv::Instance()->ID();
  shared_ptr<ParamServer<float> > ps(new ParamServer<float>(FLAGS_ps_threads));

  ps->Init();
  ps->Poll();

  return 0;
}
