

#include <string>

#include "caffe/caffe.hpp"
#include "caffe/multi_node/fc_node.hpp"

using namespace caffe; // NOLINT (build namespace)

DEFINE_int32(threads, 1, "number of work threads in fc server");

DEFINE_int32(omp_param_threads, 0, "number of OMP threads in param thread");

DEFINE_string(ip, "127.0.0.1", "the ip of the id and model server");
DEFINE_int32(id_port, 1955, "the tcp port of ID server");
DEFINE_int32(model_port, 1957, "the tcp port of model server");

DEFINE_string(request, "models/bvlc_alexnet/fc.prototxt", \
"the location of the model request configuration file");
// DEFINE_string(request, "examples/cifar10/fc.prototxt",
// "the location of the model request configuration file");

void fc_server_thread() {
  LOG(INFO) << "fc node id: " << NodeEnv::Instance()->ID();

  if (NodeEnv::Instance()->is_fc_gateway()) {
    // work threads + one parameter thread
    shared_ptr<FcGateway<float> > fgate(new FcGateway<float>(
    FLAGS_threads + 1, FLAGS_omp_param_threads));

    fgate->Init();
    fgate->Poll();
  } else {
    // work thread + one parameter thread
    shared_ptr<FcClient<float> > fclient(new FcClient<float>(
    FLAGS_threads + 1, FLAGS_omp_param_threads));

    fclient->Init();
    fclient->Poll();
  }

  return;
}

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
  NodeEnv::set_node_role(FC_NODE);

  NodeEnv::InitNode();

  fc_server_thread();

  return 0;
}
