

#include "caffe/caffe.hpp"
#include <string>
#include "caffe/multi_node/fc_node.hpp"

using namespace caffe;

DEFINE_int32(fc_threads, 2, "number of threads in fc server");
DEFINE_string(function, "fc_gateway", "function list: fc_server, fc_gateway, fc_client");

DEFINE_string(ip, "127.0.0.1", "the ip of the id and model server");
DEFINE_int32(id_port, 955, "the tcp port of ID server");
DEFINE_int32(model_port, 957, "the tcp port of model server");
DEFINE_string(request_file, "examples/cifar10/fc.prototxt", "the location of the model request configuration file");


void fc_server_thread()
{
  LOG(INFO) << "fc node id: " << NodeEnv::Instance()->ID();

  if (FLAGS_function == "fc_gateway") {
    shared_ptr<FcGateway<float> > fgate(new FcGateway<float>(FLAGS_fc_threads));

    fgate->Init();
    fgate->Poll();
  } else if (FLAGS_function == "fc_client") {
    shared_ptr<FcClient<float> > fclient(new FcClient<float>(FLAGS_fc_threads));

    fclient->Init();
    fclient->Poll();
  } else {
    LOG(ERROR) << "Unknown function: " << FLAGS_function;
  }

  return;
}



int main(int argc, char** argv)
{
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
  NodeEnv::set_request_file(FLAGS_request_file);
  NodeEnv::set_node_role(FC_NODE);
 
  fc_server_thread();

  return 0;
}



