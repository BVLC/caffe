

#include "caffe/caffe.hpp"
#include <string>
#include "caffe/multi_node/fc_node.hpp"

using namespace caffe;

DEFINE_string(fc_server, "tcp://*:9556", "the zmq server addr of the fully conneted layers");
DEFINE_int32(server_threads, 2, "number of threads in fc server");
DEFINE_string(function, "fc_gateway", "function list: fc_server, fc_gateway, fc_client");

DEFINE_string(id_server_req, "tcp://10.239.156.44:9555", "the zmq REQ addr of the id / layer-map server");
DEFINE_string(model_server, "tcp://10.239.156.44:9557", "the address of zmq model server");
DEFINE_string(request_file, "examples/cifar10/fc.prototxt", "the location of the model request configuration file");


void fc_server_thread()
{
  LOG(INFO) << "fc node id: " << NodeEnv::Instance()->ID();

  if (FLAGS_function == "fc_gateway") {
    shared_ptr<FcGateway<float> > fgate(new FcGateway<float>(FLAGS_server_threads, FLAGS_fc_server));

    fgate->Init();
    fgate->Poll();
  } else if (FLAGS_function == "fc_client") {
    shared_ptr<FcClient<float> > fclient(new FcClient<float>(FLAGS_server_threads));

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
  
  NodeEnv::set_model_server(FLAGS_model_server);
  NodeEnv::set_id_server(FLAGS_id_server_req);
  NodeEnv::set_request_file(FLAGS_request_file);
  NodeEnv::set_node_role(FC_NODE);
 
  fc_server_thread();

  return 0;

}


