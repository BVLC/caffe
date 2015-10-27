


#include "caffe/caffe.hpp"
#include "caffe/multi_node/ps_node.hpp"

using namespace caffe;

DEFINE_int32(ps_threads, 1, "number of parameter server threads");
DEFINE_string(ps_bind_addr, "tcp://*:9558", "zmq addr of the parameter server");

DEFINE_string(id_server_req, "tcp://10.239.156.44:9555", "the zmq REQ addr of the id / layer-map server");
DEFINE_string(model_server, "tcp://10.239.156.44:9557", "the address of zmq model server");
DEFINE_string(request_file, "examples/cifar10/conv.prototxt", "the location of the model request configuration file");

int main(int argc, char** argv)
{
  google::InstallFailureSignalHandler();
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  
  NodeEnv::set_model_server(FLAGS_model_server);
  NodeEnv::set_id_server(FLAGS_id_server_req);
  NodeEnv::set_request_file(FLAGS_request_file);
  NodeEnv::set_node_role(PARAM_SERVER);
 
  LOG(INFO) << "node id: " << NodeEnv::Instance()->ID();
  shared_ptr<ParamServer<float> > ps(new ParamServer<float>(FLAGS_ps_threads, FLAGS_ps_bind_addr));

  ps->Init();
  ps->Poll();

  return 0;
}


