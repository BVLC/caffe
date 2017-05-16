
#include "caffe/multi_node/sk_sock.hpp"
#include "caffe/multi_node/msg_hub.hpp"
#include "caffe/caffe.hpp"

#include "boost/unordered_map.hpp"
#include "caffe/multi_node/node_env.hpp"
#include "caffe/multi_node/conv_node.hpp"

using boost::unordered_map;
using namespace caffe;

DEFINE_int32(client_threads, 2, "number of convolution client threads");
DEFINE_string(fc_gateway_addr, "tcp://10.239.156.44:9556", "zmq address of the fc gateway");
DEFINE_string(ps_addr, "tcp://10.239.156.44:9558", "zmq address of the parameter server");

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
  NodeEnv::set_node_role(CONV_CLIENT);

  LOG(INFO) << "conv node id: " << NodeEnv::Instance()->ID();

  shared_ptr<ConvClient<float> > client(new ConvClient<float>(FLAGS_client_threads, FLAGS_fc_gateway_addr, FLAGS_ps_addr));

  client->Init();
  client->Poll();
 
  return 0;
}



