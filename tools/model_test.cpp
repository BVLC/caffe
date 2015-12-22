
#include "caffe/multi_node/sk_sock.hpp"
#include "caffe/multi_node/msg_hub.hpp"
#include "caffe/multi_node/model_test_node.hpp"
#include "caffe/caffe.hpp"

#include "boost/unordered_map.hpp"
#include "caffe/multi_node/node_env.hpp"
#include "caffe/multi_node/conv_node.hpp"

using boost::unordered_map;
using namespace caffe;

DEFINE_string(id_server_req, "tcp://127.0.0.1:1955", "the zmq REQ addr of the id / layer-map server");
DEFINE_string(model_server, "tcp://127.0.0.1:1957", "the address of zmq model server");


int main(int argc, char** argv)
{
  google::InstallFailureSignalHandler();
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  NodeEnv::set_model_server(FLAGS_model_server);
  NodeEnv::set_id_server(FLAGS_id_server_req);
  NodeEnv::set_node_role(TEST_NODE);
  
  ModelRequest rq;
  rq.mutable_node_info()->set_node_role(TEST_NODE);
  NodeEnv::set_model_request(rq);

  LOG(INFO) << "test node id: " << NodeEnv::Instance()->ID();

  shared_ptr<TestClient<float> > client(new TestClient<float>());

  client->Init();
  client->Poll();
 
  return 0;
}



