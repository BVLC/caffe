

#include <boost/unordered_map.hpp>

#include <string>

#include "caffe/caffe.hpp"
#include "caffe/multi_node/conv_node.hpp"

#include "caffe/multi_node/model_test_node.hpp"
#include "caffe/multi_node/msg_hub.hpp"
#include "caffe/multi_node/node_env.hpp"
#include "caffe/multi_node/sk_sock.hpp"

using boost::unordered_map;
// using namespace caffe;
using caffe::ModelRequest;
using caffe::NodeEnv;
using caffe::TestClient;
using caffe::TEST_NODE;

DEFINE_string(ip, "127.0.0.1", "the ip of the id and model server");
DEFINE_int32(id_port, 1955, "the tcp port of ID server");
DEFINE_int32(model_port, 1957, "the tcp port of model server");


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

  NodeEnv::set_node_role(TEST_NODE);

  ModelRequest rq;
  rq.mutable_node_info()->set_node_role(TEST_NODE);
  NodeEnv::set_model_request(rq);

  NodeEnv::InitNode();

  LOG(INFO) << "test node id: " << NodeEnv::Instance()->ID();

  shared_ptr<TestClient<float> > client(new TestClient<float>());

  client->Init();
  client->Poll();

  return 0;
}



