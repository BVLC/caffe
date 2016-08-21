
#ifndef MULTI_NODE_TEST_NODE_H_
#define MULTI_NODE_TEST_NODE_H_

#include <string>
#include <vector>

#include "caffe/multi_node/model_test_thread.hpp"
#include "caffe/multi_node/msg_hub.hpp"
#include "caffe/multi_node/node_env.hpp"

namespace caffe {

template <typename Dtype>
class TestClient : public MsgHub<Dtype>{
 public:
  // we only use 1 test client thread
  TestClient()
    : MsgHub<Dtype>(1, 1) {
    node_id_ = NodeEnv::Instance()->ID();
    ps_addrs_ = NodeEnv::Instance()->ps_addrs();
    ps_ids_ = NodeEnv::Instance()->ps_ids();
    fc_addrs_ = NodeEnv::Instance()->fc_addrs();
    fc_ids_ = NodeEnv::Instance()->fc_ids();
  }

  virtual ~TestClient() { }

 public:
  virtual int Init();

  virtual int RouteMsg();

 protected:
  virtual int SetUpPoll();

 protected:
  vector<string> ps_addrs_;

  vector<int> ps_ids_;

  vector<shared_ptr<SkSock> > ps_socks_;

  vector<string> fc_addrs_;

  vector<int> fc_ids_;

  vector<shared_ptr<SkSock> > fc_socks_;

  unordered_map<int, shared_ptr<SkSock> > node_id_to_sock_;

  int node_id_;
};


}  // end namespace caffe


#endif

