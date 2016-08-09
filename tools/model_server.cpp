

#include "boost/thread/thread.hpp"
#include "caffe/multi_node/model_map.hpp"
#include "caffe/multi_node/model_test_node.hpp"
#include "caffe/multi_node/sk_server.hpp"

DEFINE_string(id_server_req, "tcp://*:1955", \
        "the zmq REQ addr of the id / layer-map server");
DEFINE_string(model_server, "tcp://*:1957", "the address of zmq model server");

DEFINE_string(solver, "models/bvlc_alexnet/solver.prototxt",
        "location of solver");

DEFINE_string(weights, "",
          "Optional; the pretrained weights to initialize finetuning");

DEFINE_int32(workers, 0, "number of convolutional workers in fc server");

DEFINE_int32(sub_solvers, 1, 
        "number of overlapping sub-solvers in conv clients");

using caffe::ModelMap;
using caffe::PONG;
using caffe::Msg;
using caffe::SkServer;
using caffe::GET_TRAIN_MODEL;
using caffe::SkSock;
using caffe::PING;

// id server
// low frequence update
void rep_server_thread() {
  LOG(INFO) << "starting rep server in " << FLAGS_id_server_req;

  shared_ptr<SkSock> rep(new SkSock(ZMQ_REP));
  rep->Bind(FLAGS_id_server_req);

  int id = REQ_SERVER_ID + 1;

  while (true) {
    shared_ptr<Msg> m = rep->RecvMsg(true);

    if (m->type() == PING) {
      shared_ptr<Msg> s(new Msg());
      s->set_type(PONG);
      s->AppendData(&id, sizeof(id));
      rep->SendMsg(s);

      LOG(INFO) << "RP server add new sock id: " << id;

      id++;
    } else {
      LOG(ERROR) << "unknow message type: " << m->type();
    }
  }
}

// layer server, to send route info and solver parameters to clients
void model_server_thread() {
  LOG(INFO) << "starting model server in " << FLAGS_model_server;

  shared_ptr<SkServer> mserver(new SkServer());
  mserver->Bind(FLAGS_model_server);

  ModelMap<float> lmap(FLAGS_solver, FLAGS_workers, FLAGS_sub_solvers);

  if (FLAGS_weights.size() > 0) {
    lmap.CopyTrainedLayersFrom(FLAGS_weights);
  }

  while (true) {
    shared_ptr<Msg> m = mserver->RecvMsg(true);

    if (m->type() == GET_TRAIN_MODEL) {
      int r = lmap.GetModel(m);

      if (r > 0) {
        const vector<shared_ptr<Msg> > &reply = lmap.Replies();

        LOG(INFO) << "Send back response.";
        for (int i = 0; i < reply.size(); i++) {
          mserver->SendMsg(reply[i]);
        }

        // clear the message after send
        lmap.ClearMsg();
      }

    } else {
      LOG(ERROR) << "unknown message type: " << m->type();
    }
  }
}


int main(int argc, char** argv) {
  google::InstallFailureSignalHandler();
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  CHECK_GT(FLAGS_workers, 0) <<
              "Number of convolutional workers should be larger than 0";

  boost::thread id_thrd(&rep_server_thread);
  boost::thread model_thrd(&model_server_thread);

  id_thrd.join();
  model_thrd.join();

  return 0;
}


