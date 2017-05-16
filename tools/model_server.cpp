
#include "caffe/multi_node/sk_server.hpp"
#include "caffe/multi_node/model_map.hpp"
#include "boost/thread/thread.hpp"

DEFINE_string(id_server_req, "tcp://*:9555", "the zmq REQ addr of the id / layer-map server");
DEFINE_string(model_server, "tcp://*:9557", "the address of zmq model server");

DEFINE_string(full_solver, "examples/cifar10/cifar10_full_solver.prototxt", "location of full solver");

using namespace caffe;

//id server
//low frequence update
void rep_server_thread()
{
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


//layer server, to send route info and solver parameters to clients
void model_server_thread()
{
  LOG(INFO) << "starting model server in " << FLAGS_model_server;

  shared_ptr<SkServer> mserver(new SkServer());
  mserver->Bind(FLAGS_model_server);
  
  ModelMap<float> lmap(FLAGS_full_solver);
  
  while (true) {
    shared_ptr<Msg> m = mserver->RecvMsg(true);
    
    if (m->type() == GET_TRAIN_MODEL) {
      int r = lmap.GetModel(m);

      if (r > 0) {
        const vector<shared_ptr<Msg> > &reply = lmap.Replies();
        
        LOG(INFO) << "Send back response.";
        for (int i = 0; i < reply.size(); i++) {
          mserver->SendMsg( reply[i] );
        }
        
        //clear the message after send
        lmap.ClearMsg();
      }

    } else {
      LOG(ERROR) << "unknown message type: " << m->type();
    }
  }
}


int main(int argc, char** argv)
{
  google::InstallFailureSignalHandler();
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  
  boost::thread id_thrd(&rep_server_thread);
  boost::thread model_thrd(&model_server_thread);

  id_thrd.join();
  model_thrd.join();

  return 0;
}


