

#include "caffe/multi_node/msg_hub.hpp"

namespace caffe {

template <typename Dtype>
int MsgHub<Dtype>::StartThreads()
{
  for (int i = 0; i < nthreads_; i++) {
    string sk_addr(SERVER_SK_STR);
    sk_addr += "_";
    sk_addr += boost::lexical_cast<string>(i);
    
    sockp_arr_[i]->Bind(sk_addr);
    
    threads_[i]->SetWorkerId(i);
    threads_[i]->SetAddr(sk_addr);
    threads_[i]->StartInternalThread();
  }

  return 0;
}

template <typename Dtype>
int MsgHub<Dtype>::SetUpPoll()
{
  CHECK_GT (num_poll_items_, nthreads_);
  CHECK (poll_items_ != NULL);
  
  //initialize polling items for the work threads
  for (int i = 0; i < nthreads_; i++) {
    poll_items_[i].socket = sockp_arr_[i]->GetSock();
    poll_items_[i].events = ZMQ_POLLIN;
    poll_items_[i].fd = 0;
    poll_items_[i].revents = 0;
  }

  return 0;
}

//dispatch thread via poll
template <typename Dtype>
int MsgHub<Dtype>::Poll()
{
  SetUpPoll();

  while (true) {
    //blocked poll
    zmq_poll(poll_items_, num_poll_items_, -1);

    if (RouteMsg() < 0) {
      break;
    }
  }

  return 0;
}

//Worker threads process the incoming packets
template <typename Dtype>
int MsgHub<Dtype>::ProcessMsg(shared_ptr<Msg> m)
{ 
  int num_inputs = NodeEnv::Instance()->NumInputBlobs();

  if (num_inputs == 1 || num_inputs == 0) {
    sockp_arr_[0]->SendMsg(m);
  } else if (num_inputs == 2) {
    unordered_map<int64_t, shared_ptr<Msg> >::iterator iter = id_to_msg_.find(m->msg_id());

    if (iter == id_to_msg_.end()) {
        id_to_msg_[m->msg_id()] = m;
    } else {
        shared_ptr<Msg> s = iter->second;
        m->MergeMsg(s);
    }
    
    //double check the new msg
    iter = id_to_msg_.find(m->msg_id());
    shared_ptr<Msg> new_msg = iter->second;

    if (new_msg->num_blobs() == 2) {
        id_to_msg_.erase(iter);
        sockp_arr_[0]->SendMsg(m);
    }
  } else {
    LOG(ERROR) << "Cannot deal with input number: " << num_inputs;
    return -1;
  }

  return 0;
}

INSTANTIATE_CLASS(MsgHub);

} // end caffe

