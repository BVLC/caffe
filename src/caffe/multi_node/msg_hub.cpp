

#include "caffe/multi_node/msg_hub.hpp"

namespace caffe {

template <typename Dtype>
int MsgHub<Dtype>::StartThreads()
{
  for (int i = 0; i < nthreads_; i++) {
    string sk_addr(SERVER_SK_STR);
    sk_addr += "_";
    sk_addr += boost::lexical_cast<string>(i);
    
    string prior_addr = sk_addr + "_prior";

    sockp_arr_[i]->Bind(sk_addr);
    prior_socks_[i]->Bind(prior_addr);
    
    threads_[i]->SetWorkerId(i);
    threads_[i]->SetClientAddr(sk_addr);
    threads_[i]->SetPriorAddr(prior_addr);

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
int MsgHub<Dtype>::ScheduleMsg(shared_ptr<Msg> m)
{
  // we always put the message to the thread with minimal queue length
  int min_queue_size = threads_[0]->QueueSize();
  int min_queue_thread = 0;

  for (int i = 1; i < nworkers_; i++) {
    int s = threads_[i]->QueueSize();
    if (s < min_queue_size) {
      min_queue_size = s;
      min_queue_thread = i;
    }
  }

  Enqueue(min_queue_thread, m);
  return 0;
}

INSTANTIATE_CLASS(MsgHub);

} // end caffe

