

#include "caffe/multi_node/model_map.hpp"
#include <string>
#include <map>

namespace caffe {

template <typename Dtype>
int ModelMap<Dtype>::FindLayer(string name)
{
  //we only deal with consecutive layers
  for (int i = 0; i < net_->layer_names().size(); i++) {
    if ( net_->layer_names()[i] == name ) {
      return i;
    }
  }
  
  return -1;
}

template <typename Dtype>
int ModelMap<Dtype>::GenerateSolver(SolverParameter &sparam, shared_ptr<ModelRequest> r)
{
  int first = FindLayer(r->start_layer());
  int end = FindLayer(r->end_layer());

  map<string, int> blob_name_top_idx;
  
  //preparing new net param
  NetParameter nparam;

  nparam.CopyFrom(clear_net_);
  
  for (int i = first; i <= end; i++) {
    const LayerParameter& layer_param = net_param_.layer(i);
    nparam.add_layer()->CopyFrom(layer_param);

    //replace Data layer
    if (layer_param.type() == "Data") {
      nparam.mutable_layer(i)->set_type(string("AsyncData"));
    }
  }
  
  nparam.set_force_backward(true);
  
  //add unknown blobs as input blobs
  int ninputs = 0;
  for (int i = first; i <= end; i++) {
    const LayerParameter& layer_param = net_param_.layer(i);

    for (int j = 0; j < layer_param.bottom_size(); j++) {
      const string& blob_name = layer_param.bottom(j);
      
      if (blob_name_top_idx.find(blob_name) == blob_name_top_idx.end()) {

        LOG(INFO) << "add input " << blob_name;

        nparam.add_input( blob_name );

        //find the blob in net
        const shared_ptr<Blob<Dtype> > b = net_->blob_by_name ( blob_name );

        //add blob shapes
        nparam.add_input_shape();
        for (int k = 0; k < b->shape().size(); k++) {
          nparam.mutable_input_shape(ninputs)->add_dim(b->shape()[k]);
        }
        ninputs++;
      }
    }


    for (int j = 0; j < net_param_.layer(i).top_size(); j++) {
      const string& blob_name = layer_param.top(j);
      blob_name_top_idx[blob_name] = i;
    }
  }

  //preparing solver param
  sparam.CopyFrom(clear_solver_);

  //copy the generated NET to solver
  sparam.mutable_net_param()->CopyFrom(nparam);
  
  return 0;
}

template <typename Dtype>
int ModelMap<Dtype>::GenerateRoute(RouteInfo &rt, vector<shared_ptr<ModelRequest> > *pre, vector<shared_ptr<ModelRequest> > *next)
{
  //TODO: make it clearer?
  if (NULL != pre) {
    for (int i = 0; i < pre->size(); i++) {
      rt.add_prev_nodes()->CopyFrom( (*pre)[i]->node_info() );
    }
  }

  if (NULL != next) {
    for (int j = 0; j < next->size(); j++) {
      rt.add_next_nodes()->CopyFrom( (*next)[j]->node_info() );
    }
  }

  //adding bottom nodes to the route info
  for (int i = 0; i < bottom_nodes_.size(); i++) {
    rt.add_bottom_nodes()->CopyFrom(bottom_nodes_[i]);
  }
  
  rt.set_train_batch_size(train_batch_size_);

  return 0;
}

template <typename Dtype>
int ModelMap<Dtype>::PrepareMessages()
{
  //
  CHECK( fc_filled_) << "Need to get the FC layers prepared before sending route infomation";

  int start = first_fc_;
  int nlayers = net_->layer_names().size();
  
  vector<shared_ptr<ModelRequest> > *pre = NULL;
  vector<shared_ptr<ModelRequest> > *next = NULL;

  while (start < nlayers) {
    int end = FindLayer(requests_[start][0]->end_layer());

    if ((end + 1) >= nlayers) {
      next = NULL;
    } else {
      next = &(requests_[end + 1]);
    }

    RouteInfo rt;
    GenerateRoute(rt, pre, next);

    string rt_str;
    rt.SerializeToString(&rt_str);

    //generate message for each part of message
    for (int i = 0; i < requests_[start].size(); i++) {
      shared_ptr<Msg> m(new Msg());

      SolverParameter sparam;
      GenerateSolver(sparam, requests_[start][i]);

      string solver_str;
      sparam.SerializeToString(&solver_str);

      //setup m
      m->AppendData(rt_str.data(), rt_str.length());
      m->AppendData(solver_str.data(), solver_str.length());

      m->set_type(GET_TRAIN_MODEL);
      m->set_dst(requests_[start][i]->node_info().node_id());
      
      //push it to buffer
      replies_.push_back(m);
    }

    pre = &(requests_[start]);
    start = end + 1;
  }
  
  //check whether we need to prepare full model messages
  if (full_request_msg_ != NULL) {
    PrepareFullModel();
  }

  return 0;
}

template <typename Dtype>
bool ModelMap<Dtype>::CheckIntegrity()
{
  if (requests_[first_fc_].size() <= 0) {
    return false;
  }
  
  //we assume the layers in FC are always linear and straightforward
  int start = first_fc_;

  int nlayers = net_->layer_names().size();

  while (start < nlayers) {
    if (requests_[start].size() <= 0) {
      return false;
    }

    //check the layer segment is full filled
    for (int i = 0; i < requests_[start].size(); i++) {
      if (requests_[start][i] == NULL) {
        return false;
      }
    }

    //move to the next layer segment
    int end = FindLayer(requests_[start][0]->end_layer());
    start = end + 1;
  }
  
  fc_filled_ = true;

  //store all the fc nodes 
  start = first_fc_;
  while (start < nlayers) {
    for (int i = 0; i < requests_[start].size(); i++) {
      fc_requests_.push_back(requests_[start][i]);
    }

    int end = FindLayer(requests_[start][0]->end_layer());
    start = end + 1;
  }

  //all the FC layers are filled
  return true;
}

template <typename Dtype>
int ModelMap<Dtype>::ProcessConv(shared_ptr<Msg> m)
{
  shared_ptr<Msg> r(new Msg());
  
  //routing is empty
  RouteInfo rt;

  string rt_str;
  rt.SerializeToString(&rt_str);

  r->AppendData(rt_str.data(), rt_str.length());
  
  //preparing net param
  NetParameter nparam;

  nparam.CopyFrom(clear_net_);
  
  for (int i = 0; i < first_fc_; i++) {
    const LayerParameter& layer_param = net_param_.layer(i);
    nparam.add_layer()->CopyFrom(layer_param);
    
    //replace Data layer
    if (layer_param.type() == "Data") {
      nparam.mutable_layer(i)->set_type(string("AsyncData"));
    }

  }
  
  nparam.set_force_backward(true);
  
  //preparing solver param
  SolverParameter sparam;
  sparam.CopyFrom(clear_solver_);

  //copy the generated NET to solver
  sparam.mutable_net_param()->CopyFrom(nparam);
  
  string solver_str;

  sparam.SerializeToString(&solver_str);
  
  r->AppendData(solver_str.data(), solver_str.length());

  r->set_type(GET_TRAIN_MODEL);
  r->set_dst(m->src());
          
  //push it to buffer
  replies_.push_back(r);


  if (fc_filled_) {
    return 1;
  } else {
    return 0;
  }
}

template <typename Dtype>
int ModelMap<Dtype>::PrepareFullModel()
{
  CHECK(fc_filled_);

  shared_ptr<Msg> r(new Msg());

  string full_model_str;
  orig_solver_param_.SerializeToString(&full_model_str);

  r->AppendData(full_model_str.data(), full_model_str.length());

  for (int i = 0; i < fc_requests_.size(); i++) {
    string request_str;
    fc_requests_[i]->SerializeToString(&request_str);

    r->AppendData(request_str.data(), request_str.length());
  }

  r->set_dst(full_request_msg_->dst());
  r->set_type(full_request_msg_->type());

  full_request_msg_.reset();

  replies_.push_back(r);
  
  return 1;
}

template <typename Dtype>
int ModelMap<Dtype>::GetFullModel(shared_ptr<Msg> m)
{
  full_request_msg_ = m;
  
  //wait until 
  if (fc_filled_) {
    return PrepareFullModel();
  } else {
    return 0;
  }
}

template <typename Dtype>
int ModelMap<Dtype>::GetModel(shared_ptr<Msg> m)
{
  CHECK_GT(m->ZmsgCnt(), 0);

  shared_ptr<ModelRequest> rq(new ModelRequest());
  //route request is stored in the first message
  rq->ParseFromString( string( (char *)m->ZmsgData(0), m->ZmsgSize(0) ) );

  LOG(INFO) << "Get Model Request: " << std::endl << rq->DebugString();

  int start = FindLayer(rq->start_layer());
  CHECK_GE(start, 0);
  CHECK_LT(start, net_->layer_names().size());

  int end = FindLayer(rq->end_layer());
  int nlayers = net_->layer_names().size();
  CHECK_GT(end, 0);
  CHECK_LT(end, nlayers);
  
  int pos = rq->node_info().position();
  CHECK_LT(pos, rq->num_splits());
 
  //CHECK the node ID has been inited
  CHECK_GT(rq->node_info().node_id(), 0);

  if (start < first_fc_) {
    return ProcessConv(m);
  }

  if (requests_[start].size() <= 0) {
    //init the vector
    requests_[start].resize(rq->num_splits());
    requests_[start][pos] = rq;
  } else {
    CHECK_EQ(rq->end_layer(), requests_[start][0]->end_layer());
    CHECK_LT(pos, requests_[start].size());
    
    //only have one corresponding node
    CHECK(requests_[start][pos] == NULL) << "Two nodes are in the same position.";

    requests_[start][pos] = rq;
  }
  
  //adding bottom nodes
  if (end == nlayers - 1) {
    bottom_nodes_.push_back(rq->node_info());
  }

  if ( !CheckIntegrity() ) {
    return 0;
  } else {
    PrepareMessages();

    return 1;
  }

}

INSTANTIATE_CLASS(ModelMap);

} //end caffe

