

#include "caffe/multi_node/model_map.hpp"

namespace caffe {

template <typename Dtype>
int ModelMap<Dtype>::FindLayer(const string& name)
{
  map<string, int>::iterator iter = layer_name_idx_.find(name);
  if (iter != layer_name_idx_.end()) {
    return iter->second;
  } else {
    return -1;
  }
}


template <typename Dtype>
void ModelMap<Dtype>::BuildNetGraph()
{
  int nlayers = net_param_.layer_size();

  requests_.resize(nlayers);
  layers_filled_.resize(nlayers);
  sub_solver_layers_.resize(nlayers);
  net_forward_graph_.resize(nlayers);
  net_backward_graph_.resize(nlayers);
  sub_input_blobs_.resize(nlayers);
  sub_output_blobs_.resize(nlayers);
  sub_forward_graph_.resize(nlayers);
  sub_backward_graph_.resize(nlayers);
  
  sub_skip_graph_.resize(nlayers);
  sub_skip_blobs_.resize(nlayers);
  sub_solver_layer_names_.resize(nlayers);

  for (int i = 0; i < nlayers; i++) {
    layer_params_.push_back(net_param_.layer(i));
    layer_name_idx_[net_param_.layer(i).name()] = i;
    layers_filled_[i] = false;
  }
  
  map<string, int> top_blob_to_layer;
  for (int i = 0; i < nlayers; i++) {
    const LayerParameter& layer_param = layer_params_[i];
    
    // set the nearest top blob as input
    for (int j = 0; j < layer_param.bottom_size(); j++) {
      map<string, int>::iterator iter = top_blob_to_layer.find(layer_param.bottom(j));
      CHECK(iter != top_blob_to_layer.end());
      net_forward_graph_[iter->second].push_back(i);
      net_backward_graph_[i].push_back(iter->second);
    }
   
    for (int j = 0; j < layer_param.top_size(); j++) {
      top_blob_to_layer[layer_param.top(j)] = i;
    }

  }
  
  LOG(INFO) << "net graph inited.";
  // print the graph
  for (int i = 0; i < layer_params_.size(); i++) {
    string debug_str = layer_params_[i].name();
    debug_str += ": ";
    for (int j = 0; j < net_forward_graph_[i].size(); j++) {
      debug_str += layer_params_[net_forward_graph_[i][j]].name();
      debug_str += ", ";
    }

    LOG(INFO) << debug_str;
  }

}

template <typename Dtype>
void ModelMap<Dtype>::ParseInputOutput(int node_idx)
{
  const vector<int>& all_layers = sub_solver_layers_[node_idx];

  map<string, bool> top_blob_is_used;
  for (int i = 0; i < all_layers.size(); i++) {
    const LayerParameter& param = layer_params_[all_layers[i]];

    for (int j = 0; j < param.bottom_size(); j++) {
      const string& bottom_blob = param.bottom(j);
      
      if (top_blob_is_used.find(bottom_blob) == top_blob_is_used.end()) {
        sub_input_blobs_[node_idx].push_back(bottom_blob);
      } else {
        top_blob_is_used[bottom_blob] = true;
      }
    }
    
    for (int j = 0; j < param.top_size(); j++) {
      const string& top_blob = param.top(j);
      top_blob_is_used[top_blob] = false;
    }
  }

  map<string, bool>::iterator iter;
  for (iter = top_blob_is_used.begin(); iter != top_blob_is_used.end(); iter++) {
    if (!iter->second) {
      sub_output_blobs_[node_idx].push_back(iter->first);
    }
  }
}

template <typename Dtype>
void ModelMap<Dtype>::AddLayers(NetParameter *pnet, int node_idx)
{
  for (int i = 0; i < sub_solver_layers_[node_idx].size(); i++) {
    int l = sub_solver_layers_[node_idx][i];
    LayerParameter layer_param = layer_params_[l];
    
    if (layer_param.type() == "Data") {
      layer_param.set_type(string("AsyncData"));
    }
    
    pnet->add_layer()->CopyFrom(layer_param);
  }
}


template <typename Dtype>
void ModelMap<Dtype>::AddInputs(NetParameter *pnet, int node_idx)
{
  for (int i = 0; i < sub_input_blobs_[node_idx].size(); i++) {
    const string& input_blob = sub_input_blobs_[node_idx][i];
    pnet->add_input(input_blob);

    const shared_ptr<Blob<Dtype> > b = net_->blob_by_name(input_blob);
    pnet->add_input_shape();

    for (int j = 0; j < b->shape().size(); j++) {
      pnet->mutable_input_shape(i)->add_dim(b->shape()[j]);
    }
  }
}

template <typename Dtype>
void ModelMap<Dtype>::AddSolver(RouteInfo *proute, int node_idx)
{
  //generate solver
  SolverParameter new_solver;
  new_solver.CopyFrom(clear_solver_);

  AddLayers(new_solver.mutable_net_param(), node_idx);
  AddInputs(new_solver.mutable_net_param(), node_idx);

  proute->mutable_solver_param()->CopyFrom(new_solver);
}

template <typename Dtype>
void ModelMap<Dtype>::AddRoutes(RouteInfo *proute, int node_idx)
{
  for (int i = 0; i < sub_forward_graph_[node_idx].size(); i++) {
    int fwd_idx = sub_forward_graph_[node_idx][i];
    for (int j = 0; j < requests_[fwd_idx].size(); j++) {
      proute->add_bcast_nodes()->CopyFrom(requests_[fwd_idx][j]->node_info());
    }
  }
  
  for (int i = 0; i < sub_backward_graph_[node_idx].size(); i++) {
    int bwd_id = sub_backward_graph_[node_idx][i];
    for (int j = 0; j < requests_[bwd_id].size(); j++) {
      proute->add_prev_nodes()->CopyFrom(requests_[bwd_id][j]->node_info());
    }
  }
  
  // add skip nodes
  for (int i = 0; i < sub_skip_graph_[node_idx].size(); i++) {
    int skip_id = sub_skip_graph_[node_idx][i];
    for (int j = 0; j < requests_[skip_id].size(); j++) {
      proute->add_fwrd_nodes()->CopyFrom(requests_[skip_id][j]->node_info());
      proute->add_fwrd_blob(sub_skip_blobs_[node_idx][i]);
    }
  }
}

template <typename Dtype>
void ModelMap<Dtype>::PrepareFCMsg()
{
  if (status_ != INITED) {
    return;
  }

  for (int i = 0; i < fc_nodes_.size(); i++) {
    int node_idx = fc_nodes_[i];
    for (int j = 0; j < requests_[node_idx].size(); j++) {
      shared_ptr<Msg> m(new Msg());
      m->set_type(GET_TRAIN_MODEL);
      m->set_dst(requests_[node_idx][j]->node_info().node_id());
      
      RouteInfo rt;
      AddSolver(&rt, node_idx);
      AddRoutes(&rt, node_idx);

      string rt_str;
      rt.SerializeToString(&rt_str);
      m->AppendData(rt_str.data(), rt_str.length());
      
      // Copy parameters should behind add rout info
      ParamHelper<Dtype>::CopyParamDataToMsg(psolver_->net(), sub_solver_layer_names_[node_idx], m);
     
      replies_.push_back(m);
    }
  }
}

template <typename Dtype>
void ModelMap<Dtype>::PreparePSMsg()
{
  if (status_ != INITED) {
    return;
  }
  for (int i = 0; i < ps_nodes_.size(); i++) {
    int node_idx = ps_nodes_[i];
    for (int j = 0; j < requests_[node_idx].size(); j++) {
      shared_ptr<Msg> m(new Msg);
      m->set_type(GET_TRAIN_MODEL);
      m->set_dst(requests_[node_idx][j]->node_info().node_id());

      RouteInfo rt;
      AddSolver(&rt, node_idx);

      string rt_str;
      rt.SerializeToString(&rt_str);
      m->AppendData(rt_str.data(), rt_str.length());
      
      // Copy parameters
      ParamHelper<Dtype>::CopyParamDataToMsg(psolver_->net(), sub_solver_layer_names_[node_idx], m);
     
      replies_.push_back(m);
    }
  }
}


template <typename Dtype>
void ModelMap<Dtype>::PrepareTestMsg()
{
  if (status_ != INITED) {
    return;
  }
  for (int i = 0; i < test_requests_.size(); i++) {
    shared_ptr<Msg> m(new Msg);
    m->set_type(GET_TRAIN_MODEL);
    m->set_dst(test_requests_[i]->node_info().node_id());

    // add solver for conv nodes
    RouteInfo rt;
    // add ps nodes
    for (int j = 0; j < ps_nodes_.size(); j++) {
      int ps_idx = ps_nodes_[j];
      rt.add_ps_nodes()->CopyFrom(requests_[ps_idx][0]->node_info());
    }
    
    for (int j = 0; j < fc_nodes_.size(); j++) {
      int fc_idx = fc_nodes_[i];
      rt.add_fc_nodes()->CopyFrom(requests_[fc_idx][0]->node_info());
    }
   
    rt.mutable_solver_param()->CopyFrom(test_solver_);
    
    string rt_str;
    rt.SerializeToString(&rt_str);
    m->AppendData(rt_str.data(), rt_str.length());

    replies_.push_back(m);
  }
}

template <typename Dtype>
void ModelMap<Dtype>::PrepareConvMsg()
{
  if (status_ != INITED) {
    return;
  }

  for (int i = 0; i < conv_requests_.size(); i++) {
    shared_ptr<Msg> m(new Msg);
    m->set_type(GET_TRAIN_MODEL);
    m->set_dst(conv_requests_[i]->node_info().node_id());

    // add solver for conv nodes
    RouteInfo rt;
    
    // add ps nodes
    for (int j = 0; j < ps_nodes_.size(); j++) {
      int ps_idx = ps_nodes_[j];
      rt.add_ps_nodes()->CopyFrom(requests_[ps_idx][0]->node_info());
    }

    // add gateway node
    rt.mutable_gateway_node()->CopyFrom(fc_gateway_->node_info());

    // add conv solver
    rt.mutable_solver_param()->CopyFrom(conv_solver_);
    
    string rt_str;
    rt.SerializeToString(&rt_str);
    m->AppendData(rt_str.data(), rt_str.length());

    replies_.push_back(m);
  }

  conv_requests_.clear();
}

template <typename Dtype>
void ModelMap<Dtype>::PrintRouteInfo()
{
  // print fc node route info
  LOG(INFO) << "forward route: ";
  for (int i = 0; i < fc_nodes_.size(); i++) {
    int node_idx = fc_nodes_[i];
    string dbg_str = "from: ";
    for (int j = 0; j < requests_[node_idx].size(); j++) {
      dbg_str += requests_[node_idx][j]->node_info().ip();
      dbg_str += ":";
      dbg_str += boost::lexical_cast<string>(requests_[node_idx][j]->node_info().router_port());
      dbg_str += ", ";
    }
    
    dbg_str += " to: ";
    for (int j = 0; j < sub_forward_graph_[node_idx].size(); j++) {
      int to_id = sub_forward_graph_[node_idx][j];
      for (int k = 0; k < requests_[to_id].size(); k++) {
        dbg_str += requests_[to_id][k]->node_info().ip();
        dbg_str += ":";
        dbg_str += boost::lexical_cast<string>(requests_[to_id][k]->node_info().router_port());
        dbg_str += ", ";
      }
    }

    LOG(INFO) << dbg_str;
  }

}

template <typename Dtype>
void ModelMap<Dtype>::FilterGatewayForwards(int gateway_idx)
{  
  // remove the blobs that are the same as inputs
  vector<int> clear_fwd_nodes;
  vector<string> clear_fwd_blobs;
  
  for (int i = 0; i < gateway_fwd_nodes_.size(); i++) {
    if (gateway_fwd_nodes_[i] != gateway_idx) {
      clear_fwd_nodes.push_back(gateway_fwd_nodes_[i]);
      clear_fwd_blobs.push_back(gateway_fwd_blobs_[i]);
    }
  }
  
  gateway_fwd_nodes_.clear();
  gateway_fwd_blobs_.clear();

  for (int i = 0; i < clear_fwd_nodes.size(); i++) {
    gateway_fwd_nodes_.push_back(clear_fwd_nodes[i]);
    gateway_fwd_blobs_.push_back(clear_fwd_blobs[i]);

    sub_skip_graph_[gateway_idx].push_back(clear_fwd_nodes[i]);
    sub_skip_blobs_[gateway_idx].push_back(clear_fwd_blobs[i]);
  }
}

template <typename Dtype>
void ModelMap<Dtype>::PrepareConvSolver()
{
  if (!(status_ == WAIT_FC_GATEWAY || status_ == INITED)) {
    return;
  }

  // if a node isn't a FC node, then it is a conv. node
  map<int, bool> fc_node_map;

  for (int i = 0; i < fc_nodes_.size(); i++) {
    int fc_idx = fc_nodes_[i];
    for (int j = 0; j < sub_solver_layers_[fc_idx].size(); j++) {
      fc_node_map[sub_solver_layers_[fc_idx][j]] = true;
    }
  }

  // add conv. layers in BFS order
  vector<int> bfs_vec;
  int bfs_idx = 0;
  bfs_vec.push_back(0);

  conv_solver_.CopyFrom(clear_solver_);

  while (bfs_idx < bfs_vec.size()) {
    int layer_idx = bfs_vec[bfs_idx];
    LayerParameter layer_param = layer_params_[layer_idx];
    
    // remove sync data layer
    if (layer_param.type() == "Data") {
      layer_param.set_type(string("AsyncData"));
    }

    conv_solver_.mutable_net_param()->add_layer()->CopyFrom(layer_param);

    for (int i = 0; i < net_forward_graph_[layer_idx].size(); i++) {
      int next_idx = net_forward_graph_[layer_idx][i];
      // if the node isn't FC node, then add it as conv node
      if (fc_node_map.find(next_idx) == fc_node_map.end()) {
        bfs_vec.push_back(next_idx);
      }
    }
    bfs_idx++;
  }
}


template <typename Dtype>
int ModelMap<Dtype>::PrepareRoutes()
{
  for (int i = 0; i < requests_.size(); i++) {
    if (requests_[i].size() > 0) {
      NodeRole role = requests_[i][0]->node_info().node_role();
      if (role == FC_NODE) {
        fc_nodes_.push_back(i);
      } else if (role == PARAM_SERVER) {
        ps_nodes_.push_back(i);
      } else {
        LOG(ERROR) << "Unknown node role: " << role;
      }
    }
  }
  
  for (int i = 0; i < fc_nodes_.size(); i++) {
    ParseInputOutput(fc_nodes_[i]);
  }

  //generate forward map and backward map for fc nodes
  map<string, int> nearest_top_blob_idx;
  for (int i = 0; i < fc_nodes_.size(); i++) {
    int node_idx = fc_nodes_[i];
    for (int j = 0; j < sub_input_blobs_[node_idx].size(); j++) {
      const string& input_blob = sub_input_blobs_[node_idx][j];
      map<string, int>::iterator iter = nearest_top_blob_idx.find(input_blob);
      
      if (iter != nearest_top_blob_idx.end()) {
        sub_backward_graph_[node_idx].push_back(iter->second);
        sub_forward_graph_[iter->second].push_back(node_idx);
      } else {
        // the gateway should be responsible for forwarding missing inputs
        gateway_fwd_nodes_.push_back(node_idx);
        gateway_fwd_blobs_.push_back(input_blob);
      }
    }

    for (int j = 0; j < sub_output_blobs_[node_idx].size(); j++) {
      const string& output_blob = sub_output_blobs_[node_idx][j];
      nearest_top_blob_idx[output_blob] = node_idx;
    }
  }

  
  vector<int> root_fc;
  for (int i = 0; i < fc_nodes_.size(); i++) {
    int node_idx = fc_nodes_[i];
    if (sub_backward_graph_[node_idx].size() <= 0) {
      root_fc.push_back(node_idx);
    }
  }

  CHECK_GT(root_fc.size(), 0) << "ERROR: no root fc nodes are found";
  if (root_fc.size() > 1) {
    status_ = WAIT_FC_GATEWAY;
  } else if (root_fc.size() == 1) {
    if (requests_[root_fc[0]].size() > 1) {
      status_ = WAIT_FC_GATEWAY;
    } else {
      fc_gateway_ = requests_[root_fc[0]][0];
      FilterGatewayForwards(root_fc[0]);
      status_ = INITED;
    }
  }
  
  if (status_ == INITED || status_ == WAIT_FC_GATEWAY) {
    PrintRouteInfo();
  }

  if (status_ == INITED) {
    return 0;
  } else if (status_ == WAIT_FC_GATEWAY && fc_gateway_ != NULL) {
    status_ = INITED;
    return 0;
  }

  return -1;
}

template <typename Dtype>
void ModelMap<Dtype>::AddModelRequest(shared_ptr<ModelRequest> rq)
{
  int start_idx = FindLayer(rq->start_layer());
  
  map<int, bool> end_layer_map;
  for (int i = 0; i < rq->end_layers_size(); i++) {
    int l = FindLayer(rq->end_layers(i));
    CHECK_GE(l, 0) << "cannot find layer: " << rq->end_layers(i) << " in request: "
      << std::endl << rq->DebugString();
    
    // add end layers
    end_layer_map[l] = true;
  }
  

  // add and check whether the request is valid or not
  if (requests_[start_idx].size() <= 0) {
    requests_[start_idx].push_back(rq);
  } else {
    CHECK_GE(rq->num_splits(), requests_[start_idx].size())
      << "Too many request at layer: " << std::endl 
      << layer_params_[start_idx].DebugString();
    CHECK_EQ(rq->num_splits(), requests_[start_idx][0]->num_splits())
      << "un-match splittings found in layer:" << std::endl
      << layer_params_[start_idx].DebugString();
    
    requests_[start_idx].push_back(rq);
  }
  
  // need to wait for more requests
  if (rq->num_splits() > requests_[start_idx].size()) {
    return;
  }
  
  // we have got all the reqeusts for a sub-solver
  // init subnet using constrained BFS to the graph
  // sub layers are added in BFS order
  // and mark the layers in subnet as filled
  vector<bool> visited;
  visited.resize(layer_params_.size());
  for (int i = 0; i < visited.size(); i++) {
    visited[i] = false;
  }
  vector<int> bfs_vec;
  for (map<int, bool>::iterator iter = end_layer_map.begin(); 
        iter != end_layer_map.end(); iter++) {
    bfs_vec.push_back(iter->first);
    visited[iter->first] = true;
  }
  int bfs_idx = 0;

  map<int, bool> backward_layers;
  // add all the layers from backward to forward
  while (bfs_idx < bfs_vec.size()) {
    int layer_idx = bfs_vec[bfs_idx];
    visited[layer_idx] = true;
    backward_layers[layer_idx] = true;
    visited[layer_idx] = true;
    
    // we stop at the start_idx
    if (layer_idx != start_idx) {
      for (int i = 0; i < net_backward_graph_[layer_idx].size(); i++) {
        int next_layer = net_backward_graph_[layer_idx][i];
        if (!visited[next_layer]) {
          bfs_vec.push_back(next_layer);
        }
      }
    }
    bfs_idx++;
  }

  // add layers to sub solver
  bfs_vec.clear();
  bfs_vec.push_back(start_idx);
  bfs_idx = 0;
  
  // clear all the layers in the request
  for (int i = 0; i < requests_[start_idx].size(); i++) {
    requests_[start_idx][i]->mutable_node_info()->clear_layers();
  }

  while (bfs_idx < bfs_vec.size()) {
    int layer_idx = bfs_vec[bfs_idx];
    CHECK(!layers_filled_[layer_idx]) 
      << "layer is filled by multiple nodes: " << layer_idx
      << std::endl << layer_params_[layer_idx].DebugString();
    
    layers_filled_[layer_idx] = true;
    sub_solver_layers_[start_idx].push_back(layer_idx);
    const string& layer_name = layer_params_[layer_idx].name();
    sub_solver_layer_names_[start_idx].push_back(layer_name);
    
    // add the layer to request
    for (int i = 0; i < requests_[start_idx].size(); i++) {
      requests_[start_idx][i]->mutable_node_info()->add_layers(layer_name);
    }

    // if the current layer is not end layer, put its child to the bfs vector
    if (end_layer_map.find(layer_idx) == end_layer_map.end()) {
      for (int i = 0; i < net_forward_graph_[layer_idx].size(); i++) {
        int next_layer = net_forward_graph_[layer_idx][i];
        // if the layer is in the backwarding layers
        if (backward_layers.find(next_layer) != backward_layers.end()) {
          bfs_vec.push_back(next_layer);
        }
      }
    }

    bfs_idx++;
  }
}

template <typename Dtype>
bool ModelMap<Dtype>::CheckIntegrity()
{
  for (int i = 0; i < layers_filled_.size(); i++) {
    if (!layers_filled_[i]) {
      return false;
    }
  }

  return true;
}


template <typename Dtype>
int ModelMap<Dtype>::ProcessTests(shared_ptr<Msg> m)
{
  shared_ptr<ModelRequest> rq(new ModelRequest());
  rq->ParseFromString( string( (char *)m->ZmsgData(0), m->ZmsgSize(0) ) );

  test_requests_.push_back(rq);

  if (status_ == INITED) {
    PrepareTestMsg();
    return 1;
  } else {
    LOG(INFO) << "waiting for mode map to be inited";
    return 0;
  }
}

template <typename Dtype>
int ModelMap<Dtype>::ProcessConv(shared_ptr<Msg> m)
{
  shared_ptr<ModelRequest> rq(new ModelRequest());
  rq->ParseFromString( string( (char *)m->ZmsgData(0), m->ZmsgSize(0) ) );

  conv_requests_.push_back(rq);

  if (status_ == INITED) {
    PrepareConvMsg();
    return 1;
  } else {
    LOG(INFO) << "model map hasn't been inited yet";
    return 0;
  }
}


template <typename Dtype>
int ModelMap<Dtype>::ProcessModels(shared_ptr<Msg> m)
{
  shared_ptr<ModelRequest> rq(new ModelRequest());
  rq->ParseFromString( string( (char *)m->ZmsgData(0), m->ZmsgSize(0) ) );
  
  AddModelRequest(rq);

  if (!CheckIntegrity()) {
    return 0;
  }
  
  if (PrepareRoutes() < 0) {
    return -1;
  }
  
  // we have all the fc nodes and ps nodes
  PrepareFCMsg();
  PreparePSMsg();
  
  // send messages to conv nodes and test nodes
  if (status_ == INITED) {
    PrepareConvSolver();
    PrepareConvMsg();
    PrepareTestMsg();
    return 1;
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

  //CHECK the node ID has been inited
  CHECK_GT(rq->node_info().node_id(), 0);
  NodeRole role = rq->node_info().node_role();

  CHECK_NE(role, INVALID_ROLE);

  if (role == CONV_CLIENT) {
    return ProcessConv(m);
  } else if (role == PARAM_SERVER) {
    return ProcessModels(m);
  } else if (role == FC_NODE) {
    return ProcessModels(m);
  } else if (role == TEST_NODE) {
    return ProcessTests(m);
  } else {
    LOG(ERROR) << "Unknown node role: " << role;
    return -1;
  }
}

INSTANTIATE_CLASS(ModelMap);

} //end caffe

