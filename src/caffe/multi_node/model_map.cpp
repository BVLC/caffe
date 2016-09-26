
#include <map>
#include <string>
#include <vector>

#include "caffe/multi_node/model_map.hpp"

namespace caffe {

template <typename Dtype>
int ModelMap<Dtype>::FindLayer(const string& name) {
  map<string, int>::iterator iter = layer_name_idx_.find(name);
  if (iter != layer_name_idx_.end()) {
    return iter->second;
  } else {
    return -1;
  }
}


template <typename Dtype>
void ModelMap<Dtype>::BuildNetGraph() {
  int nlayers = net_param_.layer_size();

  requests_.resize(nlayers);
  request_filled_.resize(nlayers);
  request_parsed_.resize(nlayers);
  layers_filled_.resize(nlayers);
  layer_inputs_.resize(nlayers);
  sub_solver_layers_.resize(nlayers);
  net_forward_graph_.resize(nlayers);
  net_backward_graph_.resize(nlayers);
  sub_input_blobs_.resize(nlayers);
  sub_output_blobs_.resize(nlayers);
  sub_forward_graph_.resize(nlayers);
  sub_backward_graph_.resize(nlayers);
  route_nodes_.resize(nlayers);

  sub_solver_layer_names_.resize(nlayers);

  for (int i = 0; i < nlayers; i++) {
    layer_params_.push_back(net_param_.layer(i));
    layer_name_idx_[net_param_.layer(i).name()] = i;
    layers_filled_[i] = false;
    layer_inputs_[i] = 0;
    request_filled_[i] = false;
    request_parsed_[i] = false;
  }

  map<string, int> top_blob_to_layer;
  for (int i = 0; i < nlayers; i++) {
    const LayerParameter& layer_param = layer_params_[i];

    // set the nearest top blob as input
    for (int j = 0; j < layer_param.bottom_size(); j++) {
      map<string, int>::iterator iter =
                                top_blob_to_layer.find(layer_param.bottom(j));
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
void ModelMap<Dtype>::ParseInputOutput(int node_idx) {
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
  for (iter = top_blob_is_used.begin();
       iter != top_blob_is_used.end();
       iter++) {
    if (!iter->second) {
      sub_output_blobs_[node_idx].push_back(iter->first);
    }
  }
}

template <typename Dtype>
void ModelMap<Dtype>::AddLayers(NetParameter *pnet, int node_idx) {
  for (int i = 0; i < sub_solver_layers_[node_idx].size(); i++) {
    int l = sub_solver_layers_[node_idx][i];
    LayerParameter layer_param = layer_params_[l];

    if (layer_param.type() == "Data") {
      layer_param.set_type(string("AsyncData"));
    }

    // FC model parallel by splitting the output size
    if (layer_param.has_inner_product_param()) {
      int num_output = layer_param.inner_product_param().num_output();
      int num_splits = requests_[node_idx].size();

      CHECK_EQ(num_output % num_splits, 0);
      layer_param.mutable_inner_product_param()->set_num_output(
                                                 num_output / num_splits);
    }

    pnet->add_layer()->CopyFrom(layer_param);
  }
}


template <typename Dtype>
void ModelMap<Dtype>::AddInputs(NetParameter *pnet, int node_idx) {
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
void ModelMap<Dtype>::AddSolver(RouteInfo *proute, int node_idx) {
  // generate solver
  SolverParameter new_solver;
  new_solver.CopyFrom(clear_solver_);

  AddLayers(new_solver.mutable_net_param(), node_idx);
  AddInputs(new_solver.mutable_net_param(), node_idx);

  proute->mutable_solver_param()->CopyFrom(new_solver);

  // set number of workers
  proute->set_num_workers(num_workers_);
  proute->set_num_sub_solvers(num_sub_solvers_);
}

template <typename Dtype>
void ModelMap<Dtype>::AddRoutes(RouteInfo *proute, int node_idx) {
  for (int i = 0; i < sub_forward_graph_[node_idx].size(); i++) {
    int fwd_idx = sub_forward_graph_[node_idx][i];
    for (int j = 0; j < requests_[fwd_idx].size(); j++) {
      proute->add_bcast_nodes()->CopyFrom(route_nodes_[fwd_idx][j]);
    }
  }

  for (int i = 0; i < sub_backward_graph_[node_idx].size(); i++) {
    int bwd_id = sub_backward_graph_[node_idx][i];
    for (int j = 0; j < requests_[bwd_id].size(); j++) {
      proute->add_prev_nodes()->CopyFrom(route_nodes_[bwd_id][j]);
    }
  }
}

template <typename Dtype>
int ModelMap<Dtype>::PrepareFCMsg() {
  if (status_ != INITED) {
    return 0;
  }

  int fc_msg = 0;
  for (int i = 0; i < fc_nodes_.size(); i++) {
    int node_idx = fc_nodes_[i];
    for (int j = 0; j < requests_[node_idx].size(); j++) {
      shared_ptr<Msg> m(new Msg());
      m->set_type(GET_TRAIN_MODEL);
      m->set_dst(requests_[node_idx][j]->node_info().node_id());

      RouteInfo rt;
      rt.mutable_node_info()->CopyFrom(requests_[node_idx][j]->node_info());

      AddSolver(&rt, node_idx);
      AddRoutes(&rt, node_idx);

      // rt.mutable_solver_param()->set_base_lr(0.05);

      NetParameter *pnet = rt.mutable_solver_param()->mutable_net_param();
      for (int k = 0; k < pnet->input_shape_size(); k++) {
        int n = pnet->input_shape(k).dim(0);
        pnet->mutable_input_shape(k)->set_dim(0, n);
      }

      // add gateway nodes
      for (int k = 0; k < fc_gateways_.size(); k++) {
        int gw_id = fc_gateways_[k];
        for (int l = 0; l < route_nodes_[gw_id].size(); l++) {
          rt.add_gateway_nodes()->CopyFrom(route_nodes_[gw_id][l]);
        }
      }

      string rt_str;
      rt.SerializeToString(&rt_str);
      m->AppendData(rt_str.data(), rt_str.length());

      int num_splits = requests_[node_idx].size();
      // Copy parameters should behind add rout info
      ParamHelper<Dtype>::CopyParamDataToMsg(psolver_->net(),
                                             sub_solver_layer_names_[node_idx],
                                              m, j, num_splits);
      fc_msg++;
      replies_.push_back(m);
    }
  }

  return fc_msg;
}

template <typename Dtype>
int ModelMap<Dtype>::PreparePSMsg() {
  // wait for the request from all the conv nodes
  if (status_ != COMPLETED) {
    return 0;
  }

  int ps_msg = 0;
  for (int i = 0; i < ps_routes_.size(); i++) {
    shared_ptr<Msg> m(new Msg);
    m->set_type(GET_TRAIN_MODEL);
    m->set_dst(ps_routes_[i].node_info().node_id());

    string rt_str;
    ps_routes_[i].SerializeToString(&rt_str);
    m->AppendData(rt_str.data(), rt_str.length());

    int node_idx = ps_nodes_[i];
    // Copy parameters
    ParamHelper<Dtype>::CopyParamDataToMsg(psolver_->net(),
                                             sub_solver_layer_names_[node_idx],
                                             m);

    replies_.push_back(m);
    ps_msg++;
  }

  return ps_msg;
}


template <typename Dtype>
int ModelMap<Dtype>::PrepareTestMsg() {
  if (status_ != COMPLETED) {
    return 0;
  }

  int test_msg = 0;
  for (int i = 0; i < test_requests_.size(); i++) {
    shared_ptr<Msg> m(new Msg);
    m->set_type(GET_TRAIN_MODEL);
    m->set_dst(test_requests_[i]->node_info().node_id());

    // add solver for conv nodes
    RouteInfo rt;
    // add ps nodes
    for (int j = 0; j < ps_nodes_.size(); j++) {
      int ps_idx = ps_nodes_[j];
      for (int k = 0; k < route_nodes_[ps_idx].size(); k++) {
        rt.add_ps_nodes()->CopyFrom(route_nodes_[ps_idx][k]);
      }
    }

    for (int j = 0; j < fc_nodes_.size(); j++) {
      int fc_idx = fc_nodes_[j];
      for (int k = 0; k < route_nodes_[fc_idx].size(); k++) {
        rt.add_fc_nodes()->CopyFrom(route_nodes_[fc_idx][k]);
      }
    }

    rt.mutable_solver_param()->CopyFrom(test_solver_);

    string rt_str;
    rt.SerializeToString(&rt_str);
    m->AppendData(rt_str.data(), rt_str.length());

    replies_.push_back(m);
    test_msg++;
  }

  return test_msg;
}

template <typename Dtype>
int ModelMap<Dtype>::PrepareConvMsg() {
  if (status_ != COMPLETED) {
    return 0;
  }

  int conv_msg = 0;
  for (int i = 0; i < conv_routes_.size(); i++) {
    shared_ptr<Msg> m(new Msg);
    m->set_type(GET_TRAIN_MODEL);
    m->set_dst(conv_routes_[i].node_info().node_id());

    string rt_str;
    conv_routes_[i].SerializeToString(&rt_str);
    m->AppendData(rt_str.data(), rt_str.length());

    replies_.push_back(m);
    conv_msg++;
  }

  return conv_msg;
}

template <typename Dtype>
void ModelMap<Dtype>::PrintRouteInfo() {
  // print fc node route info
  LOG(INFO) << "forward route: ";
  for (int i = 0; i < fc_nodes_.size(); i++) {
    int node_idx = fc_nodes_[i];
    string dbg_str = "from: ";
    for (int j = 0; j < requests_[node_idx].size(); j++) {
      dbg_str += requests_[node_idx][j]->node_info().ip();
      dbg_str += ":";
      dbg_str += boost::lexical_cast<string>(
                        requests_[node_idx][j]->node_info().router_port());
      dbg_str += ", ";
    }

    dbg_str += " to: ";
    for (int j = 0; j < sub_forward_graph_[node_idx].size(); j++) {
      int to_id = sub_forward_graph_[node_idx][j];
      for (int k = 0; k < requests_[to_id].size(); k++) {
        dbg_str += requests_[to_id][k]->node_info().ip();
        dbg_str += ":";
        dbg_str += boost::lexical_cast<string>(
                          requests_[to_id][k]->node_info().router_port());
        dbg_str += ", ";
      }
    }

    LOG(INFO) << dbg_str;
  }
}

template <typename Dtype>
void ModelMap<Dtype>::PrepareConvSolver() {
  if (status_ != COMPLETED) {
    return;
  }

  conv_solver_.CopyFrom(clear_solver_);

  // add all the layers in PS solvers
  for (int i = 0; i < request_parsed_.size(); i++) {
    if (!request_parsed_[i]) {
      continue;
    }

    CHECK_GT(requests_[i].size(), 0);
    CHECK_GT(route_nodes_[i].size(), 0);

    // check the request is PS or FC
    if (requests_[i][0]->node_info().node_role() != PARAM_SERVER) {
      continue;
    }

    // add the layers in the ps
    for (int j = 0; j < route_nodes_[i][0].layers_size(); j++) {
      int layer_idx = FindLayer(route_nodes_[i][0].layers(j));
      LayerParameter layer_param = layer_params_[layer_idx];

      // remove sync data layer
      if (layer_param.type() == "Data") {
        layer_param.set_type(string("AsyncData"));
      }

      conv_solver_.mutable_net_param()->add_layer()->CopyFrom(layer_param);
    }
  }
}


template <typename Dtype>
int ModelMap<Dtype>::PrepareRoutes() {
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

  for (int i = 0; i < ps_nodes_.size(); i++) {
    ParseInputOutput(ps_nodes_[i]);
  }

  // FC nodes that need blobs from conv. nodes
  vector<int> fwd_nodes;
  // the corresponding name of blob
  vector<vector<string> > fwd_blobs;
  fwd_blobs.resize(fc_nodes_.size());

  map<int, int> node_id_map;

  // generate forward map and backward map for fc nodes
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
        // unknown blobs are added as inputs
        map<int, int>::iterator id_iter = node_id_map.find(node_idx);
        if (id_iter != node_id_map.end()) {
          int node_offset = id_iter->second;
          fwd_blobs[node_offset].push_back(input_blob);
        } else {
          int node_offset = fwd_nodes.size();
          node_id_map[node_idx] = node_offset;
          fwd_nodes.push_back(node_idx);
          fwd_blobs[node_offset].push_back(input_blob);
        }
      }
    }

    for (int j = 0; j < sub_output_blobs_[node_idx].size(); j++) {
      const string& output_blob = sub_output_blobs_[node_idx][j];
      nearest_top_blob_idx[output_blob] = node_idx;
    }
  }

  // filter gateway nodes and fwd nodes
  for (int i = 0; i < fwd_nodes.size(); i++) {
    int node_idx = fwd_nodes[i];

    for (int j = 0; j < route_nodes_[node_idx].size(); j++) {
      for (int k = 0; k < fwd_blobs[i].size(); k++) {
        route_nodes_[node_idx][j].add_input_blobs(fwd_blobs[i][k]);
      }
    }

    if (sub_backward_graph_[node_idx].size() <= 0) {
      fc_gateways_.push_back(node_idx);
    } else {
      conv_fwd_nodes_.push_back(node_idx);
    }
  }

  CHECK_GT(fc_gateways_.size(), 0) << "ERROR: no root fc nodes are found";

  // set as gateway nodes
  for (int i = 0; i < fc_gateways_.size(); i++) {
    int gw_id = fc_gateways_[i];
    for (int j = 0; j < requests_[gw_id].size(); j++) {
      requests_[gw_id][j]->mutable_node_info()->set_node_role(FC_GATEWAY);
      LOG(INFO) << "Gateway node ip: " << requests_[gw_id][j]->node_info().ip()
        << ", node id: " << requests_[gw_id][j]->node_info().node_id();
    }
  }

  PrintRouteInfo();

  return 0;
}

template <typename Dtype>
void ModelMap<Dtype>::ParseRequest(int start_idx) {
  if (request_parsed_[start_idx]) {
    return;
  }

  LOG(INFO) << "inputs: " << layer_inputs_[start_idx]
    << ", bottom size: " << layer_params_[start_idx].bottom_size();

  if (layer_inputs_[start_idx] < layer_params_[start_idx].bottom_size()) {
    return;
  }

  shared_ptr<ModelRequest> rq = requests_[start_idx][0];
  map<int, bool> end_layer_map;
  for (int i = 0; i < rq->end_layers_size(); i++) {
    int l = FindLayer(rq->end_layers(i));
    CHECK_GE(l, 0) << "cannot find layer: "
                   << rq->end_layers(i) << " in request: "
                   << std::endl << rq->DebugString();

    // add end layers
    end_layer_map[l] = true;
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

  // store all the backwarding layers
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

  while (bfs_idx < bfs_vec.size()) {
    int layer_idx = bfs_vec[bfs_idx];

    CHECK(!layers_filled_[layer_idx]) << "layer: " << layer_idx
      << " is filled by multiple nodes "
      << std::endl << layer_params_[layer_idx].DebugString();

    layers_filled_[layer_idx] = true;

    sub_solver_layers_[start_idx].push_back(layer_idx);
    const string& layer_name = layer_params_[layer_idx].name();
    sub_solver_layer_names_[start_idx].push_back(layer_name);

    // add the layer to RouteNode
    route_nodes_[start_idx].resize(requests_[start_idx].size());
    for (int i = 0; i < requests_[start_idx].size(); i++) {
      NodeInfo *pnode = route_nodes_[start_idx][i].mutable_node_info();
      pnode->CopyFrom(requests_[start_idx][i]->node_info());
      route_nodes_[start_idx][i].add_layers(layer_name);
    }

    // if the current layer is not end layer, put its child to the bfs vector
    for (int i = 0; i < net_forward_graph_[layer_idx].size(); i++) {
      int next_layer = net_forward_graph_[layer_idx][i];

      layer_inputs_[next_layer]++;
      int num_inputs = layer_params_[next_layer].bottom_size();
      if (layer_inputs_[next_layer] < num_inputs) {
        continue;
      }

      // if the layer is the last layer specified by the request file
      if (end_layer_map.find(layer_idx) != end_layer_map.end()) {
        continue;
      }

      // if the layer is in the backwarding layers
      if (backward_layers.find(next_layer) != backward_layers.end()) {
        bfs_vec.push_back(next_layer);
      }
    }

    bfs_idx++;
  }

  request_parsed_[start_idx] = true;
}

template <typename Dtype>
void ModelMap<Dtype>::AddModelRequest(shared_ptr<ModelRequest> rq) {
  int start_idx = FindLayer(rq->start_layer());

  int sub_model_position = rq->node_info().position();
  // num splits means we split the layer into how many parts
  CHECK_GT(rq->num_splits(), sub_model_position);

  // add and check whether the request is valid or not
  if (requests_[start_idx].size() <= 0) {
    requests_[start_idx].resize(rq->num_splits());
    requests_[start_idx][sub_model_position] = rq;
  } else {
    CHECK_EQ(rq->num_splits(), requests_[start_idx].size())
      << "un-match splittings found in layer:" << std::endl
      << layer_params_[start_idx].DebugString();

    CHECK(requests_[start_idx][sub_model_position] == NULL)
      << "overlapped model requests:" << std::endl
      << requests_[start_idx][sub_model_position]->DebugString() << std::endl
      << rq->DebugString();

    requests_[start_idx][sub_model_position] = rq;
  }

  // need to wait for more requests
  for (int i = 0; i < requests_[start_idx].size(); i++) {
    if (requests_[start_idx][i] == NULL) {
      return;
    }
  }

  request_filled_[start_idx] = true;

  // requests are stored in the index of the start layer
  for (int i = 0; i < request_filled_.size(); i++) {
    if (request_filled_[i]) {
      ParseRequest(i);
    }
  }
}

template <typename Dtype>
bool ModelMap<Dtype>::CheckIntegrity() {
  for (int i = 0; i < layers_filled_.size(); i++) {
    if (!layers_filled_[i]) {
      return false;
    }
  }

  return true;
}

template <typename Dtype>
void ModelMap<Dtype>::SetUpConvRoutes() {
  if (status_ != COMPLETED) {
    return;
  }

  conv_routes_.clear();
  conv_routes_.resize(num_workers_);

  CHECK_GE(conv_requests_.size(), conv_routes_.size());

  PrepareConvSolver();

  RouteInfo rt;
  rt.set_num_sub_solvers(num_sub_solvers_);
  // add ps nodes
  for (int j = 0; j < ps_nodes_.size(); j++) {
    int ps_idx = ps_nodes_[j];
    for (int k = 0; k < route_nodes_[ps_idx].size(); k++) {
      rt.add_ps_nodes()->CopyFrom(route_nodes_[ps_idx][k]);
    }
  }

  // add gateway nodes
  for (int j = 0; j < fc_gateways_.size(); j++) {
    int gw_id = fc_gateways_[j];
    for (int k = 0; k < route_nodes_[gw_id].size(); k++) {
      rt.add_gateway_nodes()->CopyFrom(route_nodes_[gw_id][k]);
    }
  }

  // add fwd blobs and nodes
  for (int j = 0; j < conv_fwd_nodes_.size(); j++) {
    int fwd_id = conv_fwd_nodes_[j];
    for (int k = 0; k < route_nodes_[fwd_id].size(); k++) {
      rt.add_fwrd_nodes()->CopyFrom(route_nodes_[fwd_id][k]);
    }
  }

  // add conv solver
  rt.mutable_solver_param()->CopyFrom(conv_solver_);

  for (int i = 0; i < conv_routes_.size(); i++) {
    conv_routes_[i].CopyFrom(rt);
    NodeInfo *pnode = conv_routes_[i].mutable_node_info();
    pnode->CopyFrom(conv_requests_[i]->node_info());
  }
}

template <typename Dtype>
void ModelMap<Dtype>::SetUpPSRoutes() {
  if (status_ != COMPLETED) {
    return;
  }
  ps_routes_.resize(ps_nodes_.size());

  int k = 0;
  for (int i = 0; i < ps_nodes_.size(); i++) {
    int node_idx = ps_nodes_[i];
    for (int j = 0; j < requests_[node_idx].size(); j++) {
      AddSolver(&ps_routes_[k], node_idx);
      NodeInfo *pnode = ps_routes_[k].mutable_node_info();
      pnode->CopyFrom(requests_[node_idx][j]->node_info());

      k++;
    }
  }
}

template <typename Dtype>
int ModelMap<Dtype>::BuildReduceTree() {
  if (status_ != COMPLETED) {
    return 0;
  }

  SetUpPSRoutes();
  SetUpConvRoutes();

  // don't build reduce tree with multiple parameter servers
  if (ps_routes_.size() > 1) {
    return 0;
  }

  vector<RouteInfo *> rt_tree;

  rt_tree.push_back(&ps_routes_[0]);

  for (int i = 0; i < conv_routes_.size(); i++) {
    rt_tree.push_back(&conv_routes_[i]);
  }

  for (int i = 0; i < rt_tree.size() / 2; i++) {
    int lchild = 2 * i + 1;
    int rchild = 2 * i + 2;

    RouteNode parent_node;
    parent_node.mutable_node_info()->CopyFrom(rt_tree[i]->node_info());
    // set up binary tree
    if (lchild < rt_tree.size()) {
      RouteNode lnode;
      lnode.mutable_node_info()->CopyFrom(rt_tree[lchild]->node_info());
      rt_tree[i]->add_bcast_nodes()->CopyFrom(lnode);
      rt_tree[lchild]->add_prev_nodes()->CopyFrom(parent_node);
    }

    if (rchild < rt_tree.size()) {
      RouteNode rnode;
      rnode.mutable_node_info()->CopyFrom(rt_tree[rchild]->node_info());
      rt_tree[i]->add_bcast_nodes()->CopyFrom(rnode);
      rt_tree[rchild]->add_prev_nodes()->CopyFrom(parent_node);
    }
  }

  return 1;
}

template <typename Dtype>
int ModelMap<Dtype>::UpdateWorkers() {
  if (status_ != INITED) {
    return 0;
  }

  if (conv_requests_.size() < num_workers_) {
    return 0;
  }

  status_ = COMPLETED;
  BuildReduceTree();

  int r = 0;
  r += PreparePSMsg();
  r += PrepareConvMsg();
  r += PrepareTestMsg();

  return r;
}

template <typename Dtype>
int ModelMap<Dtype>::ProcessTests(shared_ptr<Msg> m) {
  shared_ptr<ModelRequest> rq(new ModelRequest());
  rq->ParseFromString(string(reinterpret_cast<char *>(m->ZmsgData(0)),
                             m->ZmsgSize(0)));

  test_requests_.push_back(rq);

  if (status_ == COMPLETED) {
    return PrepareTestMsg();
  } else {
    LOG(INFO) << "waiting for mode map to be inited";
    return 0;
  }
}

template <typename Dtype>
int ModelMap<Dtype>::ProcessConv(shared_ptr<Msg> m) {
  shared_ptr<ModelRequest> rq(new ModelRequest());
  rq->ParseFromString(string(reinterpret_cast<char *>(m->ZmsgData(0)),
                             m->ZmsgSize(0)));

  if (conv_requests_.size() >= num_workers_) {
    LOG(WARNING) << "too many clients";
    return 0;
  }
  conv_requests_.push_back(rq);

  if (status_ == INITED) {
    if (conv_requests_.size() < num_workers_) {
      return 0;
    } else {
      return UpdateWorkers();
    }
  } else {
    LOG(INFO) << "model map hasn't been inited yet";
    return 0;
  }
}


template <typename Dtype>
int ModelMap<Dtype>::ProcessModels(shared_ptr<Msg> m) {
  shared_ptr<ModelRequest> rq(new ModelRequest());
  rq->ParseFromString(string(reinterpret_cast<char *>(m->ZmsgData(0)),
                             m->ZmsgSize(0)));

  AddModelRequest(rq);

  if (!CheckIntegrity()) {
    return 0;
  }

  if (PrepareRoutes() < 0) {
    return -1;
  }

  status_ = INITED;

  // we have all the fc nodes and ps nodes
  int r = 0;
  r += PrepareFCMsg();

  r += UpdateWorkers();
  return r;
}

template <typename Dtype>
int ModelMap<Dtype>::GetModel(shared_ptr<Msg> m) {
  CHECK_GT(m->ZmsgCnt(), 0);

  shared_ptr<ModelRequest> rq(new ModelRequest());
  // route request is stored in the first message
  rq->ParseFromString(string(reinterpret_cast<char *>(m->ZmsgData(0)),
                             m->ZmsgSize(0)));

  LOG(INFO) << "Get Model Request: " << std::endl << rq->DebugString();

  // CHECK the node ID has been inited
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

}  // end namespace caffe

