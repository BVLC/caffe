

#ifndef MULTI_NODE_MODEL_MAP_H_
#define MULTI_NODE_MODEL_MAP_H_

#include "caffe/multi_node/msg.hpp"
#include "caffe/multi_node/param_helper.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/util/insert_splits.hpp"

#include "caffe/proto/multi_node.pb.h"
#include <map>
#include <string>

namespace caffe {

template <typename Dtype>
class ModelMap
{

public:
  ModelMap(const string full_solver, int nworkers, int sub_solvers) {
    ReadProtoFromTextFileOrDie(full_solver, &solver_param_);
    
    // init test solver
    // test node should run in the same machine with model server
    test_solver_.CopyFrom(solver_param_);

    // clear test net in the solver param
    solver_param_.clear_test_net();
    solver_param_.clear_test_net_param();
    solver_param_.clear_test_iter();
    solver_param_.clear_test_interval();
    
    // currently we only deal with the net parameter is specified by a txt path
    CHECK(solver_param_.has_net());
    // init the net parameter
    NetParameter net_param;
    ReadNetParamsFromTextFileOrDie(solver_param_.net(), &net_param);

    num_sub_solvers_ = sub_solvers;
    
    // TRICK: overlapping communication and computation
    for (int i = 0; i < net_param.layer_size(); i++) {
      LayerParameter *player = net_param.mutable_layer(i);
      const string& layer_type = player->type();

      if (layer_type == "Data" || layer_type == "AsyncData") {
        int batch_size = player->data_param().batch_size();

        player->mutable_data_param()->set_batch_size(batch_size / num_sub_solvers_);
      }

      if (layer_type == "ImageData") {
        int batch_size = player->image_data_param().batch_size();
        player->mutable_image_data_param()->set_batch_size(batch_size / num_sub_solvers_);
      }
    }
    
    solver_param_.clear_net();
    solver_param_.mutable_net_param()->CopyFrom(net_param);
    
    Caffe::set_root_solver(true);
    psolver_ = SolverRegistry<Dtype>::CreateSolver(solver_param_);

    // init net parameter
    net_ = psolver_->net();

    InitTrainNetParam(net_param);

    status_ = WAIT_REQUESTS;

    num_workers_ = nworkers;

    LOG(INFO) << "Model map inited";
  }

  ~ModelMap() {
    delete psolver_;
  }
  
  // Copying weights from a trained file
  void CopyTrainedLayersFrom(const string& file_name) {
    psolver_->net()->CopyTrainedLayersFrom(file_name);
  }

  int GetModel(shared_ptr<Msg> m);

  const vector<shared_ptr<Msg> > &Replies() { return replies_; }

  void ClearMsg() { replies_.clear(); }

protected:
  void InitTrainNetParam(NetParameter &in_param) 
  {
    // filter test net
    NetState net_state;
    net_state.set_phase(TRAIN);
    net_state.MergeFrom(in_param.state());
    net_state.MergeFrom(solver_param_.train_state());
    in_param.mutable_state()->CopyFrom(net_state);

    net_->FilterNet(in_param, &net_param_);
    
    BuildNetGraph();

    // generate a clean solver(without any net)
    clear_solver_.CopyFrom(solver_param_);

    // delete all the existing nets...
    clear_solver_.release_train_net_param();
    clear_solver_.clear_train_net();
    clear_solver_.clear_net_param();
    clear_solver_.clear_net();

    // force sub solvers to use random seed
    clear_solver_.set_random_seed(-1);

    // generate clean net without any layers
    NetParameter clear_net;
    clear_net.CopyFrom(net_param_);
    clear_net.clear_layer();
    
    // TRICK to make caffe backward without loss layers
    clear_net.set_force_backward(true);
    
    clear_solver_.mutable_net_param()->CopyFrom(clear_net);
  }
  
  void BuildNetGraph();

protected:
  int FindLayer(const string& name);

  // generate input and output blobs
  void ParseInputOutput(int node_idx);
  
  // add a subnet to the graph
  // we need to guarantee that each layer has the same number of splits
  void AddModelRequest(shared_ptr<ModelRequest> rq);

  // check the integrity of the FC layers
  bool CheckIntegrity();
  
  // remove the gateway's input from its forwarding list
  // void FilterGatewayForwards(int gateway_idx);

  // prepare messages containing models and routing infomation
  int PrepareRoutes();
  
  /// process route request from conv nodes
  int ProcessConv(shared_ptr<Msg> m);

  /// process route request from nodes FC nodes or PS nodes
  /// models in FC and PS nodes should formulate a full graph
  int ProcessModels(shared_ptr<Msg> m);

  /// process requests from test nodes
  int ProcessTests(shared_ptr<Msg> m);

  // generate message for fc nodes
  void PrepareFCMsg();
  
  void PreparePSMsg();

  void PrepareConvMsg();

  void PrepareTestMsg();

  void PrepareConvSolver();

  void PrintRouteInfo();
  
  void AddLayers(NetParameter *pnet, int node_idx);

  void AddInputs(NetParameter *pnet, int node_idx);

  void AddRoutes(RouteInfo *proute, int node_idx);
  
  void AddSolver(RouteInfo *proute, int node_idx);
  
  // parse a request with start layer start_idx
  void ParseRequest(int start_idx);


protected:
  enum Status {
    WAIT_REQUESTS = 0,
    WAIT_FC_GATEWAY,
    INITED
  };

  Status status_;

  //we use the full solver to generate layers
  //ParaSolver *psolver_;
  Solver<Dtype> *psolver_;
  SolverParameter solver_param_;

  //clear solver parameter without layers, only have solver param
  SolverParameter clear_solver_;

  // solver for conv. nodes
  SolverParameter conv_solver_;

  // solver parameter for test nodes
  SolverParameter test_solver_;
  
  shared_ptr<Net<Dtype> > net_;
  NetParameter net_param_;

  // all the layers
  vector<LayerParameter> layer_params_;

  //
  vector<bool> layers_filled_;
  
  /// number of inputs of the layer
  vector<int> layer_inputs_;

  // map of layer names
  map<string, int> layer_name_idx_;

  // net forward graph
  vector<vector<int> > net_forward_graph_;

  // layers of the net is modeled as a graph
  vector<vector<int> > net_backward_graph_;
  
  // sub net forward graph
  vector<vector<int> > sub_forward_graph_;

  // sub net backward graph
  vector<vector<int> > sub_backward_graph_;
 
  // layers in a sub graph (sorted in BFS)
  vector<vector<int> > sub_solver_layers_;

  // the name of sub layers
  vector<vector<string> > sub_solver_layer_names_;

  // input, output and forward blobs are used for routing

  // name of input blobs for a sub graph
  vector<vector<string> > sub_input_blobs_;
  
  vector<vector<string> > sub_output_blobs_;

  // 
  vector<int> conv_fwd_nodes_;

  // indices to parameter server nodes
  vector<int> ps_nodes_;
  
  // indices to FC nodes
  vector<int> fc_nodes_;

  vector<int> fc_gateways_;
  
  // output nodes
  vector<int> output_nodes_;
  
  // store all the route nodes
  vector<vector<RouteNode> > route_nodes_;

  // store the model requests in a 2D vector
  vector<vector<shared_ptr<ModelRequest> > > requests_;
  
  // whether the requests of this layer is full filled
  vector<bool> request_filled_;
  
  // whether the request is parsed
  vector<bool> request_parsed_;
  
  // store the conv client request from data parallel cliens
  vector<shared_ptr<ModelRequest> > conv_requests_;
  
  // requests from testing nodes
  vector<shared_ptr<ModelRequest> > test_requests_;

  // the generated message for FC layers
  vector<shared_ptr<Msg> > replies_;
  
  int fc_batch_size_;

  int num_workers_;
  
  // number of overlapping sub solvers
  int num_sub_solvers_;

DISABLE_COPY_AND_ASSIGN(ModelMap);
};

} // end namespace caffe

#endif



