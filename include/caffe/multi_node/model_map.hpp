

#ifndef MULTI_NODE_MODEL_MAP_H_
#define MULTI_NODE_MODEL_MAP_H_

#include "caffe/multi_node/msg.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/util/insert_splits.hpp"

#include "caffe/proto/multi_node.pb.h"


namespace caffe {

template <typename Dtype>
class ModelMap
{

public:
  ModelMap(const string full_solver) {
    ReadProtoFromTextFileOrDie(full_solver, &solver_param_);
    
    orig_solver_param_.CopyFrom(solver_param_);

    //clear test net in the solver param
    solver_param_.clear_test_net();
    solver_param_.clear_test_net_param();
    solver_param_.clear_test_iter();
    solver_param_.clear_test_interval();
    
    Caffe::set_root_solver(true);
    psolver_ = SolverRegistry<Dtype>::CreateSolver(solver_param_);
    
    //we only deal with the net parameter is specified by a txt path
    CHECK(solver_param_.has_net());

    //init the net parameter
    NetParameter in_param;
    ReadNetParamsFromTextFileOrDie(solver_param_.net(), &in_param);

    //add the parameter to the original solver param
    orig_solver_param_.clear_net();
    orig_solver_param_.mutable_net_param()->CopyFrom(in_param);

    //init net parameter
    net_ = psolver_->net();

    InitTrainNetParam(in_param);

    LOG(INFO) << "Model map inited";
  }

  ~ModelMap() {
    delete psolver_;
  }

  int GetModel(shared_ptr<Msg> m);
  
  //return full model and all the nodes in FC layer
  int GetFullModel(shared_ptr<Msg> m);

  const vector<shared_ptr<Msg> > &Replies() { return replies_; }

  void ClearMsg() { replies_.clear(); }

protected:
  void InitTrainNetParam(NetParameter &in_param) 
  {
    NetState net_state;
    net_state.set_phase(TRAIN);
    net_state.MergeFrom(in_param.state());
    net_state.MergeFrom(solver_param_.train_state());
    in_param.mutable_state()->CopyFrom(net_state);

    NetParameter filtered_param;
    net_->FilterNet(in_param, &filtered_param);
    
    //we only deal with train net in this class
    InsertSplits(filtered_param, &net_param_);
    
    //check we have the same number of layers
    CHECK(net_param_.layer_size() == net_->layers().size());

    const LayerParameter& first_layer = net_param_.layer(0);

    //the first layer should be Data layer and should has the data param
    CHECK(first_layer.type() == string("Data"));
    CHECK(first_layer.has_data_param());
    
    train_batch_size_ = first_layer.data_param().batch_size();
    
    const vector<shared_ptr<Layer<Dtype> > > &layers = net_->layers();
    
    //init the size of the request table
    requests_.resize(layers.size());

    //find the first FC layer
    first_fc_ = -1;
    for (int i = 0; i < layers.size(); i++) {
      shared_ptr<Layer<Dtype> > l = layers[i];

      LOG(INFO) << "layer index: " << i << " name: " << net_->layer_names()[i] << " type: " << l->type();
      if ( 0 == strcmp(l->type(), "InnerProduct") ) {
        first_fc_ = i;
        break;
      }
    }

    CHECK( first_fc_ >= 0 );
    
    //generate a clean solver(without any net)
    clear_solver_.CopyFrom(solver_param_);

    //delete all the existing nets...
    clear_solver_.release_train_net_param();
    clear_solver_.clear_train_net();
    clear_solver_.clear_net_param();
    clear_solver_.clear_net();

    //generate clean net without any layers
    clear_net_.CopyFrom(net_param_);
    clear_net_.clear_layer();

    fc_filled_ = false;
  }


protected:
  int FindLayer(string name);

  //generate a virtual net from first to end
  int GenerateSolver(SolverParameter &sparam, shared_ptr<ModelRequest> r);
  
  //Generating Rout infomation for a layer
  //note that all the sublayers in a layer have the same route infor
  int GenerateRoute(RouteInfo &rt, vector<shared_ptr<ModelRequest> > *pre, vector<shared_ptr<ModelRequest> > *next);

  //check the integrity of the FC layers
  bool CheckIntegrity();

  //prepare messages containing models and routing infomation
  int PrepareMessages();

  int PrepareFullModel();

  int ProcessConv(shared_ptr<Msg> m);

protected:
  //we use the full solver to generate layers
  //ParaSolver *psolver_;
  Solver<Dtype> *psolver_;
  SolverParameter solver_param_;
  //with test nets
  SolverParameter orig_solver_param_;

  //clear solver parameter without net param, only have solver param
  SolverParameter clear_solver_;
  
  shared_ptr<Net<Dtype> > net_;
  NetParameter net_param_;
  
  //clear net parameter without layers, only have net params
  NetParameter clear_net_;

  int first_fc_;

  //store the model requests in a 2D vector
  vector<vector<shared_ptr<ModelRequest> > > requests_;
  
  //all the nodes in FC layers
  vector<shared_ptr<ModelRequest> > fc_requests_;

  //the full request message
  shared_ptr<Msg> full_request_msg_;

  //the generated message for FC layers
  vector<shared_ptr<Msg> > replies_;

  //whether the FC layers are fully occupied.
  bool fc_filled_;

  //node in the bottom: usually loss nodes and need labels
  vector<NodeInfo> bottom_nodes_;

  //
  int train_batch_size_;

DISABLE_COPY_AND_ASSIGN(ModelMap);
};

} // end namespace caffe

#endif



