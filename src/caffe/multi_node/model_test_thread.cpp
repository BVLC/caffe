
#include <string>
#include <vector>

#include "caffe/multi_node/model_test_thread.hpp"

namespace caffe {

template <typename Dtype>
void TestThread<Dtype>::Run() {
  SendParamRquest();

  while (!this->must_stop()) {
    shared_ptr<Msg> m = this->RecvMsg(true);

    if (m->type() == TRAIN_ITER) {
      UpdateTrainIter(m);
    } else if (m->type() == PUT_PARAM) {
      UpdateParam(m);
    } else {
      LOG(WARNING) << "unknown type: " << m->type();
    }
  }
}

template <typename Dtype>
void TestThread<Dtype>::SendParamRquest() {
  for (int i = 0; i < ps_ids_.size(); i++) {
    shared_ptr<Msg> m(new Msg());
    m->set_type(GET_PARAM);
    m->set_dst(ps_ids_[i]);
    m->set_src(NodeEnv::Instance()->ID());

    // append something, avoid send packets withou payload
    m->AppendData(&i, sizeof(i));

    this->SendMsg(m);
  }

  for (int i = 0; i < fc_ids_.size(); i++) {
    shared_ptr<Msg> m(new Msg());
    m->set_type(GET_PARAM);
    m->set_dst(fc_ids_[i]);
    m->set_src(NodeEnv::Instance()->ID());

    // append something, avoid send packets withou payload
    m->AppendData(&i, sizeof(i));

    this->SendMsg(m);
  }
}

template <typename Dtype>
void TestThread<Dtype>::UpdateTrainIter(shared_ptr<Msg> m) {
  int iter_in_msg = *(reinterpret_cast<int *>(m->ZmsgData(0)));
  if (iter_in_msg > train_iter_) {
    train_iter_ = iter_in_msg;
  }

  if (train_iter_ - tested_iter_ >= param_.test_interval()) {
    tested_iter_ = train_iter_;
    SendParamRquest();
  }
}

template <typename Dtype>
void TestThread<Dtype>::UpdateParam(shared_ptr<Msg> m) {
  ParamHelper<Dtype>::CopyParamDataFromMsg(solver_->net(), m);

  updated_map_[m->src()] = true;

  for (int i = 0; i < ps_ids_.size(); i++) {
    if (!updated_map_[ps_ids_[i]]) {
      return;
    }
  }

  for (int i = 0; i < fc_ids_.size(); i++) {
    if (!updated_map_[fc_ids_[i]]) {
      return;
    }
  }

  LOG(INFO) << "param updated";
  ResetUpdateMap();

  // print net parameters
  #if 0
  const vector<shared_ptr<Layer<Dtype> > >& layers = solver_->net()->layers();
  const vector<string>& layer_names = solver_->net()->layer_names();
  for (int i = 0; i < layers.size(); i++) {
    shared_ptr<Layer<Dtype> > l = layers[i];
    for (int j = 0; j < l->blobs().size(); j++) {
      LOG(INFO) << "layer name: " << layer_names[i]
          << " blob: " << j << " data: " << l->blobs()[j]->cpu_data()[0];
    }
  }
  #endif

  solver_->TestAll(tested_iter_);

  if (param_.snapshot() &&
    train_iter_ - snapshot_iter_ >= param_.snapshot()) {
    snapshot_iter_ = train_iter_;
    solver_->Snapshot(snapshot_iter_);
  }
}

INSTANTIATE_CLASS(TestThread);

}  // end namespace caffe


