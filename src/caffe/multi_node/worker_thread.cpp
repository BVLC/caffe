
#include <string>
#include <vector>

#include "caffe/multi_node/node_env.hpp"
#include "caffe/multi_node/worker_thread.hpp"

namespace caffe {
template <typename Dtype>
boost::mutex  WorkerThread<Dtype>::new_solver_mutex_;

template <typename Dtype>
boost::atomic_int WorkerThread<Dtype>::new_solver_cnt_(0);

template <typename Dtype>
void WorkerThread<Dtype>::SendExit() {
  shared_ptr<Msg> m(new Msg());

  // always use the 0th clock
  m->set_clock(0);
  m->set_src(ROOT_THREAD_ID);
  m->set_dst(WORKER_BCAST);
  m->set_type(EXIT_TRAIN);

  // For some reason we need a payload in message
  int pad = 0;
  m->AppendData(&pad, sizeof(pad));

  this->SendMsg(m); 
}

template <typename Dtype>
int WorkerThread<Dtype>::InitParamMap(shared_ptr<Net<Dtype> > net) {
  const vector<Blob<Dtype>*>& learn_params = net->learnable_params();
  unordered_map<void *, int> blob_to_idx;

  for (int i = 0; i < learn_params.size(); i++) {
    blob_to_idx[learn_params[i]] = i;
  }

  const vector<shared_ptr<Layer<Dtype> > >& layers = net->layers();

  layer_id_to_params_.resize(layers.size());
  int nlearn_layers = 0;

  for (int i = 0; i < layers.size(); i++) {
    shared_ptr<Layer<Dtype> > l = layers[i];
    vector<shared_ptr<Blob<Dtype> > >& layer_params = l->blobs();

    for (int j = 0; j < layer_params.size(); j++) {
      Blob<Dtype> *pblob = layer_params[j].get();
      unordered_map<void *, int>::iterator iter =
                                  blob_to_idx.find(pblob);
      CHECK(iter != blob_to_idx.end())
                << "cannot find learnable params for layer: " << i;

      layer_id_to_params_[i].push_back(iter->second);
    }

    if (layer_id_to_params_[i].size() > 0) {
      nlearn_layers++;
    }

    const string& layer_name = net->layer_names()[i];
    layer_id_by_name_[layer_name] = i;
  }

  return nlearn_layers;
}

INSTANTIATE_CLASS(WorkerThread);


}  // end namespace caffe


