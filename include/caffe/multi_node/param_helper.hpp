
#ifndef MULTI_NODE_PARAM_HELPER_H_
#define MULTI_NODE_PARAM_HELPER_H_

#include "caffe/multi_node/msg.hpp"
#include "caffe/caffe.hpp"
#include <map>
#include <string>


namespace caffe {

/*
 * help to copy layer parameters and data blobs
 */
template <typename Dtype>
class ParamHelper {

public:
  
  // check whether the two nets share parameters
  static bool IsParamShared(const shared_ptr<Net<Dtype> > net_l, const shared_ptr<Net<Dtype> > net_r) {
    const vector<Blob<Dtype>*>& params_l = net_l->learnable_params();
    const vector<Blob<Dtype>*>& params_r = net_r->learnable_params();
  
    if (params_l.size() != params_r.size()) {
      return false;
    }

    for (int i = 0; i < params_l.size(); i++) {
      if (params_l[i]->mutable_cpu_data() != params_r[i]->mutable_cpu_data()) {
        return false;
      }
    }

    return true;
  }
  
  // add src_net's diff to the diff of dst_net
  static void AddDiffFromNet(const shared_ptr<Net<Dtype> > dst_net, const shared_ptr<Net<Dtype> > src_net) {
    const vector<Blob<Dtype>*>& src_params = src_net->learnable_params();
    const vector<Blob<Dtype>*>& dst_params = dst_net->learnable_params();

    CHECK_EQ(src_params.size(), dst_params.size());

    for (int i = 0; i < src_params.size(); i++) {
      CHECK_EQ(src_params[i]->count(), dst_params[i]->count());
      caffe_axpy<Dtype>(src_params[i]->count(), 1.0, src_params[i]->cpu_diff(), dst_params[i]->mutable_cpu_diff());
    }
  }

  static void AddDiffFromMsg(const shared_ptr<Net<Dtype> > dst_net, shared_ptr<Msg> m) {
    for (int i = 0; i < m->num_blobs(); i++) {
      const BlobInfo& bi = m->blob_info(i);
      
      /// a layer is stored as a blob in the message
      const string& layer_name = bi.blob_name();
      const shared_ptr<Layer<Dtype> > l = dst_net->layer_by_name(layer_name);

      CHECK(l != NULL) << "Cannot find layer: " << layer_name;

      // TODO: filter shared layers
      CHECK_EQ(l->blobs().size(), bi.msg_index_size());

      for (int j = 0; j < l->blobs().size(); j++) {
        shared_ptr<Blob<Dtype> > pblob = l->blobs()[j];
        int m_idx = bi.msg_index(j);

        CHECK_EQ(pblob->count() * sizeof(Dtype), m->ZmsgSize(m_idx));
        caffe_axpy<Dtype>(pblob->count(), 1.0, (Dtype *)m->ZmsgData(m_idx), pblob->mutable_cpu_diff());
      }
    }
  }

  enum Action { COPY_DATA, COPY_DIFF };
  

  /// Copy layer parameters to a message
  static int CopyParamDataToMsg(shared_ptr<Net<Dtype> > net, const vector<string>& layer_names, shared_ptr<Msg> m) {
    return CopyParamToMsg(net, layer_names, m, COPY_DATA);
  }

  /// Copy layer parameter diff to a message
  static int CopyParamDiffToMsg(shared_ptr<Net<Dtype> > net, const vector<string>& layer_names, shared_ptr<Msg> m) {
    return CopyParamToMsg(net, layer_names, m, COPY_DIFF);
  }
  
  static int CopyParamToMsg(shared_ptr<Net<Dtype> > net, const vector<string>& layer_names, shared_ptr<Msg> m, Action act) {
    int blobs_copied = 0;
    for (int i = 0; i < layer_names.size(); i++) {
      const string& layer_name = layer_names[i];
      const shared_ptr<Layer<Dtype> > l = net->layer_by_name(layer_name);
      
      // TODO: filter shared layers
      blobs_copied += l->blobs().size();
      for (int j = 0; j < l->blobs().size(); j++) {
        shared_ptr<Blob<Dtype> > pblob = l->blobs()[j];
        if (act == COPY_DIFF) {
          m->AppendBlob(layer_name, pblob->cpu_diff(), pblob->count() * sizeof(Dtype));
        } else if (act == COPY_DATA) {
          m->AppendBlob(layer_name, pblob->cpu_data(), pblob->count() * sizeof(Dtype));
        } else {
          LOG(ERROR) << "unknown action: " << act;
        }
      }
    }

    return blobs_copied;
  }
  

  static int CopyParamFromMsg(shared_ptr<Net<Dtype> > net, shared_ptr<Msg> m, Action act) {
    int blobs_copied = 0;
    for (int i = 0; i < m->num_blobs(); i++) {
      const BlobInfo& bi = m->blob_info(i);
      
      /// a layer is stored as a blob in the message
      const string& layer_name = bi.blob_name();
      const shared_ptr<Layer<Dtype> > l = net->layer_by_name(layer_name);

      CHECK(l != NULL) << "Cannot find layer: " << layer_name;

      // TODO: filter shared layers
      CHECK_EQ(l->blobs().size(), bi.msg_index_size());
      blobs_copied += l->blobs().size();

      for (int j = 0; j < l->blobs().size(); j++) {
        shared_ptr<Blob<Dtype> > pblob = l->blobs()[j];
        int m_idx = bi.msg_index(j);

        CHECK_EQ(pblob->count() * sizeof(Dtype), m->ZmsgSize(m_idx));
        
        if (act == COPY_DIFF) {
          memcpy(pblob->mutable_cpu_diff(), m->ZmsgData(m_idx), pblob->count() * sizeof(Dtype)); 
        } else if (act == COPY_DATA) {
          memcpy(pblob->mutable_cpu_data(), m->ZmsgData(m_idx), pblob->count() * sizeof(Dtype));
        } else {
          LOG(ERROR) << "unknown action: " << act;
        }
      }
    }

    return blobs_copied;
  }

  /// copy parameters from a message
  static int CopyParamDataFromMsg(shared_ptr<Net<Dtype> > net, shared_ptr<Msg> m) {
    return CopyParamFromMsg(net, m, COPY_DATA);
  }
  
  /// copy diff from a message
  static int CopyParamDiffFromMsg(shared_ptr<Net<Dtype> > net, shared_ptr<Msg> m) {
    return CopyParamFromMsg(net, m, COPY_DIFF);
  }

  /// forward: net get input blob data from message
  static void CopyInputDataFromMsg(shared_ptr<Net<Dtype> > net, shared_ptr<Msg> m) {
    for (int i = 0; i < net->num_inputs(); i++) {
      int blob_index = net->input_blob_indices()[i];
      const string& blob_name = net->blob_names()[blob_index];
      Blob<Dtype>* pblob = net->input_blobs()[i];

      m->CopyBlob(blob_name, pblob->mutable_cpu_data(), pblob->count() * sizeof(Dtype));
    }
  }
  
  /// forward: net copy output blob data to a message
  static void CopyOutputDataToMsg(shared_ptr<Net<Dtype> > net, shared_ptr<Msg> m) {
    for (int i = 0; i < net->num_outputs(); i++) {
      int blob_index = net->output_blob_indices()[i];
      const string& blob_name = net->blob_names()[blob_index];
      Blob<Dtype>* pblob = net->output_blobs()[i];

      m->AddNewBlob(blob_name, pblob->cpu_data(), pblob->count() * sizeof(Dtype));
    }
  }
  
  /// backward: net copy input blob diffs to message
  static void CopyInputDiffToMsg(shared_ptr<Net<Dtype> > net, shared_ptr<Msg> m) {
    for (int i = 0; i < net->num_inputs(); i++) {
      int blob_index = net->input_blob_indices()[i];
      const string& blob_name = net->blob_names()[blob_index];
      Blob<Dtype>* pblob = net->input_blobs()[i];

      m->AddNewBlob(blob_name, pblob->cpu_diff(), pblob->count() * sizeof(Dtype));
    }
  }

  /// backward: net get output blob diffs from message
  static void CopyOutputDiffFromMsg(shared_ptr<Net<Dtype> > net, shared_ptr<Msg> m) {
    for (int i = 0; i < m->num_blobs(); i++) {
      const BlobInfo& bi = m->blob_info(i);
      const string& blob_name = bi.blob_name();
      shared_ptr<Blob<Dtype> > pblob = net->blob_by_name(blob_name);

      if (pblob != NULL) { 
        m->CopyBlob(blob_name, pblob->mutable_cpu_diff(), pblob->count() * sizeof(Dtype));
      }
    }
  }

  static int LayerSize(const shared_ptr<Layer<Dtype> > l) {
    int sz = 0;
    for (int i = 0; i < l->blobs().size(); i++) {
      sz += l->blobs()[i]->count() * sizeof(Dtype);
    }

    return sz;
  }

private:
  ParamHelper() { }
};


} // end namespace caffe


#endif

