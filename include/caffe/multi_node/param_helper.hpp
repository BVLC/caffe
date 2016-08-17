
#ifndef MULTI_NODE_PARAM_HELPER_H_
#define MULTI_NODE_PARAM_HELPER_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/multi_node/msg.hpp"
#include "caffe/sgd_solvers.hpp"


namespace caffe {

/*
 * help to copy layer parameters and data blobs
 */
template <typename Dtype>
class ParamHelper {
 public:
  // check whether the two nets share parameters
  static bool IsParamShared(const shared_ptr<Net<Dtype> > net_l,
                            const shared_ptr<Net<Dtype> > net_r) {
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
  static void AddDiffFromNet(const shared_ptr<Net<Dtype> > dst_net,
                             const shared_ptr<Net<Dtype> > src_net) {
    const vector<Blob<Dtype>*>& src_params = src_net->learnable_params();
    const vector<Blob<Dtype>*>& dst_params = dst_net->learnable_params();

    CHECK_EQ(src_params.size(), dst_params.size());

    for (int i = 0; i < src_params.size(); i++) {
      CHECK_EQ(src_params[i]->count(), dst_params[i]->count());
      caffe_axpy<Dtype>(src_params[i]->count(), 1.0,
                        src_params[i]->cpu_diff(),
                        dst_params[i]->mutable_cpu_diff());
    }
  }

  static void AddDiffFromNet(const shared_ptr<Net<Dtype> > dst_net,
                             const shared_ptr<Net<Dtype> > src_net,
                             int layer_id) {
    shared_ptr<Layer<Dtype> > dst_layer = dst_net->layers()[layer_id];
    vector<shared_ptr<Blob<Dtype> > >& dst_blobs = dst_layer->blobs();

    shared_ptr<Layer<Dtype> > src_layer = src_net->layers()[layer_id];
    vector<shared_ptr<Blob<Dtype> > >& src_blobs = src_layer->blobs();

    CHECK_EQ(dst_blobs.size(), src_blobs.size());

    for (int i = 0; i < src_blobs.size(); i++) {
      CHECK_EQ(src_blobs[i]->count(), dst_blobs[i]->count());
      caffe_axpy<Dtype>(src_blobs[i]->count(), 1.0,
                        src_blobs[i]->cpu_diff(),
                        dst_blobs[i]->mutable_cpu_diff());
    }
  }

  static void ScalDiff(const shared_ptr<Net<Dtype> > net,
                       Dtype factor,
                       int layer_id) {
    shared_ptr<Layer<Dtype> > l = net->layers()[layer_id];
    vector<shared_ptr<Blob<Dtype> > >& layer_blobs = l->blobs();

    for (int i = 0; i < layer_blobs.size(); i++) {
      caffe_scal(layer_blobs[i]->count(),
                 factor,
                 layer_blobs[i]->mutable_cpu_diff());
    }
  }

  static void ClearDiff(const shared_ptr<Net<Dtype> > net, int layer_id) {
    shared_ptr<Layer<Dtype> > l = net->layers()[layer_id];
    vector<shared_ptr<Blob<Dtype> > >& layer_blobs = l->blobs();
    for (int i = 0; i < layer_blobs.size(); i++) {
      caffe_set(layer_blobs[i]->count(),
                0,
                layer_blobs[i]->mutable_cpu_diff());
    }
  }

  static void PrintDiff(const shared_ptr<Net<Dtype> > net) {
    const vector<Blob<Dtype>*>& params = net->learnable_params();

    for (int i = 0; i < params.size(); i++) {
      LOG(INFO) << "param: "
                << i
                <<  ", diff[0]: "
                << params[i]->cpu_diff()[0];
    }
  }

  static void PrintParam(const shared_ptr<Net<Dtype> > net) {
    const vector<Blob<Dtype>*>& params = net->learnable_params();

    for (int i = 0; i < params.size(); i++) {
      LOG(INFO) << "param: "
                << i
                <<  ", data[0]: "
                << params[i]->cpu_data()[0];
    }
  }

  static void ScalDiff(const shared_ptr<Net<Dtype> > net, Dtype factor) {
    const vector<Blob<Dtype>*>& params = net->learnable_params();

    for (int i = 0; i < params.size(); i++) {
      caffe_scal(params[i]->count(), factor, params[i]->mutable_cpu_diff());
    }
  }

  static void CopyDiffFromNet(const shared_ptr<Net<Dtype> > dst_net,
                              const shared_ptr<Net<Dtype> > src_net) {
    const vector<Blob<Dtype>*>& src_params = src_net->learnable_params();
    const vector<Blob<Dtype>*>& dst_params = dst_net->learnable_params();

    CHECK_EQ(src_params.size(), dst_params.size());

    for (int i = 0; i < src_params.size(); i++) {
      CHECK_EQ(src_params[i]->count(), dst_params[i]->count());
      BlasCopy(dst_params[i]->count(),
               src_params[i]->cpu_diff(),
               dst_params[i]->mutable_cpu_diff());
    }
  }

  static void AddDiffFromMsg(const shared_ptr<Net<Dtype> > dst_net,
                             shared_ptr<Msg> m) {
    for (int i = 0; i < m->num_blobs(); i++) {
      const BlobInfo& bi = m->blob_info(i);

      // a layer is stored as a blob in the message
      const string& layer_name = bi.blob_name();
      const shared_ptr<Layer<Dtype> > l = dst_net->layer_by_name(layer_name);

      CHECK(l != NULL) << "Cannot find layer: " << layer_name;

      // TODO: filter shared layers
      CHECK_EQ(l->blobs().size(), bi.msg_index_size());

      for (int j = 0; j < l->blobs().size(); j++) {
        shared_ptr<Blob<Dtype> > pblob = l->blobs()[j];
        int m_idx = bi.msg_index(j);

        CHECK_EQ(pblob->count() * sizeof(Dtype), m->ZmsgSize(m_idx));
        caffe_axpy<Dtype>(pblob->count(), 1.0,
                         reinterpret_cast<Dtype *>(m->ZmsgData(m_idx)),
                         pblob->mutable_cpu_diff());
      }
    }
  }

  enum Action { COPY_DATA, COPY_DIFF };


  // Copy layer parameters to a message
  static int CopyParamDataToMsg(shared_ptr<Net<Dtype> > net,
                                const vector<string>& layer_names,
                                shared_ptr<Msg> m) {
    return CopyParamToMsg(net, layer_names, m, COPY_DATA, 0, 1);
  }

  static int CopyParamDataToMsg(shared_ptr<Net<Dtype> > net,
                                  const vector<string>& layer_names,
                                  shared_ptr<Msg> m,
                                  int pos, int num_splits) {
    return CopyParamToMsg(net, layer_names, m, COPY_DATA, pos, num_splits);
  }

  /// Copy layer parameter diff to a message
  static int CopyParamDiffToMsg(shared_ptr<Net<Dtype> > net,
                                const vector<string>& layer_names,
                                shared_ptr<Msg> m) {
    return CopyParamToMsg(net, layer_names, m, COPY_DIFF, 0, 1);
  }

  static int CopyParamToMsg(shared_ptr<Net<Dtype> > net,
                              const vector<string>& layer_names,
                              shared_ptr<Msg> m,
                              Action act,
                              int pos, int num_splits) {
    int blobs_copied = 0;
    for (int i = 0; i < layer_names.size(); i++) {
      const string& layer_name = layer_names[i];
      const shared_ptr<Layer<Dtype> > l = net->layer_by_name(layer_name);

      // TODO: filter shared layers
      blobs_copied += l->blobs().size();
      for (int j = 0; j < l->blobs().size(); j++) {
        shared_ptr<Blob<Dtype> > pblob = l->blobs()[j];

        int segment_len = pblob->count() / num_splits;
        int offset = pos * segment_len;

        if (act == COPY_DIFF) {
          m->AppendBlob(layer_name,
                        pblob->cpu_diff() + offset,
                        segment_len * sizeof(Dtype));
        } else if (act == COPY_DATA) {
          m->AppendBlob(layer_name,
                        pblob->cpu_data() + offset,
                        segment_len * sizeof(Dtype));
        } else {
          LOG(ERROR) << "unknown action: " << act;
        }
      }
    }

    return blobs_copied;
  }


  static int CopyParamFromMsg(shared_ptr<Net<Dtype> > net,
                              shared_ptr<Msg> m,
                              Action act) {
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
          BlasCopy(pblob->count(),
                   reinterpret_cast<Dtype *>(m->ZmsgData(m_idx)),
                   pblob->mutable_cpu_diff());
        } else if (act == COPY_DATA) {
          BlasCopy(pblob->count(),
                   reinterpret_cast<Dtype *>(m->ZmsgData(m_idx)),
                   pblob->mutable_cpu_data());
        } else {
          LOG(ERROR) << "unknown action: " << act;
        }
      }
    }

    return blobs_copied;
  }

  // copy parameters from a message
  static int CopyParamDataFromMsg(shared_ptr<Net<Dtype> > net,
                                  shared_ptr<Msg> m) {
    return CopyParamFromMsg(net, m, COPY_DATA);
  }


  static void CopyHistoryFromSolver(SGDSolver<Dtype> *dst_solver,
                                    SGDSolver<Dtype> *src_solver) {
    const vector<shared_ptr<Blob<Dtype> > >& src_hist = src_solver->history();
    const vector<shared_ptr<Blob<Dtype> > >& dst_hist = dst_solver->history();

    CHECK_EQ(src_hist.size(), dst_hist.size());

    for (int i = 0; i < src_hist.size(); i++) {
      shared_ptr<Blob<Dtype> > src_blob = src_hist[i];
      shared_ptr<Blob<Dtype> > dst_blob = dst_hist[i];

      CHECK_EQ(src_blob->count(), dst_blob->count());

      BlasCopy(src_blob->count(),
               src_blob->cpu_data(),
               dst_blob->mutable_cpu_data());
    }
  }

  static void CopyDataFromNet(const shared_ptr<Net<Dtype> > dst_net,
                              const shared_ptr<Net<Dtype> > src_net) {
    const vector<Blob<Dtype>*>& src_params = src_net->learnable_params();
    const vector<Blob<Dtype>*>& dst_params = dst_net->learnable_params();

    CHECK_EQ(src_params.size(), dst_params.size());

    for (int i = 0; i < src_params.size(); i++) {
      CHECK_EQ(src_params[i]->count(), dst_params[i]->count());
      BlasCopy(dst_params[i]->count(),
               src_params[i]->cpu_data(),
               dst_params[i]->mutable_cpu_data());
    }
  }

  // copy diff from a message
  static int CopyParamDiffFromMsg(shared_ptr<Net<Dtype> > net,
                                  shared_ptr<Msg> m) {
    return CopyParamFromMsg(net, m, COPY_DIFF);
  }

  // forward: net get input blob data from message
  static void CopyInputDataFromMsg(shared_ptr<Net<Dtype> > net,
                                   shared_ptr<Msg> m) {
    for (int i = 0; i < net->num_inputs(); i++) {
      int blob_index = net->input_blob_indices()[i];
      const string& blob_name = net->blob_names()[blob_index];
      Blob<Dtype>* pblob = net->input_blobs()[i];

      m->CopyBlob(blob_name,
                  pblob->mutable_cpu_data(),
                  pblob->count() * sizeof(Dtype));
    }
  }

  static int blob_stride(Blob<Dtype>* pblob) {
    // TODO: fixed with N * C * H * W
    return pblob->count(1);
  }

  static void CopyBlobFromMsg(Blob<Dtype>* pblob,
                              const string& blob_name,
                              shared_ptr<Msg> m,
                              Action act) {
    vector<int> msg_indices = m->blob_msg_indices(blob_name);

    vector<Dtype *> msg_data;
    for (int i = 0; i < msg_indices.size(); i++) {
      Dtype *pdata = reinterpret_cast<Dtype *>(m->ZmsgData(msg_indices[i]));
      msg_data.push_back(pdata);
    }
    int stride = blob_stride(pblob);

    CHECK_EQ(stride % msg_data.size(), 0);
    int segment_len = stride / msg_data.size();

    int msg_size = m->ZmsgSize(msg_indices[0]);
    msg_size /= sizeof(Dtype);
    CHECK_EQ(msg_size * msg_data.size(), pblob->count());

    int segment_offset = 0;
    Dtype *pblob_data = NULL;
    if (act == COPY_DATA) {
      pblob_data = pblob->mutable_cpu_data();
    } else if (act == COPY_DIFF) {
      pblob_data = pblob->mutable_cpu_diff();
    }

    CHECK(pblob_data != NULL);

    while (segment_offset < msg_size) {
      for (int i = 0; i < msg_data.size(); i++) {
        BlasCopy(segment_len, msg_data[i] + segment_offset, pblob_data);
        pblob_data += segment_len;
      }
      segment_offset += segment_len;
    }
  }

  // the blobs in "to" should be larger than "from"
  static void CopyBlobData(const vector<Blob<Dtype>*>& src_vec,
                           const vector<Blob<Dtype>*>& dst_vec,
                           int offset) {
    CHECK_EQ(src_vec.size(), dst_vec.size());

    for (int i = 0; i < src_vec.size(); i++) {
      Blob<Dtype>* src_blob = src_vec[i];
      Blob<Dtype>* dst_blob = dst_vec[i];

      int src_cnt = src_blob->count();
      int dst_cnt = dst_blob->count();

      if (src_cnt <= dst_cnt) {
        CHECK_EQ(dst_cnt % src_cnt, 0);
        CHECK_GT(dst_cnt, src_cnt * offset);

        Dtype *pdst = dst_blob->mutable_cpu_data() + src_cnt * offset;

        BlasCopy(src_cnt, src_blob->cpu_data(), pdst);
      } else {
        CHECK_EQ(src_cnt % dst_cnt, 0);
        CHECK_GT(src_cnt, dst_cnt * offset);

        const Dtype *psrc = src_blob->cpu_data() + dst_cnt * offset;
        BlasCopy(dst_cnt, psrc, dst_blob->mutable_cpu_data());
      }
    }
  }

  static void CopyBlobDiff(const vector<Blob<Dtype>*>& src_vec,
                           const vector<Blob<Dtype>*>& dst_vec,
                           int offset) {
    CHECK_EQ(src_vec.size(), dst_vec.size());

    for (int i = 0; i < src_vec.size(); i++) {
      Blob<Dtype>* src_blob = src_vec[i];
      Blob<Dtype>* dst_blob = dst_vec[i];

      int src_cnt = src_blob->count();
      int dst_cnt = dst_blob->count();

      if (src_cnt <= dst_cnt) {
        CHECK_EQ(dst_cnt % src_cnt, 0);
        CHECK_GT(dst_cnt, src_cnt * offset);

        Dtype *pdst = dst_blob->mutable_cpu_diff() + src_cnt * offset;

        BlasCopy(src_cnt, src_blob->cpu_diff(), pdst);
      } else {
        CHECK_EQ(src_cnt % dst_cnt, 0);
        CHECK_GT(src_cnt, dst_cnt * offset);

        const Dtype *psrc = src_blob->cpu_diff() + dst_cnt * offset;
        BlasCopy(dst_cnt, psrc, dst_blob->mutable_cpu_diff());
      }
    }
  }


  static void CopyBlobDataFromMsg(Blob<Dtype>* pblob,
                                  const string& blob_name,
                                  shared_ptr<Msg> m) {
    CopyBlobFromMsg(pblob, blob_name, m, COPY_DATA);
  }

  static void CopyBlobDiffFromMsg(Blob<Dtype>* pblob,
                                  const string& blob_name,
                                  shared_ptr<Msg> m) {
    CopyBlobFromMsg(pblob, blob_name, m, COPY_DIFF);
  }

  static void CopyBlobDataToMsg(shared_ptr<Net<Dtype> > net,
                                const vector<string>& blob_names,
                                shared_ptr<Msg> m,
                                int pos = 0,
                                int num_splits = 1) {
    for (int i = 0; i < blob_names.size(); i++) {
      const string& bname = blob_names[i];
      shared_ptr<Blob<Dtype> > pblob = net->blob_by_name(bname);

      CHECK(pblob != NULL);
      CHECK_EQ(pblob->count() % num_splits, 0);

      int segment_len = pblob->count() / num_splits;
      int offset = pos * segment_len;

      m->AddNewBlob(bname,
                    pblob->cpu_data() + offset,
                    segment_len * sizeof(Dtype),
                    pblob->shape());
    }
  }

  static void CopyBlobDataToMsg(const vector<shared_ptr<Net<Dtype> > >& nets,
                                const vector<string>& blob_names,
                                shared_ptr<Msg> m) {
    if (nets.size() <= 0) {
      return;
    }

    for (int i = 0; i < blob_names.size(); i++) {
      const string& bname = blob_names[i];

      shared_ptr<Blob<Dtype> > pblob = nets[0]->blob_by_name(bname);
      int blob_cnt = pblob->count();

      int blob_sz = blob_cnt * sizeof(Dtype) * nets.size();
      Dtype *p = reinterpret_cast<Dtype *>(m->AllocBlob(bname, blob_sz));

      for (int j = 0; j < nets.size(); j++) {
        shared_ptr<Blob<Dtype> > src_blob = nets[j]->blob_by_name(bname);

        CHECK(src_blob != NULL);

        BlasCopy(src_blob->count(), src_blob->cpu_data(), p);
        p += src_blob->count();
      }
    }
  }

  /// forward: net copy output blob data to a message
  static void CopyOutputDataToMsg(shared_ptr<Net<Dtype> > net,
                                  shared_ptr<Msg> m) {
    for (int i = 0; i < net->num_outputs(); i++) {
      int blob_index = net->output_blob_indices()[i];
      const string& blob_name = net->blob_names()[blob_index];
      Blob<Dtype>* pblob = net->output_blobs()[i];

      m->AddNewBlob(blob_name,
                    pblob->cpu_data(),
                    pblob->count() * sizeof(Dtype),
                    pblob->shape());
    }
  }

  static void CopyOutputDataToMsg(const vector<shared_ptr<Net<Dtype> > >& nets,
                                  shared_ptr<Msg> m) {
    if (nets.size() <= 0) {
      return;
    }

    int output_num = nets[0]->num_outputs();

    for (int i = 0; i < output_num; i++) {
      int blob_index = nets[0]->output_blob_indices()[i];
      const string& blob_name = nets[0]->blob_names()[blob_index];
      Blob<Dtype>* pblob = nets[0]->output_blobs()[i];

      int blob_cnt = pblob->count();

      int msg_sz = blob_cnt * sizeof(Dtype) * nets.size();
      Dtype *pmsg = reinterpret_cast<Dtype *>(m->AllocBlob(blob_name, msg_sz));
      Dtype *p = pmsg;

      for (int j = 0; j < nets.size(); j++) {
        Blob<Dtype>* src_blob = nets[j]->output_blobs()[i];
        BlasCopy(src_blob->count(), src_blob->cpu_data(), p);
        p += src_blob->count();
      }

      /*
      // for debug only
      Dtype sum = 0;
      for (int i = 0; i < blob_cnt * nets.size(); i++) {
        sum += pmsg[i];
      }
      LOG(INFO) << "data sum: " << sum;
      */
    }
  }

  /// backward: net copy input blob diffs to message
  static void CopyInputDiffToMsg(shared_ptr<Net<Dtype> > net,
                                 shared_ptr<Msg> m,
                                 int pos = 0,
                                 int num_splits = 1) {
    for (int i = 0; i < net->num_inputs(); i++) {
      int blob_index = net->input_blob_indices()[i];
      const string& blob_name = net->blob_names()[blob_index];
      Blob<Dtype>* pblob = net->input_blobs()[i];

      int blob_sz = pblob->count() * sizeof(Dtype) / num_splits;
      int stride = blob_stride(pblob);

      CHECK_EQ(stride % num_splits, 0);

      int nimg = pblob->count() / stride;
      int segment_len = stride / num_splits;
      int offset = pos * segment_len;

      Dtype *pmsg_data = reinterpret_cast<Dtype *>(m->AllocBlob(blob_name,
                                                                 blob_sz));
      const Dtype *pblob_diff = pblob->cpu_diff();

      if (offset != 0) {
        for (int j = 0; j < nimg; j++) {
          const Dtype *psrc = pblob_diff + offset;
          BlasCopy(segment_len, psrc, pmsg_data);
          pmsg_data += segment_len;
          pblob_diff += segment_len;
        }
      } else {
        BlasCopy(pblob->count(), pblob_diff, pmsg_data);
      }
    }
  }

  /// backward: net get output blob diffs from message
  static void CopyOutputDiffFromMsg(shared_ptr<Net<Dtype> > net,
                                    shared_ptr<Msg> m) {
    for (int i = 0; i < m->num_blobs(); i++) {
      const BlobInfo& bi = m->blob_info(i);
      const string& blob_name = bi.blob_name();
      shared_ptr<Blob<Dtype> > pblob = net->blob_by_name(blob_name);

      if (pblob != NULL) {
        m->CopyBlob(blob_name,
                    pblob->mutable_cpu_diff(),
                    pblob->count() * sizeof(Dtype));
      }
    }
  }

  static void CopyOutputDiffFromMsg(
                                const vector<shared_ptr<Net<Dtype> > >& nets,
                                shared_ptr<Msg> m) {
    CHECK_GT(nets.size(), 0);

    for (int i = 0; i < m->num_blobs(); i++) {
      const BlobInfo& bi = m->blob_info(i);
      const string& blob_name = bi.blob_name();
      shared_ptr<Blob<Dtype> > pblob = nets[0]->blob_by_name(blob_name);

      if (pblob == NULL) {
        continue;
      }

      int blob_cnt = pblob->count();
      int blob_sz = blob_cnt * sizeof(Dtype);

      CHECK_EQ(bi.msg_index_size(), 1) << "Couldn't support partial blobs";
      int msg_idx = bi.msg_index(0);
      int msg_sz = m->ZmsgSize(msg_idx);

      CHECK_EQ(blob_sz * nets.size(), msg_sz);
      Dtype *p = reinterpret_cast<Dtype *>(m->ZmsgData(msg_idx));

      for (int j = 0; j < nets.size(); j++) {
        shared_ptr<Blob<Dtype> > dst_blob = nets[j]->blob_by_name(blob_name);
        BlasCopy(blob_cnt, p, dst_blob->mutable_cpu_diff());

        p+= blob_cnt;
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

  static void BlasCopy(const int N, const Dtype* X, Dtype* Y);

 private:
  ParamHelper() { }
};


}  // end namespace caffe


#endif

