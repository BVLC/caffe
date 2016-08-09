
#ifndef MULTI_NODE_MSG_H_
#define MULTI_NODE_MSG_H_

#include <vector>
#include <string>
#include <glog/logging.h>
#include <stdlib.h>
#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>

#include <zmq.h>

#include "caffe/proto/multi_node.pb.h"

using std::vector;
using std::string;
using std::map;

const int REQ_SERVER_ID = 3;
const int ROOT_THREAD_ID = 2;
const int WORKER_BCAST = 1;
const int INVALID_ID = -1;
const int INVALID_CLOCK = -1;

/// overlapping computation and communication 
/// by splitting the solver into sub-solvers
/// const int NUM_SUB_SOLVERS = 4;

using boost::shared_ptr;

namespace caffe {

class Msg {

public:
  Msg() {
  }
  
  Msg(shared_ptr<Msg> m) {
    set_src(m->src());
    set_dst(m->dst());
    set_msg_id(m->msg_id());
    set_clock(m->clock());
    set_type(m->type());
    set_conv_id(m->conv_id());
    set_data_offset(m->data_offset());
    set_is_partial(m->is_partial());
  }

  Msg(Msg *p) {
    set_src(p->src());
    set_dst(p->dst());
    set_msg_id(p->msg_id());
    set_clock(p->clock());
    set_type(p->type());
    set_conv_id(p->conv_id());
    set_data_offset(p->data_offset());
    set_is_partial(p->is_partial());
  }

  virtual ~Msg() {
    for (int i = 0; i < zmsg_vec_.size(); i++) {
      zmq_msg_close(zmsg_vec_[i]);
      delete zmsg_vec_[i];
    }
  }

  void set_src(int src) {
    header_.set_src(src);
  }

  void set_dst(int dst) {
    header_.set_dst(dst);
  }

  int src() const {
    return header_.src();
  }

  int dst() const {
    return header_.dst();
  }
  
  int clock() const {
    return header_.clock();
  }

  void set_clock(int clock) {
    header_.set_clock(clock);
  }

  int64_t msg_id() {
    return header_.msg_id();
  }

  void set_msg_id(int64_t id) {
    header_.set_msg_id(id);
  }

  void set_type(MsgType type) {
    header_.set_type(type);
  }

  MsgType type() const {
    return header_.type();
  }
  
  inline int data_offset() const {
    return header_.data_offset();
  }

  inline void set_data_offset(int offset) {
    header_.set_data_offset(offset);
  }
  
  inline bool is_partial() const {
    return header_.is_partial();
  }

  inline void set_is_partial(bool val) {
    header_.set_is_partial(val);
  }

  int num_blobs() {
    return header_.blobs_size();
  }

  inline const BlobInfo& blob_info(int index) {
    return header_.blobs(index);
  }

  void add_blob_info(const BlobInfo& b) {
    CHECK( !has_blob(b.blob_name()) );
    blob_name_index_[b.blob_name()] = header_.blobs_size();
    header_.add_blobs()->CopyFrom(b);
  }

  int AppendZmsg(zmq_msg_t *zmsg) {
    zmsg_vec_.push_back(zmsg);

    return zmsg_vec_.size() - 1;
  }
  
  int conv_id() {
    return header_.conv_id();
  }

  void set_conv_id(int id) {
    header_.set_conv_id(id);
  }

  //return the data index
  int AppendData(const void *p, int len) {
    zmq_msg_t *m = new zmq_msg_t;
    zmq_msg_init_size(m, len);

    memcpy(zmq_msg_data (m), p, len);
    zmsg_vec_.push_back(m);

    return zmsg_vec_.size() - 1;
  }
  
  int MergeMsg(shared_ptr<Msg> m);

  shared_ptr<Msg> ExtractMsg(const string& blob_name);

  //clear messages without releasing the content
  void ClearMsg() {
    header_.clear_blobs();
    zmsg_vec_.clear();
  }
  
  void PrintHeader() {
    LOG(INFO) << "message header: " << std::endl << header_.DebugString();
  }

  void SerializeHeader(string& str) {
    header_.SerializeToString(&str);
  }

  void ParseHeader(string& str) {
    header_.ParseFromString(str);

    //init the blob name map
    for (int i = 0; i < header_.blobs_size(); i++) {
      const string& blob_name = header_.blobs(i).blob_name();
      
      CHECK(blob_name_index_.find(blob_name) == blob_name_index_.end());
      blob_name_index_[blob_name] = i;
    }
  }
  
  bool has_blob(const string& blob_name) {
    return (blob_name_index_.find(blob_name) == blob_name_index_.end()) ? false : true;
  }
  
  //return -1 if didn't find the blob
  int blob_index_by_name(const string& blob_name) {
    if (!has_blob(blob_name)) {
      return -1;
    }

    return blob_name_index_[blob_name];
  }

  vector<int> blob_msg_indices(const string& blob_name) {
    vector<int> msg_indices;
    int blob_idx = blob_index_by_name(blob_name);

    if (blob_idx < 0) {
      return msg_indices;
    }
    
    const BlobInfo& bi = blob_info(blob_idx);
    for (int i = 0; i < bi.msg_index_size(); i++) {
      msg_indices.push_back(bi.msg_index(i));
    }

    return msg_indices;
  }

  void *ZmsgData(int i) {
    CHECK_LT(i, zmsg_vec_.size()) << "ERROR: vector size is " << zmsg_vec_.size() << " index is " << i;
    return zmq_msg_data(zmsg_vec_[i]);
  }

  void AddNewBlob(const string& blob_name, const void *data, int sz);
  
  // add new blob with shape
  void AddNewBlob(const string& blob_name, const void *data, int sz, const vector<int>& shape);
  
  void CopyBlob(const string& blob_name, void *data, int sz);
  
  // return the pointer to the allocated blob
  void *AllocBlob(const string& blob_name, int sz);

  /// Append a block of data to a blob
  /// Allocate a new blob if the blob doen't exist
  void AppendBlob(const string& blob_name, const void *data, int sz);

  int ZmsgSize(int i) {
    CHECK_LT(i, zmsg_vec_.size()) << "ERROR: vector size is " << zmsg_vec_.size() << " index is " << i;
    return zmq_msg_size(zmsg_vec_[i]);
  }

  zmq_msg_t *GetZmsg(int i) {
    CHECK(i < zmsg_vec_.size()) << "Getting index " << i << " from a vector with size of: " << zmsg_vec_.size();
    
    return zmsg_vec_[i];
  }

  int ZmsgCnt() {
    return zmsg_vec_.size();
  }

protected:
  MsgHeader header_;

  vector<zmq_msg_t *> zmsg_vec_;
  
  //map blob name to the blob info index where it is stored.
  map<string, int> blob_name_index_;
};

}

#endif // MULTI_NODE_MSG_H_


