

#include "caffe/multi_node/msg.hpp"

namespace caffe {

int Msg::MergeMsg(shared_ptr<Msg> m)
{
  CHECK_EQ(m->msg_id(), header_.msg_id());
  
  for (int i = 0; i < m->num_blobs(); i++) {
    const BlobInfo &src_blob = m->blob_info(i);

    int blob_index = blob_index_by_name(src_blob.blob_name());
    
    //atatching the blobs to the packet
    if (blob_index >= 0) {
      BlobInfo *tgt_blob = header_.mutable_blobs(blob_index); 
      
      for (int k = 0; k < src_blob.logical_offset_size(); k++) {
        tgt_blob->add_logical_offset(src_blob.logical_offset(k));
        
        int msg_index = AppendZmsg(m->GetZmsg(src_blob.msg_index(k)));
        tgt_blob->add_msg_index(msg_index);
      }

    } else {
      BlobInfo new_blob(src_blob);
      
      // keep the logical offset but ajust physical offset
      new_blob.clear_msg_index();
      for (int k = 0; k < src_blob.msg_index_size(); k++) {
        new_blob.add_msg_index(AppendZmsg(m->GetZmsg(src_blob.msg_index(k))));
      }

      add_blob_info(new_blob);
    }
  }

  //clear the incoming packet
  m->ClearMsg();

  return 0;
}

//extract a blob from the message
shared_ptr<Msg> Msg::ExtractMsg(const string blob_name)
{
  shared_ptr<Msg> m;
  
  //skip if only have 1 blob
  if (num_blobs() <= 1) {
    return m;
  }

  int blob_index = blob_index_by_name(blob_name);
  
  if (blob_index < 0) {
    LOG(WARNING) << "cannot find blob: " << blob_name;
    return m;
  }

  m.reset(new Msg(this));
  
  const BlobInfo& chosen_blob = blob_info(blob_index);
  BlobInfo new_blob;
  
  new_blob.set_blob_name(chosen_blob.blob_name());

  for (int i = 0; i < chosen_blob.msg_index_size(); i++) {
    int msg_index = chosen_blob.msg_index(i);
    zmq_msg_t *pmsg = zmsg_vec_[msg_index];
    zmsg_vec_[msg_index] = NULL;

    new_blob.add_msg_index(m->AppendZmsg(pmsg));
    new_blob.add_logical_offset(chosen_blob.logical_offset(i)); 
  }

  m->add_blob_info(new_blob);

  vector<BlobInfo> update_info;
  vector<zmq_msg_t *> update_msg;

  //update the old blobs
  for (int i = 0; i < num_blobs(); i++) {
    if (i == blob_index) {
      continue;
    }
    
    const BlobInfo old_blob = blob_info(i);
    BlobInfo update_blob;
    
    update_blob.set_blob_name(old_blob.blob_name());

    for (int j = 0; j < old_blob.msg_index_size(); j++) {
      update_blob.add_msg_index( update_msg.size() );
      update_msg.push_back( zmsg_vec_[old_blob.msg_index(j)] );
      
      update_blob.add_logical_offset(old_blob.logical_offset(j));
    }

    update_info.push_back(update_blob);
  }

  header_.clear_blobs();
  for (int i = 0; i < update_info.size(); i++) {
    header_.add_blobs()->CopyFrom(update_info[i]);
  }

  zmsg_vec_.clear();
  for (int i = 0; i < update_msg.size(); i++) {
    zmsg_vec_.push_back(update_msg[i]);
  }
 
  return m;
}

void Msg::AddNewBlob(const string& blob_name, const void *data, int sz)
{
  CHECK(!has_blob(blob_name));
  BlobInfo info;
  info.set_blob_name(blob_name);
  info.add_logical_offset(0);

  int msg_index = AppendData(data, sz);
  info.add_msg_index(msg_index);

  add_blob_info(info);
}

void Msg::AppendBlob(const string& blob_name, const void *data, int sz)
{
  if (!has_blob(blob_name)) {
    return AddNewBlob(blob_name, data, sz);
  }

  int blob_idx = blob_index_by_name(blob_name);
  BlobInfo *pblob = header_.mutable_blobs(blob_idx);

  int logical_offset = pblob->logical_offset_size();
  pblob->add_logical_offset(logical_offset);

  int msg_idx = AppendData(data, sz);
  pblob->add_msg_index(msg_idx);
}

void *Msg::AllocBlob(const string& blob_name, int sz)
{
  CHECK(!has_blob(blob_name)) << "allocating an exists blob: " << blob_name;
  
  zmq_msg_t *m = new zmq_msg_t;
  zmq_msg_init_size(m, sz);

  int idx = zmsg_vec_.size();
  zmsg_vec_.push_back(m);
  
  BlobInfo info;
  info.set_blob_name(blob_name);
  info.add_logical_offset(0);
  info.add_msg_index(idx);

  add_blob_info(info);

  return zmq_msg_data(m);
}

void Msg::CopyBlob(const string& blob_name, void *data, int sz)
{
  int blob_index = blob_index_by_name(blob_name);

  CHECK_GE(blob_index, 0) << "Cannot find blob: " << blob_name;
  const BlobInfo& bi = blob_info(blob_index);
  
  vector<int> msgs;
  msgs.resize(bi.msg_index_size());
  int msg_data_size = 0;
  //sequetialize the messages
  for (int i = 0; i < bi.msg_index_size(); i++) {
    msgs[bi.logical_offset(i)] = bi.msg_index(i);
    msg_data_size += ZmsgSize(bi.msg_index(i));
  }

  CHECK_EQ(msg_data_size, sz);
  
  char *blob_data = (char *) data;
  for (int i = 0; i < msgs.size(); i++) {
    memcpy(blob_data, ZmsgData(msgs[i]), ZmsgSize(msgs[i]));
    blob_data += ZmsgSize(msgs[i]);
  }

}

} //end caffe

