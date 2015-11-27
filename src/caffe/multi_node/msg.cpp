

#include "caffe/multi_node/msg.hpp"

namespace caffe {

int Msg::MergeMsg(shared_ptr<Msg> m)
{
  CHECK_EQ(m->msg_id(), header_.msg_id());
  
  for (int i = 0; i < m->num_blobs(); i++) {
    const BlobInfo &src_blob = m->blob_info(i);

    int blob_index = blob_index_by_name(src_blob.blob_name());
    
    if (blob_index >= 0) {   //atatching the blobs to the packet
      BlobInfo *tgt_blob = header_.mutable_blobs(blob_index); 
      
      for (int k = 0; k < src_blob.fragment_offset_size(); k++) {
        tgt_blob->add_fragment_offset(src_blob.fragment_offset(k));
        
        int msg_index = AppendZmsg(m->GetZmsg(src_blob.msg_index(k)));
        tgt_blob->add_msg_index(msg_index);
      }

    } else {    //add a new blob
      //add message data
      BlobInfo new_blob(src_blob);
      
      for (int k = 0; k < src_blob.fragment_offset_size(); k++) {
        new_blob.set_msg_index(k, AppendZmsg(m->GetZmsg(src_blob.msg_index(k))));
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
  new_blob.set_num_fragments(chosen_blob.num_fragments());

  for (int i = 0; i < chosen_blob.msg_index_size(); i++) {
    int msg_index = chosen_blob.msg_index(i);
    zmq_msg_t *pmsg = zmsg_vec_[msg_index];
    zmsg_vec_[msg_index] = NULL;

    new_blob.add_msg_index(m->AppendZmsg(pmsg));
    new_blob.add_fragment_offset(chosen_blob.fragment_offset(i)); 
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
    update_blob.set_num_fragments(old_blob.num_fragments());

    for (int j = 0; j < old_blob.msg_index_size(); j++) {
      update_blob.add_msg_index( update_msg.size() );
      update_msg.push_back( zmsg_vec_[old_blob.msg_index(j)] );
      
      update_blob.add_fragment_offset(old_blob.fragment_offset(j));
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
  BlobInfo info;
  info.set_blob_name(blob_name);
  info.set_num_fragments(1);
  info.add_fragment_offset(0);

  int msg_index = AppendData(data, sz);
  info.add_msg_index(msg_index);

  add_blob_info(info);
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
    msgs[bi.fragment_offset(i)] = bi.msg_index(i);
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

