

#include "caffe/multi_node/async_reader.hpp"

namespace caffe {

boost::once_flag AsyncReader::flag_once_;

AsyncReader *AsyncReader::instance_ = NULL;

int AsyncReader::RegisterLayer(const LayerParameter& param)
{
  boost::mutex::scoped_lock lock(register_mutex_);
  
  string key = source_key(param);
  map<const string, int>::iterator iter = index_map_.find(key);
  
  int index = -1;

  if (iter == index_map_.end()) {
    index = AddBlockQueue(param);
  } else {
    index = iter->second;
  }
  
  BlockingQueue<Datum*> *p = free_queue_[index];
  int sz = param.data_param().prefetch() * param.data_param().batch_size();

  //increase the buffer queue
  for (int i = 0; i < sz; i++) {
    p->push(new Datum());
  }

  return index;
}


int AsyncReader::AddBlockQueue(const LayerParameter& param)
{
  BlockingQueue<Datum*> *p = new BlockingQueue<Datum*>();

  free_queue_.push_back(p);
  full_queue_.push_back(new BlockingQueue<Datum*>());

  CHECK_EQ(free_queue_.size(), full_queue_.size());
  
  int index = free_queue_.size() - 1;

  //start a new reader thread
  thread_arr_.push_back(shared_ptr<ReaderThread>(new ReaderThread( param,
                                      free_queue_[index], full_queue_[index] )));

  return index;
}


void ReaderThread::InternalThreadEntry()
{
  shared_ptr<db::DB> db(db::GetDB(param_.data_param().backend()));
  db->Open(param_.data_param().source(), db::READ);
  shared_ptr<db::Cursor> cursor(db->NewCursor());

  while (!must_stop()) {
    Datum* datum = free_->pop();

    datum->ParseFromString(cursor->value());
    full_->push(datum);

    cursor->Next();
    if (!cursor->valid()) {
      LOG(INFO) << "Restarting data prefetching from start.";
      cursor->SeekToFirst();
    }
  }
}

}


