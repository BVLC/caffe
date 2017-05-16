

#ifndef MULTI_NODE_ASYNC_READER_H_
#define MULTI_NODE_ASYNC_READER_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"

#include "boost/thread/once.hpp"
#include "boost/thread/thread.hpp"
#include "boost/thread/mutex.hpp"


namespace caffe {

class ReaderThread : public InternalThread 
{
public:
  explicit ReaderThread(const LayerParameter& param, BlockingQueue<Datum*> *free, BlockingQueue<Datum*> * full) 
    : param_(param)
  {
    free_ = free;
    full_ = full;

    StartInternalThread();
  }

  virtual ~ReaderThread() {
    StopInternalThread();
  }

protected:
  void InternalThreadEntry();
  
protected:
  LayerParameter param_;
  BlockingQueue<Datum*> *free_;
  BlockingQueue<Datum*> * full_;

DISABLE_COPY_AND_ASSIGN(ReaderThread);
};

class AsyncReader {

public:
  virtual ~AsyncReader() {
    Datum* datum = NULL;
    
    for (int i = 0; i < free_queue_.size(); i++) {
      while (free_queue_[i]->try_pop(&datum)) {
        delete datum;
      }
    }

    for (int i = 0; i < full_queue_.size(); i++) {
      while (full_queue_[i]->try_pop(&datum)) {
        delete datum;
      }
    }
  }

  static AsyncReader* Instance()
  {
    boost::call_once(flag_once_, InitReader);
    
    return instance_;
  }


public:
  int RegisterLayer(const LayerParameter& param);
  BlockingQueue<Datum*> *GetFree(int i) { return free_queue_[i]; }
  BlockingQueue<Datum*> *GetFull(int i) { return full_queue_[i]; }

protected:
  int AddBlockQueue(const LayerParameter& param);

protected:
  static void InitReader()
  {
    instance_ = new AsyncReader();
  }
  
  static inline string source_key(const LayerParameter& param) 
  {
    return param.name() + ":" + param.data_param().source();
  }

protected:
  vector<BlockingQueue<Datum*> *> free_queue_;
  vector<BlockingQueue<Datum*> *> full_queue_;
  
  //for the reader threads
  vector<shared_ptr<ReaderThread> > thread_arr_;

  map<const string, int> index_map_;
  boost::mutex register_mutex_;

protected:
  static AsyncReader *instance_;
  static boost::once_flag flag_once_;


private:
  AsyncReader() {}

DISABLE_COPY_AND_ASSIGN(AsyncReader);

};


} //namespace caffe


#endif


