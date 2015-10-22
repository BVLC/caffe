#include <boost/thread.hpp>
#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

using boost::weak_ptr;

map<const string, weak_ptr<DataReader::Body> > DataReader::bodies_;
static boost::mutex bodies_mutex_;

DataReader::DataReader(const LayerParameter& param)
    : queue_pair_(new QueuePair(  //
        param.data_param().prefetch() * param.data_param().batch_size())) {
  // Get or create a body
  boost::mutex::scoped_lock lock(bodies_mutex_);
  string key = source_key(param);
  weak_ptr<Body>& weak = bodies_[key];
  body_ = weak.lock();
  if (!body_) {
    body_.reset(new Body(param));
    bodies_[key] = weak_ptr<Body>(body_);
  }
  body_->new_queue_pairs_.push(queue_pair_);
}

DataReader::~DataReader() {
  string key = source_key(body_->param_);
  body_.reset();
  boost::mutex::scoped_lock lock(bodies_mutex_);
  if (bodies_[key].expired()) {
    bodies_.erase(key);
  }
}

//

DataReader::QueuePair::QueuePair(int size) {
  // Initialize the free queue with requested number of datums
  for (int i = 0; i < size; ++i) {
    free_.push(new Datum());
  }
}

DataReader::QueuePair::~QueuePair() {
  Datum* datum;
  while (free_.try_pop(&datum)) {
    delete datum;
  }
  while (full_.try_pop(&datum)) {
    delete datum;
  }
}

//

DataReader::Body::Body(const LayerParameter& param)
    : param_(param),
      read(0),
      new_queue_pairs_() {
  StartInternalThread();
}

DataReader::Body::~Body() {
  StopInternalThread();
}

void DataReader::Body::InternalThreadEntry() {
  shared_ptr<db::DB> db(db::GetDB(param_.data_param().backend()));
  db->Open(param_.data_param().source(), db::READ);
  shared_ptr<db::Cursor> cursor(db->NewCursor());
  vector<shared_ptr<QueuePair> > qps;
  try {
    // Main loop
    while (!must_stop()) {
      // To ensure deterministic runs, only start running once all solvers
      // are ready. But solvers need to peek on one item during initialization,
      // so read one item, then wait for the next solver.
      while (new_queue_pairs_.size()) {
        shared_ptr<QueuePair> qp(new_queue_pairs_.pop());
        qps.push_back(qp);
      }

      for (int i = 0; i < qps.size(); ++i) {
        read_one(cursor.get(), qps[i].get());
      }
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}

void DataReader::Body::read_one(db::Cursor* cursor, QueuePair* qp) {
  if (qp->free_.size() == 0) {
    return;
  }
  Datum* datum = qp->free_.pop();
  // TODO deserialize in-place instead of copy?
  datum->ParseFromString(cursor->value());
  qp->full_.push(datum);
  ++read;

  // go to the next iter
  cursor->Next();
  if (!cursor->valid()) {
    LOG(INFO) << "Restarting data prefetching from start, read: " << read;
    read = 0;
    cursor->SeekToFirst();
  }
}

}  // namespace caffe
