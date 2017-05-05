#include <boost/thread.hpp>
#include <map>
#include <string>
#include <vector>
#include <iostream>

#include "caffe/common.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/layers/annotated_data_layer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

using boost::weak_ptr;

template<>
map<const string, weak_ptr<DataReader<Datum>::Body> > DataReader<Datum>::bodies_ = {};
template<>
map<const string, weak_ptr<DataReader<SparseDatum>::Body> > DataReader<SparseDatum>::bodies_ = {};
template<>
map<const string, weak_ptr<DataReader<AnnotatedDatum>::Body> > DataReader<AnnotatedDatum>::bodies_ = {};  
static boost::mutex bodies_mutex_;
template<>
std::hash<std::thread::id> DataReader<Datum>::idhasher_ = std::hash<std::thread::id>();
template<>
std::hash<std::thread::id> DataReader<SparseDatum>::idhasher_ = std::hash<std::thread::id>();
template<>
std::hash<std::thread::id> DataReader<AnnotatedDatum>::idhasher_ = std::hash<std::thread::id>();
template<class TDatum>
DataReader<TDatum>::DataReader(const LayerParameter& param)
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

template<class TDatum>
DataReader<TDatum>::~DataReader() {
  string key = source_key(body_->param_);
  body_.reset();
  boost::mutex::scoped_lock lock(bodies_mutex_);
  if (bodies_[key].expired()) {
    bodies_.erase(key);
  }
}

//
template<class TDatum>
DataReader<TDatum>::QueuePair::QueuePair(int size) {
  // Initialize the free queue with requested number of datums
  for (int i = 0; i < size; ++i) {
    free_.push(new TDatum());
  }
}

template<class TDatum>
DataReader<TDatum>::QueuePair::~QueuePair() {
  TDatum* datum;
  while (free_.try_pop(&datum)) {
    delete datum;
  }
  while (full_.try_pop(&datum)) {
    delete datum;
  }
}

//
template<class TDatum>
DataReader<TDatum>::Body::Body(const LayerParameter& param)
    : param_(param),
      new_queue_pairs_() {
  StartInternalThread();
}


template<class TDatum>
DataReader<TDatum>::Body::~Body() {
  StopInternalThread();
}

template<class TDatum>
void DataReader<TDatum>::Body::InternalThreadEntry() {
  shared_ptr<db::DB> db(db::GetDB(param_.data_param().backend()));
  //std::cerr << "\nopening db\n";
  db->Open(param_.data_param().source(), db::READ);
  shared_ptr<db::Cursor> cursor(db->NewCursor());
  vector<shared_ptr<QueuePair> > qps;
  try {
    int solver_count = param_.phase() == TRAIN ? Caffe::solver_count() : 1;

    //std::cerr << "\nsolver_count=" << solver_count << std::endl;

    // To ensure deterministic runs, only start running once all solvers
    // are ready. But solvers need to peek on one item during initialization,
    // so read one item, then wait for the next solver.
    for (int i = 0; i < solver_count; ++i) {
      shared_ptr<QueuePair> qp(new_queue_pairs_.pop());
      //std::cerr << "\nreading one\n";
      read_one(cursor.get(), qp.get());
      qps.push_back(qp);
    }
    // Main loop
    while (!must_stop()) {
      for (int i = 0; i < solver_count; ++i) {
        read_one(cursor.get(), qps[i].get());
      }
      // Check no additional readers have been created. This can happen if
      // more than one net is trained at a time per process, whether single
      // or multi solver. It might also happen if two data layers have same
      // name and same source.
      CHECK_EQ(new_queue_pairs_.size(), 0);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}

template<class TDatum>
void DataReader<TDatum>::Body::read_one(db::Cursor* cursor, QueuePair* qp) {
  TDatum* datum = qp->free_.pop();
  // TODO deserialize in-place instead of copy?
  datum->ParseFromString(cursor->value());
  qp->full_.push(datum);

  // go to the next iter
  cursor->Next();
  if (!cursor->valid()) {
    DLOG(INFO) << "Restarting data prefetching from start.";
    cursor->SeekToFirst();
  }
}

// Instance classes
template class DataReader<Datum>;
template class DataReader<SparseDatum>;
template class DataReader<AnnotatedDatum>;

}  // namespace caffe
