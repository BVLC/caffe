#ifndef CAFFE_DATA_READER_HPP_
#define CAFFE_DATA_READER_HPP_

//#include <boost/thread.hpp>
#include <map>
#include <string>
#include <vector>
#include <thread>

#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"

namespace caffe {

/**
 * @brief Reads data from a source to queues available to data layers.
 * A single reading thread is created per source, even if multiple solvers
 * are running in parallel, e.g. for multi-GPU training. This makes sure
 * databases are read sequentially, and that each solver accesses a different
 * subset of the database. Data is distributed to solvers in a round-robin
 * way to keep parallel training deterministic.
 */
template<class TDatum>
class DataReader {
 public:
  explicit DataReader(const LayerParameter& param);
  ~DataReader();

  inline BlockingQueue<TDatum*>& free() const {
    return queue_pair_->free_;
  }
  inline BlockingQueue<TDatum*>& full() const {
    return queue_pair_->full_;
  }

 protected:
  // Queue pairs are shared between a body and its readers
  class QueuePair {
   public:
    explicit QueuePair(int size);
    ~QueuePair();

    BlockingQueue<TDatum*> free_;
    BlockingQueue<TDatum*> full_;

  DISABLE_COPY_AND_ASSIGN(QueuePair);
  };

  // A single body is created per source
  class Body : public InternalThread {
   public:
    explicit Body(const LayerParameter& param);
    virtual ~Body();

   protected:
    void InternalThreadEntry();
    void read_one(db::Cursor* cursor, QueuePair* qp);

    const LayerParameter param_;
    BlockingQueue<shared_ptr<QueuePair> > new_queue_pairs_;

    friend class DataReader<TDatum>;

  DISABLE_COPY_AND_ASSIGN(Body);
  };

  // A source is uniquely identified by its layer name + path, in case
  // the same database is read from two different locations in the net.
  // DD: supplemented with thread id, as this was breaking learning the same
  // model several times in parallel with the same instance of the lib.
  static inline string source_key(const LayerParameter& param) {
    int tid = idhasher_(std::this_thread::get_id());
    return std::to_string(tid) + ":" + param.name() + ":" + param.data_param().source();
  }

  const shared_ptr<QueuePair> queue_pair_;
  shared_ptr<Body> body_;

  static map<const string, boost::weak_ptr<DataReader<TDatum>::Body> > bodies_;

  static std::hash<std::thread::id> idhasher_;

  //boost::mutex bodies_mutex_;
DISABLE_COPY_AND_ASSIGN(DataReader);
};

}  // namespace caffe

#endif  // CAFFE_DATA_READER_HPP_
