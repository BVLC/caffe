/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef CAFFE_DATA_READER_HPP_
#define CAFFE_DATA_READER_HPP_

#include <map>
#include <string>
#include <utility>
#include <vector>

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
class DataReader {
 public:
  explicit DataReader(const LayerParameter& param);
  ~DataReader();

  inline BlockingQueue<std::string*>& free() const {
    return queue_pair_->free_;
  }
  inline BlockingQueue<std::string*>& full() const {
    return queue_pair_->full_;
  }

 protected:
  // Queue pairs are shared between a body and its readers
  class QueuePair {
   public:
    explicit QueuePair(int size);
    ~QueuePair();

    BlockingQueue<std::string*> free_;
    BlockingQueue<std::string*> full_;

  DISABLE_COPY_AND_ASSIGN(QueuePair);
  };

  class DBWrapper  {
   public:
    explicit DBWrapper(const LayerParameter& param);
    virtual ~DBWrapper();
    virtual string value() = 0;
    virtual void Next() = 0;
   protected:
    shared_ptr<db::DB> db;
    shared_ptr<db::Cursor> cursor;
  };

  class DBShuffle: public DBWrapper {
   public:
    explicit DBShuffle(const LayerParameter& param);
    virtual string value() {
      return string(static_cast<const char*>(current_image_->first),
                                                      current_image_->second);
    }
    virtual void Next();
   protected:
    vector<std::pair<void*, int> > image_pointers_;
    vector<std::pair<void*, int> >::iterator current_image_;
    shared_ptr<Caffe::RNG> prefetch_rng_;

    void ShuffleImages();
  };

  class DBSequential: public DBWrapper {
   public:
    explicit DBSequential(const LayerParameter& param): DBWrapper(param)  {}
    virtual string value()  { return cursor->value(); }
    virtual void Next();
  };

  // A single body is created per source
  class Body : public InternalThread {
   public:
    explicit Body(const LayerParameter& param);
    virtual ~Body();

   protected:
    void InternalThreadEntry();
    void read_one(DBWrapper* img, QueuePair* qp);
    void ShuffleImages();

    const LayerParameter param_;
    BlockingQueue<shared_ptr<QueuePair> > new_queue_pairs_;
    bool first_read_;

    friend class DataReader;

  DISABLE_COPY_AND_ASSIGN(Body);
  };

  // A source is uniquely identified by its layer name + path, in case
  // the same database is read from two different locations in the net.
  static inline string source_key(const LayerParameter& param) {
    return param.name() + ":" + param.data_param().source();
  }

  const shared_ptr<QueuePair> queue_pair_;
  shared_ptr<Body> body_;

  map<const string, boost::weak_ptr<DataReader::Body> > bodies_;

DISABLE_COPY_AND_ASSIGN(DataReader);
};

}  // namespace caffe

#endif  // CAFFE_DATA_READER_HPP_
