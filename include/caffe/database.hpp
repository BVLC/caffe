#ifndef CAFFE_DATABASE_H_
#define CAFFE_DATABASE_H_

#include <algorithm>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "caffe/common.hpp"

namespace caffe {

class Database {
 public:
  enum Mode {
    New,
    ReadWrite,
    ReadOnly
  };

  typedef vector<char> buffer_t;

  virtual void open(const string& filename, Mode mode) = 0;
  virtual void put(buffer_t* key, buffer_t* value) = 0;
  virtual void commit() = 0;
  virtual void close() = 0;

  Database() { }
  virtual ~Database() { }

  class iterator;
  typedef iterator const_iterator;

  virtual const_iterator begin() const = 0;
  virtual const_iterator cbegin() const = 0;
  virtual const_iterator end() const = 0;
  virtual const_iterator cend() const = 0;

 protected:
  class DatabaseState;

 public:
  class iterator : public std::iterator<
      std::forward_iterator_tag, pair<buffer_t, buffer_t> > {
   public:
    typedef pair<buffer_t, buffer_t> T;
    typedef T value_type;
    typedef T& reference_type;
    typedef T* pointer_type;

    iterator()
        : parent_(NULL) { }

    iterator(const Database* parent, shared_ptr<DatabaseState> state)
        : parent_(parent),
          state_(state) { }
    ~iterator() { }

    bool operator==(const iterator& other) const {
      return parent_->equal(state_, other.state_);
    }

    bool operator!=(const iterator& other) const {
      return !(*this == other);
    }

    iterator& operator++() {
      parent_->increment(state_);
      return *this;
    }
    iterator operator++(int) {
      iterator copy(*this);
      parent_->increment(state_);
      return copy;
    }

    reference_type operator*() const {
      return parent_->dereference(state_);
    }

    pointer_type operator->() const {
      return &parent_->dereference(state_);
    }

   protected:
    const Database* parent_;
    shared_ptr<DatabaseState> state_;
  };

 protected:
  class DatabaseState {
   public:
    virtual ~DatabaseState() { }
  };

  virtual bool equal(shared_ptr<DatabaseState> state1,
      shared_ptr<DatabaseState> state2) const = 0;
  virtual void increment(shared_ptr<DatabaseState> state) const = 0;
  virtual pair<buffer_t, buffer_t>& dereference(
      shared_ptr<DatabaseState> state) const = 0;
};

}  // namespace caffe

#endif  // CAFFE_DATABASE_H_
