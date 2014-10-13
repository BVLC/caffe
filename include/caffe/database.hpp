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

  struct KV {
    buffer_t key;
    buffer_t value;
  };

  virtual bool open(const string& filename, Mode mode) = 0;
  virtual bool put(buffer_t* key, buffer_t* value) = 0;
  virtual bool get(buffer_t* key, buffer_t* value) = 0;
  virtual bool commit() = 0;
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
  class iterator : public std::iterator<std::forward_iterator_tag, KV> {
   public:
    typedef KV T;
    typedef T value_type;
    typedef T& reference_type;
    typedef T* pointer_type;

    iterator()
        : parent_(NULL) { }

    iterator(const Database* parent, shared_ptr<DatabaseState> state)
        : parent_(parent),
          state_(state) { }
    ~iterator() { }

    iterator(const iterator& other)
        : parent_(other.parent_),
          state_(other.state_->clone()) { }

    iterator& operator=(iterator copy) {
      copy.swap(*this);
      return *this;
    }

    void swap(iterator& other) throw() {
      std::swap(this->parent_, other.parent_);
      std::swap(this->state_, other.state_);
    }

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
    virtual shared_ptr<DatabaseState> clone() = 0;
  };

  virtual bool equal(shared_ptr<DatabaseState> state1,
      shared_ptr<DatabaseState> state2) const = 0;
  virtual void increment(shared_ptr<DatabaseState> state) const = 0;
  virtual KV& dereference(
      shared_ptr<DatabaseState> state) const = 0;
};

}  // namespace caffe

#endif  // CAFFE_DATABASE_H_
