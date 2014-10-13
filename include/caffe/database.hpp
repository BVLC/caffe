#ifndef CAFFE_DATABASE_H_
#define CAFFE_DATABASE_H_

#include <algorithm>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "caffe/common.hpp"

namespace caffe {

namespace database_internal {

template <typename T>
struct Coder {
  static bool serialize(const T& obj, string* serialized) {
    return obj.SerializeToString(serialized);
  }

  static bool serialize(const T& obj, vector<char>* serialized) {
    serialized->resize(obj.ByteSize());
    return obj.SerializeWithCachedSizesToArray(
        reinterpret_cast<unsigned char*>(serialized->data()));
  }

  static bool deserialize(const string& serialized, T* obj) {
    return obj->ParseFromString(serialized);
  }

  static bool deserialize(const char* data, size_t size, T* obj) {
    return obj->ParseFromArray(data, size);
  }
};

template <>
struct Coder<string> {
  static bool serialize(string obj, string* serialized) {
    *serialized = obj;
    return true;
  }

  static bool serialize(const string& obj, vector<char>* serialized) {
    vector<char> temp(obj.data(), obj.data() + obj.size());
    serialized->swap(temp);
    return true;
  }

  static bool deserialize(const string& serialized, string* obj) {
    *obj = serialized;
    return true;
  }

  static bool deserialize(const char* data, size_t size, string* obj) {
    string temp_string(data, size);
    obj->swap(temp_string);
    return true;
  }
};

template <>
struct Coder<vector<char> > {
  static bool serialize(vector<char> obj, string* serialized) {
    string tmp(obj.data(), obj.size());
    serialized->swap(tmp);
    return true;
  }

  static bool serialize(const vector<char>& obj, vector<char>* serialized) {
    *serialized = obj;
    return true;
  }

  static bool deserialize(const string& serialized, vector<char>* obj) {
    vector<char> tmp(serialized.data(), serialized.data() + serialized.size());
    obj->swap(tmp);
    return true;
  }

  static bool deserialize(const char* data, size_t size, vector<char>* obj) {
    vector<char> tmp(data, data + size);
    obj->swap(tmp);
    return true;
  }
};

}  // namespace database_internal

template <typename K, typename V>
class Database {
 public:
  enum Mode {
    New,
    ReadWrite,
    ReadOnly
  };

  typedef K key_type;
  typedef V value_type;

  struct KV {
    K key;
    V value;
  };

  virtual bool open(const string& filename, Mode mode) = 0;
  virtual bool put(const K& key, const V& value) = 0;
  virtual bool get(const K& key, V* value) = 0;
  virtual bool commit() = 0;
  virtual void close() = 0;

  virtual void keys(vector<K>* keys) = 0;

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
          state_(other.state_ ? other.state_->clone()
              : shared_ptr<DatabaseState>()) { }

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
      parent_->increment(&state_);
      return *this;
    }
    iterator operator++(int) {
      iterator copy(*this);
      parent_->increment(&state_);
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
  virtual void increment(shared_ptr<DatabaseState>* state) const = 0;
  virtual KV& dereference(
      shared_ptr<DatabaseState> state) const = 0;

  template <typename T>
  static bool serialize(const T& obj, string* serialized) {
    return database_internal::Coder<T>::serialize(obj, serialized);
  }

  template <typename T>
  static bool serialize(const T& obj, vector<char>* serialized) {
    return database_internal::Coder<T>::serialize(obj, serialized);
  }

  template <typename T>
  static bool deserialize(const string& serialized, T* obj) {
    return database_internal::Coder<T>::deserialize(serialized, obj);
  }

  template <typename T>
  static bool deserialize(const char* data, size_t size, T* obj) {
    return database_internal::Coder<T>::deserialize(data, size, obj);
  }
};

}  // namespace caffe

#define INSTANTIATE_DATABASE(type) \
  template class type<string, string>; \
  template class type<string, vector<char> >; \
  template class type<string, Datum>;

#endif  // CAFFE_DATABASE_H_
