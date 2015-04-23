#ifndef CAFFE_SCHEMES_HPP_
#define CAFFE_SCHEMES_HPP_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

using boost::weak_ptr;

// A URI scheme adds support for an input/output transport.
// E.g. lmdb://path, tcp://host:port, hdfs://path
class Scheme;

// Reads datums to a queue available to data layers.
class Reader {
 public:
  ~Reader();

  inline blocking_queue<Datum*>& free() {
    return body_->free_;
  }
  inline blocking_queue<Datum*>& full() {
    return body_->full_;
  }

  // Makes sure only one reader is created per source, needed in
  // particular for parallel training
  class Body : public InternalThread {
   public:
    Body(const DataParameter& param, int index);
    virtual ~Body();

   protected:
    const DataParameter param_;
    const int index_;
    blocking_queue<Datum*> free_;
    blocking_queue<Datum*> full_;

    friend class Reader;

  DISABLE_COPY_AND_ASSIGN(Body);
  };

 protected:
  explicit Reader(const shared_ptr<Body>& body)
      : body_(body) {
  }

  shared_ptr<Body> body_;

  friend class Scheme;

DISABLE_COPY_AND_ASSIGN(Reader);
};

// TODO Writer for output layers

class Scheme {
 public:
  virtual ~Scheme() {
  }

  inline const vector<string>& names() const {
    return names_;
  }

  static void add(const shared_ptr<Scheme>& scheme);
  static const shared_ptr<Scheme>& get(const string& name);

  shared_ptr<Reader> get_reader(const DataParameter& param, int index) const;

 protected:
  Scheme() {
  }

  virtual Reader::Body* new_reader(const DataParameter& param, int i) const = 0;

  vector<string> names_;

  static map<const string, shared_ptr<Scheme> > schemes_;
  static map<const string, weak_ptr<Reader::Body> > readers_;

  friend class Reader;

DISABLE_COPY_AND_ASSIGN(Scheme);
};

#ifndef NO_IO_DEPENDENCIES

class DefaultDatabases : public Scheme {
 public:
  DefaultDatabases() {
    names_.push_back("lmdb");
    names_.push_back("leveldb");
  }
  virtual ~DefaultDatabases() {
  }

 protected:
  class DBReader : public Reader::Body {
   public:
    DBReader(const DataParameter& param, int index);
    virtual ~DBReader() {
    }

    virtual void InternalThreadEntry();

    using Reader::Body::free_;
    using Reader::Body::full_;
    using Reader::Body::param_;
    using Reader::Body::index_;
  };

  virtual Reader::Body* new_reader(const DataParameter& param, int i) const {
    return new DBReader(param, i);
  }
};

#endif

class FileScheme : public Scheme {
 public:
  FileScheme() {
    names_.push_back("file");
  }
  virtual ~FileScheme() {
  }

 protected:
  class DescriptorReader : public Reader::Body {
   public:
    DescriptorReader(const DataParameter& param, int index)
        : Reader::Body(param, index) {
    }
    virtual ~DescriptorReader() {
    }

    void read_descriptor(int file_descriptor);

    using Reader::Body::free_;
    using Reader::Body::full_;
    using Reader::Body::param_;
    using Reader::Body::index_;
  };

  class FileReader : public DescriptorReader {
   public:
    FileReader(const DataParameter& param, int index);
    virtual ~FileReader() {
    }

    virtual void InternalThreadEntry();
  };

  virtual Reader::Body* new_reader(const DataParameter& param, int i) const {
    return new FileReader(param, i);
  }
};

class SocketScheme : public FileScheme {
 public:
  SocketScheme() {
    names_.clear();
    names_.push_back("tcp");
    names_.push_back("http");
  }
  virtual ~SocketScheme() {
  }

 protected:
  class SocketReader : public DescriptorReader {
   public:
    SocketReader(const DataParameter& param, int index);
    virtual ~SocketReader() {
    }

    virtual void InternalThreadEntry();
  };

  virtual Reader::Body* new_reader(const DataParameter& param, int i) const {
    return new SocketReader(param, i);
  }
};

// TODO hdf5, images

// Loads default schemes
class DefaultSchemes {
 public:
  DefaultSchemes() {
#ifndef NO_IO_DEPENDENCIES
    Scheme::add(shared_ptr<Scheme>(new DefaultDatabases()));
#endif
    Scheme::add(shared_ptr<Scheme>(new FileScheme()));
    Scheme::add(shared_ptr<Scheme>(new SocketScheme()));
  }
};

}  // namespace caffe

#endif  // CAFFE_SCHEMES_HPP_
