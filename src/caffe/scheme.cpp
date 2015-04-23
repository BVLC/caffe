#include <boost/thread.hpp>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <opencv2/core/core.hpp>

#include <fcntl.h>
#include <stdint.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

using boost::weak_ptr;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::CodedInputStream;

map<const string, shared_ptr<Scheme> > Scheme::schemes_;
map<const string, weak_ptr<Reader::Body> > Scheme::readers_;
static boost::mutex scheme_mutex_;

// Load default schemes
static DefaultSchemes schemes_;

void Scheme::add(const shared_ptr<Scheme>& scheme) {
  boost::mutex::scoped_lock lock(scheme_mutex_);
  for (int i = 0; i < scheme->names_.size(); ++i) {
    schemes_[scheme->names_[i]] = scheme;
  }
}

const shared_ptr<Scheme>& Scheme::get(const string& name) {
  boost::mutex::scoped_lock lock(scheme_mutex_);
  const shared_ptr<Scheme>& instance = schemes_[name];
  CHECK(instance) << "Unknown URI scheme: " << name;
  return instance;
}

shared_ptr<Reader> Scheme::get_reader(const DataParameter& param,
                                      int index) const {
  boost::mutex::scoped_lock lock(scheme_mutex_);
  const string& source = param.source(index);
  weak_ptr<Reader::Body> weak = readers_[source];
  shared_ptr<Reader::Body> shared = weak.lock();
  if (!shared) {
    shared.reset(new_reader(param, index));
    readers_[source] = weak_ptr<Reader::Body>(shared);
  }
  return shared_ptr<Reader>(new Reader(shared));
}

Reader::~Reader() {
  boost::mutex::scoped_lock lock(scheme_mutex_);
  string source = body_->param_.source(body_->index_);
  body_.reset();
  if (Scheme::readers_[source].expired())
    Scheme::readers_.erase(source);
}

//

Reader::Body::Body(const DataParameter& param, int index)
    : param_(param),
      index_(index),
      free_(),
      full_() {
  // Add prefetch datums to layer free queue
  int prefetch = param.prefetch() * param.batch_size();
  for (int i = 0; i < prefetch; ++i) {
    free_.push(new Datum());
  }
}

Reader::Body::~Body() {
  StopInternalThread();
  Datum* datum;
  while (free_.try_pop(&datum)) {
    delete datum;
  }
  while (full_.try_pop(&datum)) {
    delete datum;
  }
}

//

#ifndef NO_IO_DEPENDENCIES

DefaultDatabases::DBReader::DBReader(const DataParameter& param, int index)
    : Reader::Body(param, index) {
  StartInternalThread();
}

void DefaultDatabases::DBReader::InternalThreadEntry() {
  URI uri(param_.source(index_), true);
  DataParameter_DB backend =
      uri.scheme() == "lmdb" ? DataParameter::LMDB : DataParameter::LEVELDB;
  shared_ptr<db::DB> db(db::GetDB(backend));
  db->Open(uri.path(), db::READ);
  shared_ptr<db::Cursor> cursor(db->NewCursor());
  try {
    while (!must_stop()) {
      Datum* datum = free_.pop();
      // TODO deserialize in-place instead of copy?
      datum->ParseFromString(cursor->value());
      full_.push(datum);

      // go to the next iter
      cursor->Next();
      if (!cursor->valid()) {
        DLOG(INFO) << "Restarting data prefetching from start.";
        cursor->SeekToFirst();
      }
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}

#endif

//

void FileScheme::DescriptorReader::read_descriptor(int file_descriptor) {
  FileInputStream input(file_descriptor);
  while (!must_stop()) {
    Datum* datum = free_.pop();
    CodedInputStream coded(&input);
    uint32_t length;
    CHECK(coded.ReadVarint32(&length));
    if (!length) {
      break;
    }
    coded.PushLimit(length);
    CHECK(datum->ParseFromCodedStream(&coded));
    CHECK((datum->data().size() > 0 || datum->float_data().size() > 0) &&
        datum->has_label()) << "Received invalid datum";
    full_.push(datum);
  }
}

FileScheme::FileReader::FileReader(const DataParameter& param, int index)
    : DescriptorReader(param, index) {
  StartInternalThread();
}

void FileScheme::FileReader::InternalThreadEntry() {
  try {
    while (!must_stop()) {
      URI uri(param_.source(index_), true);
      File file(uri.path(), O_RDONLY);
      DLOG(INFO) << "Opened file " << uri.path();
      read_descriptor(file.descriptor());
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}

//

SocketScheme::SocketReader::SocketReader(const DataParameter& param, int index)
    : DescriptorReader(param, index) {
  StartInternalThread();
}

void SocketScheme::SocketReader::InternalThreadEntry() {
  try {
    while (!must_stop()) {
      URI uri(param_.source(index_));
      Socket socket(uri);
      DLOG(INFO) << "Connected to " << uri.host() << ":" << uri.port();
      if (uri.scheme() == "http") {
        string get = "GET " + uri.path() + " HTTP/1.1\r\n\r\n";
        size_t len = get.size();
        CHECK_EQ(write(socket.descriptor(), get.c_str(), len), len);
        // Skip headers
        for (;;) {
          int line = 0;
          char c = 0;
          while (c != '\n') {
            CHECK_EQ(read(socket.descriptor(), &c, 1), 1);
            line++;
          }
          if (line == 2)  // Break if line is /r/n
            break;
        }
      }
      read_descriptor(socket.descriptor());
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}

}  // namespace caffe
