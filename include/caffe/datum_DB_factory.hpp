#ifndef CAFFE_DATUMDB_FACTORY_H_
#define CAFFE_DATUMDB_FACTORY_H_

#include <map>
#include <string>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

class DatumDB;

class DatumDBRegistry {
 public:
  typedef DatumDB* (*Creator)(const DatumDBParameter&);
  typedef std::map<string, Creator> CreatorRegistry;
  typedef std::map<string, DatumDB* > SourceRegistry;

  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  static SourceRegistry& Sources() {
    static SourceRegistry* s_registry_ = new SourceRegistry();
    return *s_registry_;
  }

  // Adds a creator.
  static void AddCreator(const string& backend,
                         Creator creator) {
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(backend), 0)
        << "DatumDB backend " << backend << " already registered.";
    registry[backend] = creator;
  }

  // Get a DatumDB using a DatumDBParameter.
  static DatumDB* GetDatumDB(const DatumDBParameter& param) {
    SourceRegistry& sources = Sources();
    SourceRegistry::iterator it = sources.find(param.source());
    if (it != sources.end()) {
      LOG(INFO) << "Reusing DatumDB " << param.source();
      return (*it).second;
    } else {
      LOG(INFO) << "Creating DatumDB " << param.source();
      const string& backend = param.backend();
      CreatorRegistry& registry = Registry();
      CHECK_EQ(registry.count(backend), 1);
      DatumDB* datumdb = registry[backend](param);
      if (param.unique_source()) {
        sources[param.source()] = datumdb;
      }
      return datumdb;
    }
  }

  static bool RemoveSource(const string& source) {
    SourceRegistry& sources = Sources();
    SourceRegistry::iterator it = sources.find(source);
    if (it != sources.end()) {
      LOG(INFO) << "Removing Source " << source;
      sources.erase(it);
      return true;
    }
    return false;
  }

 private:
  // Layer registry should never be instantiated - everything is done with its
  // static variables.
  DatumDBRegistry() {}
};

class DatumDBRegisterer {
 public:
  DatumDBRegisterer(const string& backend,
                  DatumDB* (*creator)(const DatumDBParameter&)) {
    LOG(INFO) << "Registering DatumDB backend: " << backend;
    DatumDBRegistry::AddCreator(backend, creator);
  }
};


#define REGISTER_DATUMDB_CREATOR(backend, creator)                     \
  static DatumDBRegisterer g_datumdb_##creator(backend, creator);

#define REGISTER_DATUMDB_CLASS(backend, clsname)                       \
  DatumDB* Creator_##clsname(const DatumDBParameter& param) {          \
    return new clsname(param);                                         \
  }                                                                    \
  REGISTER_DATUMDB_CREATOR(backend, Creator_##clsname)


}  // namespace caffe

#endif  // CAFFE_DATUMDB_FACTORY_H_
