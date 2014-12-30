#include <string>

#include "caffe/datum_DB.hpp"

namespace caffe {

shared_ptr<DatumDB> DatumDB::GetDatumDB(const DatumDBParameter& param) {
  shared_ptr<DatumDB> datumdb;
  const string& backend = param.backend();
  CHECK(backend == "leveldb" || backend == "lmdb"
    || backend == "imagesdb") << "Unknown backend " << backend;
  if (backend == "leveldb") {
    datumdb.reset(new DatumLevelDB(param));
  }
  if (backend == "lmdb") {
    datumdb.reset(new DatumLMDB(param));
  }
  if (backend == "imagesdb") {
    datumdb.reset(new DatumImagesDB(param));
  }
  CHECK_NOTNULL(datumdb.get());
  datumdb->Open();
  return datumdb;
}

shared_ptr<DatumDB::Generator> DatumDB::GetGenerator(const DatumDBParameter& param) {
 shared_ptr<DatumDB> datumdb = GetDatumDB(param);
 return datumdb->GetGenerator();
}

}  // namespace caffe
