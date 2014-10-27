#include "caffe/datum_DB.hpp"

namespace caffe {

shared_ptr<DatumDB> DatumDB::GetDatumDB(const DatumDBParameter& param) {
  shared_ptr<DatumDB> datumdb;
  switch (param.backend()) {
  case DatumDBParameter_Backend_LEVELDB:
    datumdb.reset(new DatumLevelDB(param));
    break;
  case DatumDBParameter_Backend_LMDB:
    datumdb.reset(new DatumLMDB(param));
    break;
  case DatumDBParameter_Backend_IMAGESDB:
    datumdb.reset(new DatumImagesDB(param));
    break;
  default:
    LOG(FATAL) << "DatumDB has unknown backend " << param.backend();
  }
  return datumdb;
}

DatumDBParameter_Backend DatumDB::GetBackend(const string& backend) {
  CHECK(backend == "leveldb" || backend == "lmdb"
    || backend == "imagesdb") << "Unknown backend " << backend;
  DatumDBParameter_Backend enum_backend;
  if (backend == "leveldb") {
    enum_backend = DatumDBParameter_Backend_LEVELDB;
  }
  if (backend == "lmdb") {
    enum_backend = DatumDBParameter_Backend_LMDB;
  }
  if (backend == "imagesdb") {
    enum_backend = DatumDBParameter_Backend_IMAGESDB;
  }
  return enum_backend;
}

}  // namespace caffe