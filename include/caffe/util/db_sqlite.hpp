#ifndef CAFFE_UTIL_DB_SQLITE_HPP_
#define CAFFE_UTIL_DB_SQLITE_HPP_

#ifdef USE_SQLITE

#include "caffe/common.hpp"
#include "sqlite3.h"

namespace caffe {

class SQLiteHelper {
 public:
  SQLiteHelper(string db_name);
  ~SQLiteHelper();
  void LogDatabaseError();
  void CreateTables();
  void StoreKernel(int64_t hash,
                   const char* flags,
                   size_t flags_size,
                   const char* code,
                   size_t code_size,
                   const char* program,
                   size_t program_size);
  int64_t GetKernelInfo(int64_t hash,
                        const char* flags,
                        size_t flags_size,
                        const char* code,
                        size_t code_size,
                        size_t* program_size);
  bool LoadKernel(int64_t id, char* program);

 private:
  string parse_error_code(const int rc) const;
  bool is_error(const int rc) const;
  sqlite3* db_;
};

}  // namespace caffe

#endif  // USE_SQLITE

#endif  // CAFFE_UTIL_DB_SQLITE_HPP_
