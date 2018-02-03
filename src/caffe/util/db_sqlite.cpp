#ifdef USE_SQLITE
#include "caffe/util/db_sqlite.hpp"
#include "caffe/util/persistent_storage.hpp"

namespace caffe {

SQLiteHelper::SQLiteHelper(string db_name) : db_(nullptr) {
  int rc;
  string db_path = persistent_storage_path() + db_name + ".db";
  rc = sqlite3_open(db_path.c_str(), &db_);
  if (rc || !db_) {
     LOG(ERROR) << "Can't open database: " << db_path
                << " (" << parse_error_code(rc) << ")";
  }
  CHECK(db_);
}

SQLiteHelper::~SQLiteHelper() {
  sqlite3_close(db_);
}

void SQLiteHelper::LogDatabaseError() {
  int errcode = sqlite3_errcode(db_);
  const char* errmsg = sqlite3_errmsg(db_);
  if (errcode) {
    string errstr(errmsg);
    if (errstr.size() <= 0 || errmsg == nullptr) {
      errstr = "";
    }
    LOG(ERROR) << "Database error: " << parse_error_code(errcode)
               << ", " << errstr;
  }
}

bool SQLiteHelper::is_error(const int rc) const {
  return !(rc == SQLITE_OK || rc == SQLITE_ROW || rc == SQLITE_DONE);
}

string SQLiteHelper::parse_error_code(const int rc) const {
  string errcode = "";
  switch(rc) {
    case SQLITE_OK:
      errcode = "SQLITE_OK";
      break;
    case SQLITE_ERROR:
      errcode = "SQLITE_ERROR";
      break;
    case SQLITE_INTERNAL:
      errcode = "SQLITE_INTERNAL";
      break;
    case SQLITE_PERM:
      errcode = "SQLITE_PERM";
      break;
    case SQLITE_ABORT :
      errcode = "SQLITE_ABORT";
      break;
    case SQLITE_BUSY:
      errcode = "SQLITE_BUSY";
      break;
    case SQLITE_LOCKED:
      errcode = "SQLITE_LOCKED";
      break;
    case SQLITE_NOMEM:
      errcode = "SQLITE_NOMEM";
      break;
    case SQLITE_READONLY:
      errcode = "SQLITE_READONLY";
      break;
    case SQLITE_INTERRUPT:
      errcode = "SQLITE_INTERRUPT";
      break;
    case SQLITE_IOERR:
      errcode = "SQLITE_IOERR";
      break;
    case SQLITE_CORRUPT:
      errcode = "SQLITE_CORRUPT";
      break;
    case SQLITE_NOTFOUND:
      errcode = "SQLITE_NOTFOUND";
      break;
    case SQLITE_FULL:
      errcode = "SQLITE_FULL";
      break;
    case SQLITE_CANTOPEN:
      errcode = "SQLITE_CANTOPEN";
      break;
    case SQLITE_PROTOCOL:
      errcode = "SQLITE_PROTOCOL";
      break;
    case SQLITE_EMPTY:
      errcode = "SQLITE_EMPTY";
      break;
    case SQLITE_SCHEMA:
      errcode = "SQLITE_SCHEMA";
      break;
    case SQLITE_TOOBIG:
      errcode = "SQLITE_TOOBIG";
      break;
    case SQLITE_CONSTRAINT:
      errcode = "SQLITE_CONSTRAINT";
      break;
    case SQLITE_MISMATCH:
      errcode = "SQLITE_MISMATCH";
      break;
    case SQLITE_MISUSE:
      errcode = "SQLITE_MISUSE";
      break;
    case SQLITE_NOLFS:
      errcode = "SQLITE_NOLFS";
      break;
    case SQLITE_AUTH:
      errcode = "SQLITE_AUTH";
      break;
    case SQLITE_FORMAT:
      errcode = "SQLITE_FORMAT";
      break;
    case SQLITE_RANGE:
      errcode = "SQLITE_RANGE";
      break;
    case SQLITE_NOTADB:
      errcode = "SQLITE_NOTADB";
      break;
    case SQLITE_NOTICE:
      errcode = "SQLITE_NOTICE";
      break;
    case SQLITE_WARNING :
      errcode = "SQLITE_WARNING";
      break;
    case SQLITE_ROW:
      errcode = "SQLITE_ROW";
      break;
    case SQLITE_DONE:
      errcode = "SQLITE_DONE";
      break;
    default:
      errcode = "UNKNOWN ERROR";
      break;
  }
  const char *errmsg = sqlite3_errstr(rc);
  string errstr(errmsg);
  if (errstr.size() <= 0 || errmsg == nullptr) {
    errstr = "UNKNOWN_ERROR";
  }
  return errcode + ": " + errstr;
}

void SQLiteHelper::CreateTables() {
  int rc = 0;

  {
    stringstream ss;
    ss << "CREATE TABLE IF NOT EXISTS kernel_cache (";
    ss << "id INTEGER PRIMARY KEY NOT NULL,";
    ss << "hash INTEGER NOT NULL,";
    ss << "flags TEXT,";
    ss << "code TEXT,";
    ss << "program BLOB);";
    rc = sqlite3_exec(db_, ss.str().c_str(), nullptr, nullptr, nullptr);

    if (is_error(rc)) {
      LOG(ERROR) << "Can't create table: kernel_cache "
                 << "(" << parse_error_code(rc) << ")";
      LogDatabaseError();
    }
  }

  {
    stringstream ss;
    ss << "CREATE INDEX IF NOT EXISTS idx_kernel_cache_hash "
       << "ON kernel_cache (hash);";
    rc = sqlite3_exec(db_, ss.str().c_str(), nullptr, nullptr, nullptr);

    if (is_error(rc)) {
      LOG(ERROR) << "Can't create index: idx_kernel_cache_hash "
                 << "(" << parse_error_code(rc) << ")";
      LogDatabaseError();
    }
  }
}

void SQLiteHelper::StoreKernel(int64_t hash,
                               const char* flags,
                               size_t flags_size,
                               const char* code,
                               size_t code_size,
                               const char* program,
                               size_t program_size) {
  int rc = 0;

  stringstream ss;
  ss << "INSERT INTO kernel_cache (hash, flags, code, program) "
     << "VALUES (?, ?, ?, ?)";

  sqlite3_stmt *stmt;
  rc = sqlite3_prepare_v2(db_, ss.str().c_str(), ss.str().length(), &stmt,
                          nullptr);
  if (is_error(rc)) {
    LOG(ERROR) << "Can't prepare insert into kernel_cache "
               << "(" << parse_error_code(rc) << ")";
    LogDatabaseError();
  }
  sqlite3_bind_int64(stmt, 1, hash);
  sqlite3_bind_text64(stmt, 2, flags, flags_size, SQLITE_TRANSIENT,
                      SQLITE_UTF8);
  sqlite3_bind_text64(stmt, 3, code, code_size, SQLITE_TRANSIENT,
                      SQLITE_UTF8);
  sqlite3_bind_blob64(stmt, 4, program, program_size, SQLITE_TRANSIENT);

  rc = sqlite3_step(stmt);
  if (is_error(rc)) {
    LOG(ERROR) << "Can't step insert into kernel_cache "
               << "(" << parse_error_code(rc) << ")";
    LogDatabaseError();
  }
  rc = sqlite3_finalize(stmt);
  if (is_error(rc)) {
    LOG(ERROR) << "Can't finalize insert into kernel_cache "
               << "(" << parse_error_code(rc) << ")";
    LogDatabaseError();
  }
}

int64_t SQLiteHelper::GetKernelInfo(int64_t hash,
                                    const char* flags,
                                    size_t flags_size,
                                    const char* code,
                                    size_t code_size,
                                    size_t* program_size) {
  *program_size = 0;
  int64_t id = -1;
  int rc = 0;

  {
    stringstream ss;
    ss << "SELECT id, flags, code FROM kernel_cache WHERE hash = ?";
    sqlite3_stmt *stmt;
    rc = sqlite3_prepare_v2(db_, ss.str().c_str(), ss.str().length(), &stmt,
                            nullptr);
    if (is_error(rc)) {
      LOG(ERROR) << "Can't prepare select from kernel_cache "
                 << "(" << parse_error_code(rc) << ")";
      LogDatabaseError();
    }
    sqlite3_bind_int64(stmt, 1, hash);
    rc = sqlite3_step(stmt);
    if (is_error(rc)) {
      LOG(ERROR) << "Can't step select from kernel_cache "
                 << "(" << parse_error_code(rc) << ")";
      LogDatabaseError();
    }
    while (rc == SQLITE_ROW) {
      bool is_same = true;
      int64_t temp_id = sqlite3_column_int64(stmt, 0);
      if (is_same) {
        size_t stored_flags_size = sqlite3_column_bytes(stmt, 1);
        // Flags size has not changed
        if (flags_size == stored_flags_size) {
          const char* stored_flags = reinterpret_cast<const char*>(
                                                  sqlite3_column_text(stmt, 1));
          for (size_t i = 0; i < flags_size; ++i) {
            is_same = is_same && (stored_flags[i] == flags[i]);
            if (!is_same) {
              // Code content is different - can't be identical
              break;
            }
          }
        } else {
          // Flags size is different - can't be identical
          is_same = false;
        }
      }
      if (is_same) {
        size_t stored_code_size = sqlite3_column_bytes(stmt, 2);
        // Code size has not changed
        if (code_size == stored_code_size) {
          const char* stored_code = reinterpret_cast<const char*>(
                                                  sqlite3_column_text(stmt, 2));
          for (size_t i = 0; i < code_size; ++i) {
            is_same = is_same && (stored_code[i] == code[i]);
            if (!is_same) {
              // Code content is different - can't be identical
              break;
            }
          }
        } else {
          // Code size is different - can't be identical
          is_same = false;
        }
      }
      if (is_same) {
        // Identical code found, return row id
        id = temp_id;
        break;
      }
      rc = sqlite3_step(stmt);
      if (is_error(rc)) {
        LOG(ERROR) << "Can't step select from kernel_cache "
                   << "(" << parse_error_code(rc) << ")";
        LogDatabaseError();
      }
    }
    rc = sqlite3_finalize(stmt);
    if (is_error(rc)) {
      LOG(ERROR) << "Can't finalize select from kernel_cache "
                 << "(" << parse_error_code(rc) << ")";
      LogDatabaseError();
    }
  }

  if (id > 0) {
    stringstream ss;
    ss << "SELECT program FROM kernel_cache WHERE id = ?";
    sqlite3_stmt *stmt;
    rc = sqlite3_prepare_v2(db_, ss.str().c_str(), ss.str().length(), &stmt,
                            nullptr);
    if (is_error(rc)) {
      LOG(ERROR) << "Can't prepare select from kernel_cache "
                 << "(" << parse_error_code(rc) << ")";
      LogDatabaseError();
    }
    sqlite3_bind_int64(stmt, 1, id);
    rc = sqlite3_step(stmt);
    if (is_error(rc)) {
      LOG(ERROR) << "Can't step select from kernel_cache "
                 << "(" << parse_error_code(rc) << ")";
      LogDatabaseError();
    }
    if (rc == SQLITE_ROW && program_size != nullptr) {
      *program_size = sqlite3_column_bytes(stmt, 0);
    }
    rc = sqlite3_finalize(stmt);
    if (is_error(rc)) {
      LOG(ERROR) << "Can't finalize select from kernel_cache "
                 << "(" << parse_error_code(rc) << ")";
      LogDatabaseError();
    }
    if (*program_size == 0) {
      // Program binary invalid
      id = -1;
    }
  }

  return id;
}

bool SQLiteHelper::LoadKernel(int64_t id, char* program) {
  int rc = 0;

  stringstream ss;
  ss << "SELECT program FROM kernel_cache WHERE id = ?";

  bool success = true;

  sqlite3_stmt *stmt;
  rc = sqlite3_prepare_v2(db_, ss.str().c_str(), ss.str().length(), &stmt,
                          nullptr);
  if (is_error(rc)) {
    LOG(ERROR) << "Can't prepare select from kernel_cache "
               << "(" << parse_error_code(rc) << ")";
    LogDatabaseError();
  }
  sqlite3_bind_int64(stmt, 1, id);
  rc = sqlite3_step(stmt);
  if (is_error(rc)) {
    LOG(ERROR) << "Can't step select from kernel_cache "
               << "(" << parse_error_code(rc) << ")";
    LogDatabaseError();
  }
  if (rc == SQLITE_ROW) {
    size_t program_size = sqlite3_column_bytes(stmt, 0);
    memcpy(program, sqlite3_column_blob(stmt, 0), program_size);  // NOLINT
  } else {
    success = false;
  }

  rc = sqlite3_finalize(stmt);
  if (is_error(rc)) {
    LOG(ERROR) << "Can't finalize select from kernel_cache "
               << "(" << parse_error_code(rc) << ")";
    LogDatabaseError();
  }
  return success;
}

}  // namespace caffe

#endif  // USE_SQLITE
