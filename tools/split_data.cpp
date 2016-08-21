// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

int main(int argc, char** argv) {
    // ::google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();

    if (argc != 3) {
        LOG(ERROR) << "./split_data.bin db_path split_num";
        return -1;
    }

    int num_dbs = 0;
    sscanf(argv[2], "%d", &num_dbs);


    if (num_dbs <= 0) {
        LOG(ERROR) << "invalid number of dbs: " << num_dbs;
        return -1;
    }

    scoped_ptr<db::DB> db(db::GetDB("lmdb"));

    db->Open(argv[1], db::READ);
    scoped_ptr<db::Cursor> cursor(db->NewCursor());

    int cnt = 0;
    #if 1
    for (; cursor->valid(); cursor->Next()) {
        cnt++;
        if (cnt % 10000 == 0) {
            LOG(INFO) << "found " << cnt << " items in data base " << argv[1];
            // LOG(INFO) << "key: " << cursor->key();
        }
    }
    cursor->SeekToFirst();
    #endif

    int per_db_items = cnt / num_dbs;

    // num_dbs = 1;
    // per_db_items = 50000;

    LOG(INFO) << "splitting the db into " << num_dbs <<" each" \
                 " db has " << per_db_items << " items";

    for (int i = 0; i < num_dbs; i++) {
        char db_name[500];
        snprintf(db_name, sizeof(db_name), "%s_%d", argv[1], i);

        scoped_ptr<db::DB> new_db(db::GetDB("lmdb"));
        new_db->Open(db_name, db::NEW);
        scoped_ptr<db::Transaction> txn(new_db->NewTransaction());

        LOG(INFO) << "start to copy";

        int j = 0;
        for (j = 0; j < per_db_items; j++, cursor->Next()) {
            txn->Put(cursor->key(), cursor->value());
            if (j % 1000 == 0) {
                txn->Commit();
                txn.reset(new_db->NewTransaction());
                LOG(ERROR) << "Processed " << j << " items in db: " << db_name;
            }
        }

        if (j % 1000 != 0) {
            txn->Commit();
            LOG(ERROR) << "Processed " << j << " items in db: " << db_name;
        }
    }

    return 0;
}


