#include <gflags/gflags.h>
#include <glog/logging.h>
#include <lmdb.h>
#include <fstream>
#include <iostream>
#include <boost/algorithm/string.hpp>
#include <sys/stat.h>

#include "caffe/proto/caffe.pb.h"

using namespace std;
using namespace boost;
using namespace caffe;

void opendb(const string& db_path, MDB_env*& mdb_env, MDB_txn*& mdb_txn,
            MDB_dbi& mdb_dbi)
{
    LOG(INFO) << "Opening lmdb " << db_path;
    CHECK_EQ(mkdir(db_path.c_str(), 0744), 0)
        << "mkdir " << db_path << "failed";
    CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)  // 1TB
        << "mdb_env_set_mapsize failed";
    CHECK_EQ(mdb_env_open(mdb_env, db_path.c_str(), 0, 0664), MDB_SUCCESS)
        << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
        << "mdb_open failed. Does the lmdb already exist? ";
}

int line2vec(const string& line, vector<float>& vec, uint64_t dim)
{
    vector<string> strVec;
    split( strVec, line, is_any_of(" "));
    vec.resize(dim);
    std::fill(vec.begin(),vec.end(),0);

    int label = (int)atof(strVec[0].c_str());
    float sum = 0.0f;
    for(int i=1;i<strVec.size();i++)
    {
        vector<string> kv;
        split(kv,strVec[i],is_any_of(":"));
        uint64_t key = (uint64_t)atoi(kv[0].c_str());
        float val = atof(kv[1].c_str());
        vec[key]=val;
        sum+=val;
    }

    /*
    if (sum>0)
    {
        for(int i=0;i<vec.size();i++)
        {
            vec[i]/=sum;
        }
    }
    */
    
    return label;
}
void convert(const string& txtFilename, const string& dbFilename, uint64_t dimension)
{
    ifstream txtfile(txtFilename.c_str());
    CHECK(!txtfile.fail()) << "Failed to open file: " << txtFilename;
    string line;

    vector<float> buffer;
    Datum datum;
    datum.set_channels(1);
    datum.set_height(1);
    string dbvalue;
    int iline = 0;
    
    MDB_val mdb_key, mdb_value;

    MDB_env* mdb_env;
    MDB_txn* mdb_txn;
    MDB_dbi mdb_dbi;
    
    opendb(dbFilename, mdb_env, mdb_txn, mdb_dbi);

    while(true)
    {
        if (iline % 10000 == 0)
        {
            LOG(INFO) << iline;
        }
        getline(txtfile,line);
        if (txtfile.eof())
        {
            break;
        }

        trim(line);
        int label = line2vec(line,buffer,dimension);
        datum.set_width(buffer.size());
        //cout << buffer[0] << endl;
        //datum.set_float_data(&buffer[0], buffer.size());
        datum.clear_float_data();
        for(int i=0;i<buffer.size();i++)
        {
            datum.add_float_data(buffer[i]);
        }
        datum.set_label(label);
        /*
        if (label==10)
        {
            cout << label << endl;
        }
        */
        datum.SerializeToString(&dbvalue);
        char buf[32];
        sprintf(buf,"%d",iline);
        string keystr = buf;
        mdb_value.mv_size = dbvalue.size();
        mdb_value.mv_data = reinterpret_cast<void*>(&dbvalue[0]);
        mdb_key.mv_size = keystr.size();
        mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
        CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_value, 0), MDB_SUCCESS)
          << "mdb_put failed";      
        iline++;
        if (iline % 10000==0)
        {
            CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
                << "mdb_txn_commit failed";
            CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
                << "mdb_txn_begin failed";
        }
    }

    CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
        << "mdb_txn_commit failed";
    mdb_close(mdb_env, mdb_dbi);
    mdb_env_close(mdb_env);
}

int main(int argn, char** argv)
{
    convert(argv[1],argv[2],atoi(argv[3]));
}

