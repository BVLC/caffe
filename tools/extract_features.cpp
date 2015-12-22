#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/dataset_factory.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod/post-rebase-error-fix
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
using std::string;
namespace db = caffe::db;
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
using boost::shared_ptr;
=======
>>>>>>> BVLC/master
=======
using boost::shared_ptr;
>>>>>>> pod-caffe-pod.hpp-merge
using std::string;
namespace db = caffe::db;
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
using boost::shared_ptr;
using caffe::Blob;
using caffe::Caffe;
using caffe::Dataset;
using caffe::DatasetFactory;
using caffe::Datum;
using caffe::Net;
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
using boost::shared_ptr;
using std::string;
namespace db = caffe::db;
>>>>>>> device-abstraction
=======
using boost::shared_ptr;
using std::string;
namespace db = caffe::db;
>>>>>>> pod/post-rebase-error-fix
=======
>>>>>>> pod-caffe-pod.hpp-merge

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
  return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  const int num_required_args = 7;
  if (argc < num_required_args) {
    LOG(ERROR)<<
    "This program takes in a trained network and an input data layer, and then"
    " extract features of the input data produced by the net.\n"
    "Usage: extract_features  pretrained_net_param"
    "  feature_extraction_proto_file  extract_feature_blob_name1[,name2,...]"
    "  save_feature_dataset_name1[,name2,...]  num_mini_batches  db_type"
    "  [CPU/GPU] [DEVICE_ID=0]\n"
    "Note: you can extract multiple features in one pass by specifying"
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
    " multiple feature blob names and dataset names separated by ','."
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
    " multiple feature blob names and dataset names separated by ','."
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
    " multiple feature blob names and dataset names separated by ','."
=======
>>>>>>> pod/device/blob.hpp
=======
    " multiple feature blob names and dataset names separated by ','."
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
    " multiple feature blob names and dataset names separated by ','."
=======
>>>>>>> pod/caffe-merge
=======
    " multiple feature blob names and dataset names separated by ','."
=======
=======
    " multiple feature blob names and dataset names separated by ','."
=======
>>>>>>> pod/device/blob.hpp
=======
    " multiple feature blob names and dataset names separated by ','."
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
    " multiple feature blob names and dataset names separated by ','."
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
    " multiple feature blob names and dataset names separated by ','."
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
    " multiple feature blob names and dataset names separated by ','."
=======
<<<<<<< HEAD
<<<<<<< HEAD
    " multiple feature blob names and dataset names separated by ','."
=======
>>>>>>> pod/device/blob.hpp
=======
    " multiple feature blob names and dataset names separated by ','."
=======
>>>>>>> pod-caffe-pod.hpp-merge
    " multiple feature blob names and dataset names seperated by ','."
>>>>>>> origin/BVLC/parallel
=======
    " multiple feature blob names and dataset names separated by ','."
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
    " multiple feature blob names and dataset names separated by ','."
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
    " multiple feature blob names and dataset names separated by ','."
>>>>>>> device-abstraction
=======
    " multiple feature blob names and dataset names separated by ','."
>>>>>>> pod/post-rebase-error-fix
=======
>>>>>>> pod-caffe-pod.hpp-merge
    " The names cannot contain white space characters and the number of blobs"
    " and datasets must be equal.";
    return 1;
  }
  int arg_pos = num_required_args;

  arg_pos = num_required_args;
  if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
    LOG(ERROR)<< "Using GPU";
    int device_id = 0;
    if (argc > arg_pos + 1) {
      device_id = atoi(argv[arg_pos + 1]);
      CHECK_GE(device_id, 0);
    }
    LOG(ERROR) << "Using Device_id=" << device_id;
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  arg_pos = 0;  // the name of the executable
  std::string pretrained_binary_proto(argv[++arg_pos]);

  // Expected prototxt contains at least one data layer such as
  //  the layer data_layer_name and one feature blob such as the
  //  fc7 top blob to extract features.
  /*
   layers {
     name: "data_layer_name"
     type: DATA
     data_param {
       source: "/path/to/your/images/to/extract/feature/images_leveldb"
       mean_file: "/path/to/your/image_mean.binaryproto"
       batch_size: 128
       crop_size: 227
       mirror: false
     }
     top: "data_blob_name"
     top: "label_blob_name"
   }
   layers {
     name: "drop7"
     type: DROPOUT
     dropout_param {
       dropout_ratio: 0.5
     }
     bottom: "fc7"
     top: "fc7"
   }
   */
  std::string feature_extraction_proto(argv[++arg_pos]);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  boost::shared_ptr<Net<Dtype> > feature_extraction_net(
=======
  shared_ptr<Net<Dtype> > feature_extraction_net(
>>>>>>> pod-caffe-pod.hpp-merge
=======
  boost::shared_ptr<Net<Dtype> > feature_extraction_net(
>>>>>>> BVLC/master
=======
  shared_ptr<Net<Dtype> > feature_extraction_net(
>>>>>>> device-abstraction
=======
  shared_ptr<Net<Dtype> > feature_extraction_net(
>>>>>>> pod/post-rebase-error-fix
=======
  shared_ptr<Net<Dtype> > feature_extraction_net(
>>>>>>> pod-caffe-pod.hpp-merge
      new Net<Dtype>(feature_extraction_proto, caffe::TEST));
  feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

  std::string extract_feature_blob_names(argv[++arg_pos]);
  std::vector<std::string> blob_names;
  boost::split(blob_names, extract_feature_blob_names, boost::is_any_of(","));

  std::string save_feature_dataset_names(argv[++arg_pos]);
  std::vector<std::string> dataset_names;
  boost::split(dataset_names, save_feature_dataset_names,
               boost::is_any_of(","));
  CHECK_EQ(blob_names.size(), dataset_names.size()) <<
      " the number of blob names and dataset names must be equal";
  size_t num_features = blob_names.size();

  for (size_t i = 0; i < num_features; i++) {
    CHECK(feature_extraction_net->has_blob(blob_names[i]))
        << "Unknown feature blob name " << blob_names[i]
        << " in the network " << feature_extraction_proto;
  }

  int num_mini_batches = atoi(argv[++arg_pos]);

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
  std::vector<shared_ptr<db::DB> > feature_dbs;
  std::vector<shared_ptr<db::Transaction> > txns;
  const char* db_type = argv[++arg_pos];
  for (size_t i = 0; i < num_features; ++i) {
    LOG(INFO)<< "Opening dataset " << dataset_names[i];
    shared_ptr<db::DB> db(db::GetDB(db_type));
    db->Open(dataset_names.at(i), db::NEW);
    feature_dbs.push_back(db);
    shared_ptr<db::Transaction> txn(db->NewTransaction());
    txns.push_back(txn);
=======
  std::vector<shared_ptr<Dataset<std::string, Datum> > > feature_dbs;
  for (size_t i = 0; i < num_features; ++i) {
    LOG(INFO)<< "Opening dataset " << dataset_names[i];
    shared_ptr<Dataset<std::string, Datum> > dataset =
        DatasetFactory<std::string, Datum>(argv[++arg_pos]);
    CHECK(dataset->open(dataset_names.at(i), Dataset<std::string, Datum>::New));
    feature_dbs.push_back(dataset);
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
  std::vector<shared_ptr<db::DB> > feature_dbs;
  std::vector<shared_ptr<db::Transaction> > txns;
<<<<<<< HEAD
<<<<<<< HEAD
=======
  std::vector<boost::shared_ptr<db::DB> > feature_dbs;
  std::vector<boost::shared_ptr<db::Transaction> > txns;
>>>>>>> BVLC/master
  const char* db_type = argv[++arg_pos];
  for (size_t i = 0; i < num_features; ++i) {
    LOG(INFO)<< "Opening dataset " << dataset_names[i];
    boost::shared_ptr<db::DB> db(db::GetDB(db_type));
    db->Open(dataset_names.at(i), db::NEW);
    feature_dbs.push_back(db);
    boost::shared_ptr<db::Transaction> txn(db->NewTransaction());
=======
=======
>>>>>>> pod/device/blob.hpp
  const char* db_type = argv[++arg_pos];
  for (size_t i = 0; i < num_features; ++i) {
    LOG(INFO)<< "Opening dataset " << dataset_names[i];
    shared_ptr<db::DB> db(db::GetDB(db_type));
    db->Open(dataset_names.at(i), db::NEW);
    feature_dbs.push_back(db);
    shared_ptr<db::Transaction> txn(db->NewTransaction());
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
    txns.push_back(txn);
<<<<<<< HEAD
=======
  std::vector<shared_ptr<Dataset<std::string, Datum> > > feature_dbs;
  for (size_t i = 0; i < num_features; ++i) {
    LOG(INFO)<< "Opening dataset " << dataset_names[i];
    shared_ptr<Dataset<std::string, Datum> > dataset =
        DatasetFactory<std::string, Datum>(argv[++arg_pos]);
    CHECK(dataset->open(dataset_names.at(i), Dataset<std::string, Datum>::New));
    feature_dbs.push_back(dataset);
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
  std::vector<shared_ptr<db::DB> > feature_dbs;
  std::vector<shared_ptr<db::Transaction> > txns;
  const char* db_type = argv[++arg_pos];
  for (size_t i = 0; i < num_features; ++i) {
    LOG(INFO)<< "Opening dataset " << dataset_names[i];
    shared_ptr<db::DB> db(db::GetDB(db_type));
    db->Open(dataset_names.at(i), db::NEW);
    feature_dbs.push_back(db);
    shared_ptr<db::Transaction> txn(db->NewTransaction());
    txns.push_back(txn);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
  std::vector<shared_ptr<Dataset<std::string, Datum> > > feature_dbs;
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
  for (size_t i = 0; i < num_features; ++i) {
    LOG(INFO)<< "Opening dataset " << dataset_names[i];
    shared_ptr<Dataset<std::string, Datum> > dataset =
        DatasetFactory<std::string, Datum>(argv[++arg_pos]);
    CHECK(dataset->open(dataset_names.at(i), Dataset<std::string, Datum>::New));
    feature_dbs.push_back(dataset);
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/post-rebase-error-fix
=======
>>>>>>> pod-caffe-pod.hpp-merge
  std::vector<shared_ptr<db::DB> > feature_dbs;
  std::vector<shared_ptr<db::Transaction> > txns;
  const char* db_type = argv[++arg_pos];
  for (size_t i = 0; i < num_features; ++i) {
    LOG(INFO)<< "Opening dataset " << dataset_names[i];
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
  std::vector<shared_ptr<Dataset<std::string, Datum> > > feature_dbs;
>>>>>>> pod-caffe-pod.hpp-merge
  for (size_t i = 0; i < num_features; ++i) {
    LOG(INFO)<< "Opening dataset " << dataset_names[i];
    shared_ptr<Dataset<std::string, Datum> > dataset =
        DatasetFactory<std::string, Datum>(argv[++arg_pos]);
    CHECK(dataset->open(dataset_names.at(i), Dataset<std::string, Datum>::New));
    feature_dbs.push_back(dataset);
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
  std::vector<shared_ptr<db::DB> > feature_dbs;
  std::vector<shared_ptr<db::Transaction> > txns;
  const char* db_type = argv[++arg_pos];
  for (size_t i = 0; i < num_features; ++i) {
    LOG(INFO)<< "Opening dataset " << dataset_names[i];
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
  std::vector<shared_ptr<db::DB> > feature_dbs;
  std::vector<shared_ptr<db::Transaction> > txns;
  const char* db_type = argv[++arg_pos];
  for (size_t i = 0; i < num_features; ++i) {
    LOG(INFO)<< "Opening dataset " << dataset_names[i];
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
    shared_ptr<db::DB> db(db::GetDB(db_type));
=======
  std::vector<boost::shared_ptr<db::DB> > feature_dbs;
  std::vector<boost::shared_ptr<db::Transaction> > txns;
  const char* db_type = argv[++arg_pos];
  for (size_t i = 0; i < num_features; ++i) {
    LOG(INFO)<< "Opening dataset " << dataset_names[i];
    boost::shared_ptr<db::DB> db(db::GetDB(db_type));
>>>>>>> BVLC/master
    db->Open(dataset_names.at(i), db::NEW);
    feature_dbs.push_back(db);
    boost::shared_ptr<db::Transaction> txn(db->NewTransaction());
    txns.push_back(txn);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
    shared_ptr<db::DB> db(db::GetDB(db_type));
=======
  std::vector<boost::shared_ptr<db::DB> > feature_dbs;
  std::vector<boost::shared_ptr<db::Transaction> > txns;
  const char* db_type = argv[++arg_pos];
  for (size_t i = 0; i < num_features; ++i) {
    LOG(INFO)<< "Opening dataset " << dataset_names[i];
    boost::shared_ptr<db::DB> db(db::GetDB(db_type));
>>>>>>> BVLC/master
    db->Open(dataset_names.at(i), db::NEW);
    feature_dbs.push_back(db);
    boost::shared_ptr<db::Transaction> txn(db->NewTransaction());
    txns.push_back(txn);
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
  std::vector<shared_ptr<db::DB> > feature_dbs;
  std::vector<shared_ptr<db::Transaction> > txns;
  const char* db_type = argv[++arg_pos];
  for (size_t i = 0; i < num_features; ++i) {
    LOG(INFO)<< "Opening dataset " << dataset_names[i];
=======
>>>>>>> pod/post-rebase-error-fix
=======
>>>>>>> pod-caffe-pod.hpp-merge
    shared_ptr<db::DB> db(db::GetDB(db_type));
    db->Open(dataset_names.at(i), db::NEW);
    feature_dbs.push_back(db);
    shared_ptr<db::Transaction> txn(db->NewTransaction());
    txns.push_back(txn);
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> device-abstraction
=======
>>>>>>> pod/post-rebase-error-fix
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  }

  LOG(ERROR)<< "Extacting Features";

  Datum datum;
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
  const int kMaxKeyStrLength = 100;
  char key_str[kMaxKeyStrLength];
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod/post-rebase-error-fix
=======
>>>>>>> pod-caffe-pod.hpp-merge
  std::vector<Blob<float>*> input_vec;
  std::vector<int> image_indices(num_features, 0);
  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
    feature_extraction_net->Forward(input_vec);
    for (int i = 0; i < num_features; ++i) {
      const boost::shared_ptr<Blob<Dtype> > feature_blob =
        feature_extraction_net->blob_by_name(blob_names[i]);
      int batch_size = feature_blob->num();
      int dim_features = feature_blob->count() / batch_size;
      const Dtype* feature_blob_data;
      for (int n = 0; n < batch_size; ++n) {
        datum.set_height(feature_blob->height());
        datum.set_width(feature_blob->width());
        datum.set_channels(feature_blob->channels());
        datum.clear_data();
        datum.clear_float_data();
        feature_blob_data = feature_blob->cpu_data() +
            feature_blob->offset(n);
        for (int d = 0; d < dim_features; ++d) {
          datum.add_float_data(feature_blob_data[d]);
        }
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod/post-rebase-error-fix
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
        string key_str = caffe::format_int(image_indices[i], 10);

        string out;
        CHECK(datum.SerializeToString(&out));
        txns.at(i)->Put(key_str, out);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
        ++image_indices[i];
        if (image_indices[i] % 1000 == 0) {
          txns.at(i)->Commit();
          txns.at(i).reset(feature_dbs.at(i)->NewTransaction());
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
        ++image_indices[i];
        if (image_indices[i] % 1000 == 0) {
          txns.at(i)->Commit();
          txns.at(i).reset(feature_dbs.at(i)->NewTransaction());
<<<<<<< HEAD
=======
        int length = snprintf(key_str, kMaxKeyStrLength, "%d",
            image_indices[i]);
        CHECK(feature_dbs.at(i)->put(std::string(key_str, length), datum));
<<<<<<< HEAD
        ++image_indices[i];
        if (image_indices[i] % 1000 == 0) {
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
        int length = snprintf(key_str, kMaxKeyStrLength, "%d",
            image_indices[i]);
        CHECK(feature_dbs.at(i)->put(std::string(key_str, length), datum));
        ++image_indices[i];
        if (image_indices[i] % 1000 == 0) {
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
        ++image_indices[i];
        if (image_indices[i] % 1000 == 0) {
          txns.at(i)->Commit();
          txns.at(i).reset(feature_dbs.at(i)->NewTransaction());
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
        ++image_indices[i];
        if (image_indices[i] % 1000 == 0) {
          txns.at(i)->Commit();
          txns.at(i).reset(feature_dbs.at(i)->NewTransaction());
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
=======
        int length = snprintf(key_str, kMaxKeyStrLength, "%d",
            image_indices[i]);
        CHECK(feature_dbs.at(i)->put(std::string(key_str, length), datum));
        ++image_indices[i];
        if (image_indices[i] % 1000 == 0) {
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
        ++image_indices[i];
        if (image_indices[i] % 1000 == 0) {
          txns.at(i)->Commit();
          txns.at(i).reset(feature_dbs.at(i)->NewTransaction());
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
        int length = snprintf(key_str, kMaxKeyStrLength, "%d",
            image_indices[i]);
        CHECK(feature_dbs.at(i)->put(std::string(key_str, length), datum));
        ++image_indices[i];
        if (image_indices[i] % 1000 == 0) {
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
=======
        ++image_indices[i];
        if (image_indices[i] % 1000 == 0) {
          txns.at(i)->Commit();
          txns.at(i).reset(feature_dbs.at(i)->NewTransaction());
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
        int length = snprintf(key_str, kMaxKeyStrLength, "%d",
            image_indices[i]);
        CHECK(feature_dbs.at(i)->put(std::string(key_str, length), datum));
        ++image_indices[i];
        if (image_indices[i] % 1000 == 0) {
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
          CHECK(feature_dbs.at(i)->commit());
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> device-abstraction
=======
        ++image_indices[i];
        if (image_indices[i] % 1000 == 0) {
          txns.at(i)->Commit();
          txns.at(i).reset(feature_dbs.at(i)->NewTransaction());
>>>>>>> pod/post-rebase-error-fix
=======
        ++image_indices[i];
        if (image_indices[i] % 1000 == 0) {
          CHECK(feature_dbs.at(i)->commit());
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
          LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
              " query images for feature blob " << blob_names[i];
        }
      }  // for (int n = 0; n < batch_size; ++n)
    }  // for (int i = 0; i < num_features; ++i)
  }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
  // write the last batch
  for (int i = 0; i < num_features; ++i) {
    if (image_indices[i] % 1000 != 0) {
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
      txns.at(i)->Commit();
<<<<<<< HEAD
<<<<<<< HEAD
=======
      txns.at(i)->Commit();
>>>>>>> pod/device/blob.hpp
=======
      txns.at(i)->Commit();
>>>>>>> pod-caffe-pod.hpp-merge
=======
      txns.at(i)->Commit();
>>>>>>> pod/caffe-merge
=======
      txns.at(i)->Commit();
>>>>>>> pod/device/blob.hpp
=======
      txns.at(i)->Commit();
>>>>>>> pod/device/blob.hpp
=======
      txns.at(i)->Commit();
>>>>>>> pod/device/blob.hpp
=======
      txns.at(i)->Commit();
>>>>>>> device-abstraction
=======
      txns.at(i)->Commit();
>>>>>>> pod/post-rebase-error-fix
=======
      txns.at(i)->Commit();
>>>>>>> pod-caffe-pod.hpp-merge
    }
    LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
        " query images for feature blob " << blob_names[i];
    feature_dbs.at(i)->Close();
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod/device/blob.hpp
      txns.at(i)->Commit();
    }
    LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
        " query images for feature blob " << blob_names[i];
    feature_dbs.at(i)->Close();
<<<<<<< HEAD
<<<<<<< HEAD
=======
      CHECK(feature_dbs.at(i)->commit());
    }
    LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
        " query images for feature blob " << blob_names[i];
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
      txns.at(i)->Commit();
    }
    LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
        " query images for feature blob " << blob_names[i];
    feature_dbs.at(i)->Close();
=======
<<<<<<< HEAD
      CHECK(feature_dbs.at(i)->commit());
    }
    LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
        " query images for feature blob " << blob_names[i];
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
    }
    LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
        " query images for feature blob " << blob_names[i];
    feature_dbs.at(i)->Close();
=======
      CHECK(feature_dbs.at(i)->commit());
    }
    LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
        " query images for feature blob " << blob_names[i];
>>>>>>> pod/caffe-merge
=======
    }
    LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
        " query images for feature blob " << blob_names[i];
    feature_dbs.at(i)->Close();
=======
      CHECK(feature_dbs.at(i)->commit());
    }
    LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
        " query images for feature blob " << blob_names[i];
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
      CHECK(feature_dbs.at(i)->commit());
    }
    LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
        " query images for feature blob " << blob_names[i];
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
      CHECK(feature_dbs.at(i)->commit());
    }
    LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
        " query images for feature blob " << blob_names[i];
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
      txns.at(i)->Commit();
    }
    LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
        " query images for feature blob " << blob_names[i];
    feature_dbs.at(i)->Close();
=======
      CHECK(feature_dbs.at(i)->commit());
    }
    LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
        " query images for feature blob " << blob_names[i];
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
=======
      CHECK(feature_dbs.at(i)->commit());
    }
    LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
        " query images for feature blob " << blob_names[i];
>>>>>>> pod-caffe-pod.hpp-merge
    feature_dbs.at(i)->close();
>>>>>>> origin/BVLC/parallel
=======
      txns.at(i)->Commit();
    }
    LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
        " query images for feature blob " << blob_names[i];
    feature_dbs.at(i)->Close();
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod/post-rebase-error-fix
=======
>>>>>>> pod-caffe-pod.hpp-merge
  }

  LOG(ERROR)<< "Successfully extracted the features!";
  return 0;
}
