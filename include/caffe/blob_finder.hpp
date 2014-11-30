#pragma once
#include <map>
#include <string>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"

namespace caffe {

template <typename Dtype>
struct BlobFinder {
  typedef std::map<Blob<Dtype>*, std::string> BlobToNameMap;
  typedef std::map<std::string, Blob<Dtype>*> NameToBlobMap;
  typedef boost::shared_ptr<BlobFinder<Dtype> > Ptr;

  void AddBlob(const std::string& name, Blob<Dtype>* blob_ptr);
  Blob<Dtype>* PointerFromName(const std::string& name);
  std::string NameFromPointer(Blob<Dtype>* blob_pointer) const;

  bool Exists(const std::string& name) const;
 private:
  BlobToNameMap blob_to_name_;
  NameToBlobMap name_to_blob_;
};

}  // namespace caffe
