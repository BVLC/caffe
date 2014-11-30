#include <string>

#include "caffe/blob_finder.hpp"
#include "caffe/common.hpp"

namespace caffe {

template <typename Dtype>
void BlobFinder<Dtype>::AddBlob(
    const std::string& name, Blob<Dtype>* blob_ptr ) {
  blob_to_name_[blob_ptr] = name;
  name_to_blob_[name] = blob_ptr;
}

template <typename Dtype>
Blob<Dtype>* BlobFinder<Dtype>::PointerFromName(const std::string& name) {
  return name_to_blob_[name];
}

template <typename Dtype>
std::string BlobFinder<Dtype>::NameFromPointer(
                      Blob<Dtype>* blob_pointer) const {
  typename BlobToNameMap::const_iterator it = blob_to_name_.find(blob_pointer);
  if (it == blob_to_name_.end()) {
    CHECK(it != blob_to_name_.end()) << "Blob pointer not found in pointer-to-"
                                     << "name map.";
  }

  return it->second;
}

template <typename Dtype>
bool BlobFinder<Dtype>::Exists(const std::string& name) const {
  return name_to_blob_.find(name) != name_to_blob_.end();
}

INSTANTIATE_CLASS(BlobFinder);

}  // namespace caffe
