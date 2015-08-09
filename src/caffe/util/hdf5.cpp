#include "caffe/util/hdf5.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace caffe {

template <typename Dtype>
void HDF5PrepareBlob(hid_t file_id, const char* dataset_name, int num,
                     Blob<Dtype>* blob) {
  // Verify that the dataset exists.
  CHECK(H5LTfind_dataset(file_id, dataset_name))
      << "Failed to find HDF5 dataset " << dataset_name;
  herr_t status;
  int ndims;
  CHECK_LE(0, H5LTget_dataset_ndims(file_id, dataset_name, &ndims))
      << "Failed to get dataset ndims for " << dataset_name;
  CHECK_GE(ndims, 1) << "HDF5 dataset must have at least 1 dimension.";
  CHECK_LE(ndims, kMaxBlobAxes)
      << "HDF5 dataset must have at most "
      << kMaxBlobAxes << " dimensions, to fit in a Blob.";

  // Verify that the data format is what we expect: float or double.
  std::vector<hsize_t> dims(ndims);
  H5T_class_t h5_class;
  status = H5LTget_dataset_info(
      file_id, dataset_name, dims.data(), &h5_class, NULL);
  CHECK_GE(status, 0) << "Failed to get dataset info for " << dataset_name;
  CHECK_EQ(h5_class, H5T_FLOAT) << "Expected float or double data";
  CHECK_GE(num, -1) << "num must be -1 (to indicate the number of rows"
                       "in the dataset) or non-negative.";

  vector<int> blob_dims(dims.size());
  blob_dims[0] = (num == -1) ? dims[0] : num;
  for (int i = 1; i < dims.size(); ++i) {
    blob_dims[i] = dims[i];
  }
  blob->Reshape(blob_dims);
}

template
void HDF5PrepareBlob<float>(hid_t file_id, const char* dataset_name, int num,
                            Blob<float>* blob);

template
void HDF5PrepareBlob<double>(hid_t file_id, const char* dataset_name, int num,
                             Blob<double>* blob);

template <typename Dtype>
int HDF5ReadRowsToBlob(hid_t file_id, const char* dataset_name,
                       int h5_offset, int blob_offset, Blob<Dtype>* blob) {
  int ndims;
  CHECK_LE(0, H5LTget_dataset_ndims(file_id, dataset_name, &ndims))
      << "Failed to get dataset ndims for " << dataset_name;
  std::vector<hsize_t> dims(ndims);
  H5T_class_t h5_class;
  herr_t status = H5LTget_dataset_info(
      file_id, dataset_name, dims.data(), &h5_class, NULL);
  CHECK_GE(status, 0) << "Failed to get dataset info for " << dataset_name;
  CHECK_EQ(h5_class, H5T_FLOAT) << "Expected float or double data";
  hid_t dataset = H5Dopen2(file_id, dataset_name, H5P_DEFAULT);
  hid_t dataspace = H5Dget_space(dataset);
  vector<hsize_t> slab_start(ndims, 0);
  slab_start[0] = h5_offset;
  const int num_rows_available = dims[0] - h5_offset;
  const int num_rows = std::min(blob->num() - blob_offset, num_rows_available);
  if (num_rows <= 0) {
    return 0;
  }
  vector<hsize_t> slab_count(ndims, num_rows);
  for (int i = 1; i < ndims; ++i) {
    slab_count[i] = dims[i];
  }
  status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET,
      slab_start.data(), NULL, slab_count.data(), NULL);
  CHECK_GE(status, 0) << "Failed to select slab.";
  hid_t memspace = H5Screate_simple(ndims, slab_count.data(), NULL);
  const int data_size = blob->count() / blob->num();
  // separate multiplication to avoid a possible overflow
  const int blob_offset_size = blob_offset * data_size;
  hid_t type = (sizeof(Dtype) == 4) ? H5T_NATIVE_FLOAT : H5T_NATIVE_DOUBLE;
  status = H5Dread(dataset, type, memspace, dataspace, H5P_DEFAULT,
                   blob->mutable_cpu_data() + blob_offset_size);
  CHECK_GE(status, 0) << "Failed to read dataset " << dataset_name;
  H5Dclose(dataset);
  H5Sclose(dataspace);
  H5Sclose(memspace);
  return num_rows;
}

template
int HDF5ReadRowsToBlob<float>(hid_t file_id, const char* dataset_name,
    int h5_offset, int blob_offset, Blob<float>* data);

template
int HDF5ReadRowsToBlob<double>(hid_t file_id, const char* dataset_name,
    int h5_offset, int blob_offset, Blob<double>* data);

// Verifies format of data stored in HDF5 file and reshapes blob accordingly.
template <typename Dtype>
void hdf5_load_nd_dataset_helper(
    hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
    Blob<Dtype>* blob) {
  // Verify that the dataset exists.
  CHECK(H5LTfind_dataset(file_id, dataset_name_))
      << "Failed to find HDF5 dataset " << dataset_name_;
  // Verify that the number of dimensions is in the accepted range.
  herr_t status;
  int ndims;
  status = H5LTget_dataset_ndims(file_id, dataset_name_, &ndims);
  CHECK_GE(status, 0) << "Failed to get dataset ndims for " << dataset_name_;
  CHECK_GE(ndims, min_dim);
  CHECK_LE(ndims, max_dim);

  // Verify that the data format is what we expect: float or double.
  std::vector<hsize_t> dims(ndims);
  H5T_class_t class_;
  status = H5LTget_dataset_info(
      file_id, dataset_name_, dims.data(), &class_, NULL);
  CHECK_GE(status, 0) << "Failed to get dataset info for " << dataset_name_;
  CHECK_EQ(class_, H5T_FLOAT) << "Expected float or double data";

  vector<int> blob_dims(dims.size());
  for (int i = 0; i < dims.size(); ++i) {
    blob_dims[i] = dims[i];
  }
  blob->Reshape(blob_dims);
}

template <>
void hdf5_load_nd_dataset<float>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim, Blob<float>* blob) {
  hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob);
  herr_t status = H5LTread_dataset_float(
    file_id, dataset_name_, blob->mutable_cpu_data());
  CHECK_GE(status, 0) << "Failed to read float dataset " << dataset_name_;
}

template <>
void hdf5_load_nd_dataset<double>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim, Blob<double>* blob) {
  hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob);
  herr_t status = H5LTread_dataset_double(
    file_id, dataset_name_, blob->mutable_cpu_data());
  CHECK_GE(status, 0) << "Failed to read double dataset " << dataset_name_;
}

template <>
void hdf5_save_nd_dataset<float>(
    const hid_t file_id, const string& dataset_name, const Blob<float>& blob,
    bool write_diff) {
  int num_axes = blob.num_axes();
  std::vector<hsize_t> dims(num_axes);
  for (int i = 0; i < num_axes; ++i) {
    dims[i] = blob.shape(i);
  }
  const float* data;
  if (write_diff) {
    data = blob.cpu_diff();
  } else {
    data = blob.cpu_data();
  }
  herr_t status = H5LTmake_dataset_float(
      file_id, dataset_name.c_str(), num_axes, dims.data(), data);
  CHECK_GE(status, 0) << "Failed to make float dataset " << dataset_name;
}

template <>
void hdf5_save_nd_dataset<double>(
    hid_t file_id, const string& dataset_name, const Blob<double>& blob,
    bool write_diff) {
  int num_axes = blob.num_axes();
  std::vector<hsize_t> dims(num_axes);
  for (int i = 0; i < num_axes; ++i) {
    dims[i] = blob.shape(i);
  }
  const double* data;
  if (write_diff) {
    data = blob.cpu_diff();
  } else {
    data = blob.cpu_data();
  }
  herr_t status = H5LTmake_dataset_double(
      file_id, dataset_name.c_str(), num_axes, dims.data(), data);
  CHECK_GE(status, 0) << "Failed to make double dataset " << dataset_name;
}

string hdf5_load_string(hid_t loc_id, const string& dataset_name) {
  // Get size of dataset
  size_t size;
  H5T_class_t class_;
  herr_t status = \
    H5LTget_dataset_info(loc_id, dataset_name.c_str(), NULL, &class_, &size);
  CHECK_GE(status, 0) << "Failed to get dataset info for " << dataset_name;
  char *buf = new char[size];
  status = H5LTread_dataset_string(loc_id, dataset_name.c_str(), buf);
  CHECK_GE(status, 0)
    << "Failed to load int dataset with name " << dataset_name;
  string val(buf);
  delete[] buf;
  return val;
}

void hdf5_save_string(hid_t loc_id, const string& dataset_name,
                      const string& s) {
  herr_t status = \
    H5LTmake_dataset_string(loc_id, dataset_name.c_str(), s.c_str());
  CHECK_GE(status, 0)
    << "Failed to save string dataset with name " << dataset_name;
}

int hdf5_load_int(hid_t loc_id, const string& dataset_name) {
  int val;
  herr_t status = H5LTread_dataset_int(loc_id, dataset_name.c_str(), &val);
  CHECK_GE(status, 0)
    << "Failed to load int dataset with name " << dataset_name;
  return val;
}

void hdf5_save_int(hid_t loc_id, const string& dataset_name, int i) {
  hsize_t one = 1;
  herr_t status = \
    H5LTmake_dataset_int(loc_id, dataset_name.c_str(), 1, &one, &i);
  CHECK_GE(status, 0)
    << "Failed to save int dataset with name " << dataset_name;
}

int hdf5_get_num_links(hid_t loc_id) {
  H5G_info_t info;
  herr_t status = H5Gget_info(loc_id, &info);
  CHECK_GE(status, 0) << "Error while counting HDF5 links.";
  return info.nlinks;
}

string hdf5_get_name_by_idx(hid_t loc_id, int idx) {
  ssize_t str_size = H5Lget_name_by_idx(
      loc_id, ".", H5_INDEX_NAME, H5_ITER_NATIVE, idx, NULL, 0, H5P_DEFAULT);
  CHECK_GE(str_size, 0) << "Error retrieving HDF5 dataset at index " << idx;
  char *c_str = new char[str_size+1];
  ssize_t status = H5Lget_name_by_idx(
      loc_id, ".", H5_INDEX_NAME, H5_ITER_NATIVE, idx, c_str, str_size+1,
      H5P_DEFAULT);
  CHECK_GE(status, 0) << "Error retrieving HDF5 dataset at index " << idx;
  string result(c_str);
  delete[] c_str;
  return result;
}

}  // namespace caffe
