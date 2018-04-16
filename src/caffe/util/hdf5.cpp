#ifdef USE_HDF5
#include "caffe/util/hdf5.hpp"

#include <string>
#include <vector>

namespace caffe {

// Verifies format of data stored in HDF5 file and reshapes blob accordingly.
template <typename Dtype>
void hdf5_load_nd_dataset_helper(
    hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
    Blob<Dtype>* blob, bool reshape) {
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
  switch (class_) {
  case H5T_FLOAT:
    { LOG_FIRST_N(INFO, 1) << "Datatype class: H5T_FLOAT"; }
    break;
  case H5T_INTEGER:
    { LOG_FIRST_N(INFO, 1) << "Datatype class: H5T_INTEGER"; }
    break;
  case H5T_TIME:
    LOG(FATAL) << "Unsupported datatype class: H5T_TIME";
  case H5T_STRING:
    LOG(FATAL) << "Unsupported datatype class: H5T_STRING";
  case H5T_BITFIELD:
    LOG(FATAL) << "Unsupported datatype class: H5T_BITFIELD";
  case H5T_OPAQUE:
    LOG(FATAL) << "Unsupported datatype class: H5T_OPAQUE";
  case H5T_COMPOUND:
    LOG(FATAL) << "Unsupported datatype class: H5T_COMPOUND";
  case H5T_REFERENCE:
    LOG(FATAL) << "Unsupported datatype class: H5T_REFERENCE";
  case H5T_ENUM:
    LOG(FATAL) << "Unsupported datatype class: H5T_ENUM";
  case H5T_VLEN:
    LOG(FATAL) << "Unsupported datatype class: H5T_VLEN";
  case H5T_ARRAY:
    LOG(FATAL) << "Unsupported datatype class: H5T_ARRAY";
  default:
    LOG(FATAL) << "Datatype class unknown";
  }


  vector<int> blob_dims(dims.size());
  for (int i = 0; i < dims.size(); ++i) {
    blob_dims[i] = dims[i];
  }

  if (reshape) {
    blob->Reshape(blob_dims);
  } else {
    if (blob_dims != blob->shape()) {
      // create shape string for error message
      ostringstream stream;
      int count = 1;
      for (int i = 0; i < blob_dims.size(); ++i) {
        stream << blob_dims[i] << " ";
        count = count * blob_dims[i];
      }
      stream << "(" << count << ")";
      string source_shape_string = stream.str();

      CHECK(blob_dims == blob->shape()) << "Cannot load blob from hdf5; shape "
            << "mismatch. Source shape is " << source_shape_string
            << " target shape is " << blob->shape_string();
    }
  }
}

template <>
void hdf5_load_nd_dataset<float>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim, Blob<float>* blob, bool reshape) {
  hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob,
                              reshape);
  herr_t status = H5LTread_dataset_float(
    file_id, dataset_name_, blob->mutable_cpu_data());
  CHECK_GE(status, 0) << "Failed to read float dataset " << dataset_name_;
}

template <>
void hdf5_load_nd_dataset<double>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim, Blob<double>* blob, bool reshape) {
  hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob,
                              reshape);
  herr_t status = H5LTread_dataset_double(
    file_id, dataset_name_, blob->mutable_cpu_data());
  CHECK_GE(status, 0) << "Failed to read double dataset " << dataset_name_;
}

template <>
void hdf5_save_nd_dataset<float>(
    const hid_t file_id, const string& dataset_name, const Blob<float>& blob,
    bool write_diff) {
  int num_axes = blob.num_axes();
  hsize_t *dims = new hsize_t[num_axes];
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
      file_id, dataset_name.c_str(), num_axes, dims, data);
  CHECK_GE(status, 0) << "Failed to make float dataset " << dataset_name;
  delete[] dims;
}

template <>
void hdf5_save_nd_dataset<double>(
    hid_t file_id, const string& dataset_name, const Blob<double>& blob,
    bool write_diff) {
  int num_axes = blob.num_axes();
  hsize_t *dims = new hsize_t[num_axes];
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
      file_id, dataset_name.c_str(), num_axes, dims, data);
  CHECK_GE(status, 0) << "Failed to make double dataset " << dataset_name;
  delete[] dims;
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
#endif  // USE_HDF5
