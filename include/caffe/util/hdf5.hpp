#ifndef CAFFE_UTIL_HDF5_H_
#define CAFFE_UTIL_HDF5_H_

#include <string>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/blob.hpp"

namespace caffe {

/**
 * @brief Shapes a Blob to read "num" rows of HDF5 data.  If num == -1, take
 *        the num of the HDF5 dataset.
 *
 * @param file_id      the HDF5 file handle
 * @param dataset_name the name of the HDF5 dataset to read
 * @param num          the number of rows to read: either num >= 0,
 *                     or num == -1 for the number of rows in the HDF5 dataset
 * @param blob         the Blob to shape
 *
 * The HDF5 dataset could be N(>=1) dimensions as long as N doesn't exceed
 * Blob's maximum dimension.
 */
template <typename Dtype>
void HDF5PrepareBlob(hid_t file_id, const char* dataset_name, int num,
    Blob<Dtype>* blob);

/**
 * @brief Reads rows [offset, offset + data->num() - 1] into Blob* data, which
 *        must have been pre-shaped using HDF5PrepareBlob (or otherwise).
 */
template <typename Dtype>
int HDF5ReadRowsToBlob(hid_t file_id, const char* dataset_name, int h5_offset,
    int blob_offset, Blob<Dtype>* blob);

template <typename Dtype>
void hdf5_load_nd_dataset_helper(hid_t file_id, const char* dataset_name_,
    int min_dim, int max_dim, Blob<Dtype>* blob);

template <typename Dtype>
void hdf5_load_nd_dataset(hid_t file_id, const char* dataset_name_, int min_dim,
    int max_dim, Blob<Dtype>* blob);

template <typename Dtype>
void hdf5_save_nd_dataset(
    const hid_t file_id, const string& dataset_name, const Blob<Dtype>& blob,
    bool write_diff = false);
int hdf5_load_int(hid_t loc_id, const string& dataset_name);
void hdf5_save_int(hid_t loc_id, const string& dataset_name, int i);
string hdf5_load_string(hid_t loc_id, const string& dataset_name);
void hdf5_save_string(hid_t loc_id, const string& dataset_name,
                      const string& s);

int hdf5_get_num_links(hid_t loc_id);
string hdf5_get_name_by_idx(hid_t loc_id, int idx);

}  // namespace caffe

#endif   // CAFFE_UTIL_HDF5_H_
