



/*-----------------------------------------------------------------------*/
/*_______________________________________________________________________*/

#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"
#include "stdint.h"

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
HDF5CategoricalDLayer<Dtype>::~HDF5CategoricalDLayer<Dtype>() { }

// loads an integer vector containg the number of categories
// and for each categorie the number of possible values
// if the number of values for some category is N>1 than the possible values
//  are-1(not set),0,..,N-1 if N==1 than the values are -1 (not set) and 0 (set)
void hdf5_load_description(hid_t file_id, vector<unsigned int>& description) {
  char* dataset_name_="description";
  int ndims, DESCRIPTION_DIM=1;
  herr_t status; 
  // verifies that 'description' is there
  CHECK(H5LTfind_dataset(file_id, dataset_name_))
      << "Failed to find HDF5 dataset " << dataset_name_;
  // verify description dataset is a 1dim-vector
  status= H5LTget_dataset_ndims(file_id, dataset_name_, &ndims);
  CHECK_GE(status, 0) << "Failed to get dataset ndims for " << dataset_name_;
  CHECK_EQ(ndims, DESCRIPTION_DIM);

  // Verify that the data format is what we expect: integer
  hsize_t dim;
  H5T_class_t class_;
  status = H5LTget_dataset_info(
      file_id, dataset_name_, &dim, &class_, NULL);
  CHECK_GE(status, 0) << "Failed to get dataset info for " << dataset_name_; 
  CHECK_EQ(class_, H5T_INTEGER) << "Expected integer data";

  description.resize(dim);
  status = H5LTread_dataset (file_id, dataset_name_,H5T_NATIVE_UINT,
			     description.data());
  CHECK_GE(status, 0) << "Failed to read int dataset: description";
  CHECK_EQ(description[0],dim-1);
}

void hdf5_load_int_data(hid_t file_id,Blob<int>* blob,
			 int numcategories){
  char* dataset_name_="data";
  // verifies that 'data' is there
  CHECK(H5LTfind_dataset(file_id, dataset_name_))
    << "Failed to find HDF5 dataset " << dataset_name_;
  /*int DATA_DIM=2;*/
  herr_t status; 
  int ndims, DATA_DIM=2;
  // verify description dataset is a 2dim-matrix
  status= H5LTget_dataset_ndims(file_id, dataset_name_, &ndims);
  CHECK_GE(status, 0) << "Failed to get dataset ndims for " << dataset_name_;
  CHECK_EQ(ndims, DATA_DIM);

  std::vector<hsize_t> dims(ndims);
  H5T_class_t class_;
  status = H5LTget_dataset_info(
      file_id, dataset_name_, dims.data(), &class_, NULL);
  CHECK_GE(status, 0) << "Failed to get dataset info for " << dataset_name_;
  CHECK_EQ(class_, H5T_INTEGER) << "Expected integer data";
  // REMOVE OR CORRECT----------------------------------------!!!!!!!!!!!!!!!
  // H5T_sign_t sgn = H5Tget_sign(class_);
  // CHECK_EQ(sgn,H5T_SGN_2);
  CHECK_EQ(dims[1],numcategories);
  blob->Reshape(dims[0],numcategories,1,1);
  status = H5LTread_dataset_int(file_id, dataset_name_,
				blob->mutable_cpu_data());
  CHECK_GE(status, 0) << "Failed to read int dataset " << dataset_name_;
}

// load hdf5 categorical data file and insert 'data' and 'label' to 
// respective data blobs
// also create a vector<unsigned int> description with the information about
// the number of categories and the possible values for each 
template <typename Dtype>
void 
HDF5CategoricalDLayer<Dtype>::LoadHDF5FileCategoricalData(const char* filename)
{
  LOG(INFO) << "Loading HDF5 file" << filename;
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    LOG(ERROR) << "Failed opening HDF5 file" << filename;
    return;
  }
  hdf5_load_description(file_id, description_);
  int numcategories=description_[0];
  hdf5_load_int_data(file_id, &data_blob_, numcategories);
  const int MIN_LABEL_DIM = 1;
  const int MAX_LABEL_DIM = 2;
  hdf5_load_nd_dataset(
    file_id, "label", MIN_LABEL_DIM, MAX_LABEL_DIM, &label_blob_);

  herr_t status = H5Fclose(file_id);
  CHECK_GE(status, 0) << "Failed to close HDF5 file " << filename;
  CHECK_EQ(data_blob_.num(), label_blob_.num());
  LOG(INFO) << "Successully loaded " << data_blob_.num() << " rows";
}

// set to 1.0 the field corresponding to the value indicated in the 'input'
// for each categorie (number of possible values is given in 'description')
// to indicate that there is no value for that category input data is
// -1 (not set)
template <typename Dtype> void 
HDF5CategoricalDLayer<Dtype>::setcategoriesvalues(const int * input,
						  Dtype*onehotencodedvector){
  int numcategories=(int) description_[0];

  for(int i=0;i<numcategories;++i)
    if(input[i]>-1)
      onehotencodedvector[accumulatedvalues_[i]+input[i]]=	\
	(Dtype)1.0;
}

template <typename Dtype>
void 
HDF5CategoricalDLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
					 const vector<Blob<Dtype>*>& top) {

  // Read the source to parse the filenames.
  const string& source 
    = this->layer_param_.hdf5_categorical_data_param().source();
  LOG(INFO) << "Loading filename from " << source;
  hdf_filenames_.clear();
  std::ifstream source_file(source.c_str());
  if (source_file.is_open()) {
    std::string line;
    while (source_file >> line) {
      hdf_filenames_.push_back(line);
    }
  }
  source_file.close();
  num_files_ = hdf_filenames_.size();
  current_file_ = 0;

  LOG(INFO) << "Number of files: " << num_files_;

  // Load the first HDF5 file and initialize the line counter.
  LoadHDF5FileCategoricalData(hdf_filenames_[current_file_].c_str());
  current_row_ = 0;
  int numcategories=description_[0];
  accumulatedvalues_.resize(numcategories+1);
  accumulatedvalues_[0]=0;
  // use 'description' to compute the number of channels
  // in the top blob
  int nchannels=0;
  for(int i=0;i<numcategories;++i){
    nchannels+=description_[i+1];
    accumulatedvalues_[i+1]=nchannels;
  }
  const int batch_size =
    this->layer_param_.hdf5_categorical_data_param().batch_size();
  // Reshape top blobs
  top[0]->Reshape(batch_size, nchannels, 1,1);
  top[1]->Reshape(batch_size, label_blob_.channels(),
                     label_blob_.height(), label_blob_.width());
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
}

template <typename Dtype>
void
HDF5CategoricalDLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
				   const vector<Blob<Dtype>*>& top) {
  int batch_size = 
    this->layer_param_.hdf5_categorical_data_param().batch_size();
  int top_data_count = top[0]->count() / top[0]->num();
  int inputdata_count = data_blob_.channels();
  int label_data_count = top[1]->count() / top[1]->num();

  // all 0.0
  caffe_set( batch_size*top_data_count, (Dtype) 0.0,
	     top[0]->mutable_cpu_data());
     
  for (int i = 0; i < batch_size; ++i, ++current_row_) {
    if (current_row_ == data_blob_.num()) {
      if (num_files_ > 1) {
	current_file_ += 1;
	if (current_file_ == num_files_) {
	  current_file_ = 0;
	  LOG(INFO) << "looping around to first file";
	}
	LoadHDF5FileCategoricalData(hdf_filenames_[current_file_].c_str());
      }
      current_row_ = 0;
    }
    // one hot encoding of each category
    setcategoriesvalues(&data_blob_.cpu_data()[current_row_*inputdata_count],
			&top[0]->mutable_cpu_data()[i*top_data_count]);
    caffe_copy(label_data_count,
	       &label_blob_.cpu_data()[current_row_ * label_data_count],
	       &top[1]->mutable_cpu_data()[i * label_data_count]);
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(HDF5CategoricalDLayer, Forward);
#endif

  INSTANTIATE_CLASS(HDF5CategoricalDLayer);
  REGISTER_LAYER_CLASS(HDF5CATEGORICAL,HDF5CategoricalDLayer);
}  // namespace caffe
