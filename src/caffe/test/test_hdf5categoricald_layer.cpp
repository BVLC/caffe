#include <string>
#include <vector>
#include "hdf5.h"
#include "hdf5_hl.h"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {


template <typename Dtype>
void hdf5_save_label(const hid_t file_id,hsize_t dims,Dtype*data);

template <>
void hdf5_save_label<float>(const hid_t file_id,hsize_t dims,float*data){
  herr_t status=H5LTmake_dataset(file_id,"label",1,&dims,H5T_NATIVE_FLOAT,data);
  CHECK_EQ(status,0);
}

template <>
void hdf5_save_label<double>(const hid_t file_id,hsize_t dims,double*data){

  herr_t status=H5LTmake_dataset(file_id,"label",1,&dims,
				 H5T_NATIVE_DOUBLE,data);
  CHECK_EQ(status,0);
}

template <typename Dtype>
void generate_test_data(int numcategories, int numsamples,
			const string&fname0, const string&fname1,Dtype *){
  // each sample has numcategories categories. first category is binary,
  // set or not set,
  // second has two possible values and may be not set,
  // third has three possible values and may be not set, and so on ..
  // when a category is not set, its data value is -1
  int description[numcategories+1];
  description[0]=numcategories;
  for(int i=0;i<numcategories;++i)
    description[i+1]=i+1;
  //firstly unset all categories in all samples
  vector<int> data(numsamples*numcategories,-1);
  for(int i=0;i<numsamples;++i)
    for(int j=0;j<numcategories;++j)
      if(i%(description[j+1]+1))
	data[numcategories*i+j]=i%(description[j+1]+1)-1;
  
  vector<float> label(numsamples);
  for(int i=0;i<numsamples;++i)
    label[i]=(Dtype) (i%2);
  
  // divide data in two unequal parts
  int beginning=numsamples/3;
  int numsamplesend=numsamples-beginning;
  
  // save data on files
  
  hsize_t descriptdims=numcategories+1;
  hsize_t data0dims[2]={beginning,numcategories};
  hsize_t data1dims[2]={numsamplesend,numcategories};
  hsize_t labels0dims=beginning;
  hsize_t labels1dims=numsamplesend;
  // fname0
  hid_t file_id=H5Fcreate (fname0.c_str(), H5F_ACC_TRUNC,
			   H5P_DEFAULT, H5P_DEFAULT);
  herr_t status=H5LTmake_dataset(file_id,"description",1,&descriptdims,
				 H5T_NATIVE_INT,description);
  CHECK_EQ(status,0);
  status=H5LTmake_dataset(file_id,"data",2,data0dims,
			  H5T_NATIVE_INT,&data[0]);
  CHECK_EQ(status,0);
  // status=H5LTmake_dataset(file_id,"label",1,&labels0dims,
  // 				   H5T_NATIVE_FLOAT,&label[0]);
  // CHECK_EQ(status,0);
  hdf5_save_label(file_id, labels0dims, &label[0]);
  
  status = H5Fclose (file_id);
  CHECK_EQ(status,0);
  // fname1
  file_id=H5Fcreate (fname1.c_str(), H5F_ACC_TRUNC,H5P_DEFAULT, H5P_DEFAULT);
  status=H5LTmake_dataset(file_id,"description",1,&descriptdims,
			  H5T_NATIVE_INT,description);
  CHECK_EQ(status,0);
  status=H5LTmake_dataset(file_id,"data",2,data1dims,
			  H5T_NATIVE_INT,&data[beginning*numcategories]);
  CHECK_EQ(status,0);
  // status=H5LTmake_dataset(file_id,"label",1,&labels1dims,
  // 				   H5T_NATIVE_FLOAT,&label[beginning]);
  //CHECK_EQ(status,0);
  hdf5_save_label(file_id, labels1dims, &label[beginning]);
  status = H5Fclose (file_id);
  CHECK_EQ(status,0);
}


template <typename TypeParam>
class HDF5CategoricalDLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;


protected:
  HDF5CategoricalDLayerTest()
    : filename(NULL),
      blob_top_data_(new Blob<Dtype>()),
      blob_top_label_(new Blob<Dtype>()){}
  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
    
    //filename = new string(
    //CMAKE_SOURCE_DIR "caffe/test/test_data/categorical_hdf5_sample_data_list.txt" CMAKE_EXT);
    //LOG(INFO)<< "Using sample categorical HDF5 data file " << *filename;
    //string testdatafname0(CMAKE_SOURCE_DIR "caffe/test/test_data/categorical_hdf5_test_data0.h5" CMAKE_EXT);
    //string testdatafname1(CMAKE_SOURCE_DIR "caffe/test/test_data/categorical_hdf5_test_data1.h5" CMAKE_EXT);
filename = new string("../../src/caffe/test/test_data/categorical_hdf5_sample_data_list.txt");
//LOG(INFO)<< "Using sample categorical HDF5 data file " << *filename;
string testdatafname0("../../src/caffe/test/test_data/categorical_hdf5_test_data0.h5");
    string testdatafname1("../../src/caffe/test/test_data/categorical_hdf5_test_data1.h5");

    //generate test data
    numcategories=5;
    numsamples=50;
    // only used to deduce template, probably there is a much better solution...
    Dtype x=1.0;
    generate_test_data(numcategories, numsamples,
		       testdatafname0, testdatafname1,&x);
    // description vector is similar to  HDF5CategoricalDLayer.description
    // but slightly different
    description.resize(numcategories);
    for(int i=0;i<numcategories;++i)
      description[i]=i+1;
  }

  virtual ~HDF5CategoricalDLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
    delete filename;
  }

  string* filename;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<int> description;
  int numcategories;
  int numsamples;
};

TYPED_TEST_CASE(HDF5CategoricalDLayerTest, TestDtypesAndDevices);

TYPED_TEST(HDF5CategoricalDLayerTest, TestRead) {
  typedef typename TypeParam::Dtype Dtype;
  // Create LayerParameter with the known parameters.
  LayerParameter param;
  param.add_top("data");
  param.add_top("label");

  HDF5CategoricalDParameter* hdf5_categorical_data_param=
    param.mutable_hdf5_categorical_data_param ();
  int batch_size = 20;
  hdf5_categorical_data_param->set_batch_size(batch_size);
  hdf5_categorical_data_param->set_source(*(this->filename));
  // see 'generate_test_data'
  int numcategories=this->numcategories;
  int num_cols =(numcategories+1)*numcategories/2;
  int height = 1;
  int width = 1;

  // Test that the layer setup got the correct parameters.
  HDF5CategoricalDLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), batch_size);
  EXPECT_EQ(this->blob_top_data_->channels(), num_cols);
  EXPECT_EQ(this->blob_top_data_->height(), height);
  EXPECT_EQ(this->blob_top_data_->width(), width);

  EXPECT_EQ(this->blob_top_label_->num(), batch_size);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);

  //helper data to check one-hot-encoding
  vector<int> accummulatedvalues(numcategories);
  accummulatedvalues[0]=0;
  for(int i=1;i<numcategories;++i)
    accummulatedvalues[i]=accummulatedvalues[i-1]+i;

  int numsamples=this->numsamples;
  int datasample,value, idx;  
  int numiterations=6;
  for (int iter = 0; iter < numiterations; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    for (int i = 0; i < batch_size; ++i) {
      datasample=(iter*batch_size+i)%numsamples;
      // verify one-hot-encoded batch data  (see  'generate_test_data')
      for(int j=0;j<numcategories;++j){
	//first data entry corresponding to this category in this row
	idx=i*num_cols+accummulatedvalues[j];
	// check if this category is 'set' for this datasample
	if(datasample%(j+2)){
	  value=datasample%(j+2)-1;
	  // there are j possible values for category j
	  for(int k=0;k<j;++k)
	    if(k==value)
	      ASSERT_EQ(1,this->blob_top_data_->cpu_data()[idx+k]);
	    else
	      ASSERT_EQ(0,this->blob_top_data_->cpu_data()[idx+k]);
	  
	}
	else
	// no value is set for this category
	  for(int k=0;k<j;++k)
	    ASSERT_EQ(0,this->blob_top_data_->cpu_data()[idx+k]);
      }

      //check label value
      ASSERT_EQ((float) (datasample%2),
        this->blob_top_label_->cpu_data()[i]);
    }
  }
}

}  // namespace caffe
