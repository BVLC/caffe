#include <stdint.h>
#include <string>
#include <vector>
#include <fstream>  // NOLINT(readability/streams)

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/common.hpp"
#include "caffe/ImageFeatureDataLayer.h"
namespace caffe {

template <typename Dtype>
ImageFeatureDataLayer<Dtype>::~ImageFeatureDataLayer<Dtype>() { }

// Load data and label from HDF5 filename into the class property blobs.
template <typename Dtype>
void ImageFeatureDataLayer<Dtype>::AddImageFeatureData(const std::list<cv::Mat>& src, const std::list<std::vector<float> >& features)
{
  int row = src.size();
  int cols = src.begin()->channels();
  int height = src.begin()->rows;
  int width = src.begin()->cols;
  data_blob_.Reshape(row, cols, height, width);
  int labelsize = features.begin()->size();
  label_blob_.Reshape(row, labelsize, 1, 1);
  
  //num_files = row;
  current_file_ = 0;

  CHECK_EQ(data_blob_.num(), label_blob_.num());
  //LOG(INFO) << "Successully loaded " << data_blob_.num() << " rows";
}

template <typename Dtype>
void ImageFeatureDataLayer<Dtype>::AddImageFeatureData(const std::list<cv::Mat>& src)
{
  int row = src.size();
  int cols = src.begin()->channels();
  int height = src.begin()->rows;
  int width = src.begin()->cols;  
  data_.reset(new Dtype[sizeof(Dtype) * cols * row * height * width]);
  data_blob_.Reshape(row, cols, height, width);
  Dtype* ptr = data_.get();  
  //LOG(INFO) << "Successully loaded " << row << "," << cols << "," << height << "," <<width;
  int blocksize = height * width;
  for (std::list<cv::Mat>::const_iterator it = src.begin(); it != src.end(); it++)
  {
    std::vector<cv::Mat> planes(cols);
    if(cols == 3)
    {
      cv::split(*it, planes);
    }
    else
    {
      planes[0] = *it;
    }
    for (int c = 0; c < cols; c++)
    {
      cv::Mat aplane = planes[c];
      memcpy((void*)ptr, (void*)aplane.ptr<Dtype>(), blocksize  * sizeof(Dtype));
      ptr += blocksize;
    }
  }
  data_blob_.set_cpu_data(data_.get());

  int feature_size = this->layer_param_.image_feature_data_param().feature_size();
  label_blob_.Reshape(row, feature_size, 1, 1);

  //num_files = row;
  current_file_ = 0;

  CHECK_EQ(data_blob_.num(), label_blob_.num());
  //LOG(INFO) << "Successully loaded " << data_blob_.num() << " rows";
}

template <typename Dtype>
void ImageFeatureDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  int c = this->layer_param_.image_feature_data_param().channel();
  int w = this->layer_param_.image_feature_data_param().width();
  int h = this->layer_param_.image_feature_data_param().height();
  int feature_size = this->layer_param_.image_feature_data_param().feature_size();
  // Reshape blobs.
  const int batch_size = 1;
  (*top)[0]->Reshape(batch_size, c, h, w);
  (*top)[1]->Reshape(batch_size, feature_size, 1, 1);
  LOG(INFO) << "output data size: " << batch_size << "," << c << "," << h << "," << w;
}

template <typename Dtype>
Dtype ImageFeatureDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const int batch_size = 1;//this->layer_param_.hdf5_data_param().batch_size();
  const int data_count = (*top)[0]->count() / (*top)[0]->num();
  const int label_data_count = (*top)[1]->count() / (*top)[1]->num();  
  current_row_ = 0;
  //LOG(INFO) << "Loop iter: "<< current_row_ <<"," << data_count << "," << label_data_count;
  for (int i = 0; i < batch_size; ++i, ++current_row_) {    
    memcpy(&(*top)[0]->mutable_cpu_data()[i * data_count],
           &data_blob_.cpu_data()[current_row_ * data_count],
           sizeof(Dtype) * data_count);
    memcpy(&(*top)[1]->mutable_cpu_data()[i * label_data_count],
            &label_blob_.cpu_data()[current_row_ * label_data_count],
            sizeof(Dtype) * label_data_count);
  }
#if 0
  int w = (*top)[0]->width();
  int h = (*top)[0]->height();;
  //LOG(INFO) << "result image size:" << result[iresult]->num() << " " << h << "x"<< w <<" channels: " << result[iresult]->channels();
  Dtype* data = (*top)[0]->mutable_cpu_data();
  cv::Mat img = cv::Mat(h,w,CV_8UC3);
  for(int y = 0; y < h; y++){
    for(int x = 0; x < w; x++){
      int r = (int)(data[ 0 * w * h + y * w + x] * 255);
      int g = (int)(data[ 1 * w * h + y * w + x] * 255);
      int b = (int)(data[ 2 * w * h + y * w + x] * 255);
      //if( r == g && g == b)
      //LOG(INFO) << "gray image";
      r = std::min(std::max(r,0),255);
      g = std::min(std::max(g,0),255);
      b = std::min(std::max(b,0),255);

      img.at<cv::Vec3b>(y,x) = cv::Vec3b(r,g,b);
    }// end x
  }// end y

  char fname[255];
  cv::Mat imorg = img.clone();
  sprintf(fname, "predict_process/img_%d.jpg", 0);
  //cv::imwrite(fname, imorg);
  IplImage i1 = imorg;
  cvSaveImage(fname, &i1);
#endif
  return Dtype(0.);
}

INSTANTIATE_CLASS(ImageFeatureDataLayer);

}  // namespace caffe
