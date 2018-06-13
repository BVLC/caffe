/*
TODO:
- only load parts of the file, in accordance with a prototxt param "max_mem"
*/

#include <stdint.h>
#include <vector>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/layers/hdf5_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void HDF5DataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.hdf5_data_param().batch_size();
  bool has_transform = this->layer_param_.has_transform_param();
  bool is_image = this->layer_param_.hdf5_data_param().image();
  for (int i = 0; i < batch_size; ++i, ++current_row_) {
    if (current_row_ == hdf_blobs_[0]->shape(0)) {
      if (num_files_ > 1) {
        current_file_ += 1;
        if (current_file_ == num_files_) {
          current_file_ = 0;
          if (this->layer_param_.hdf5_data_param().shuffle()) {
            std::random_shuffle(file_permutation_.begin(),
                                file_permutation_.end());
          }
          DLOG(INFO) << "Looping around to first file.";
        }
        LoadHDF5FileData(
            hdf_filenames_[file_permutation_[current_file_]].c_str());
      }
      current_row_ = 0;
      if (this->layer_param_.hdf5_data_param().shuffle())
        std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
    }
#ifdef USE_OPENCV
    if (is_image && has_transform) {
      for (int j = 0; j < this->layer_param_.top_size(); ++j)
	{
	  int data_dim = top[j]->count() / top[j]->shape(0);
	  if (j == 0) { // data, get image + apply transforms + copy to top blob
	    vector<int> bshape = hdf_blobs_[j]->shape();
	    int height = bshape[2];
	    int width = bshape[3];
	    vector<Dtype> img_data;
	    img_data.insert(img_data.begin(),&hdf_blobs_[j]->cpu_data()[data_permutation_[current_row_]*data_dim],
	    		    &hdf_blobs_[j]->cpu_data()[data_permutation_[current_row_]*data_dim]+data_dim);
            cv::Mat cv_img(height,width,CV_8UC3);
	    for (int c=0;c<3;c++)
	      {
		for (int h=0;h<height;h++)
		  {
		    for (int w=0;w<width;w++)
		      {
			cv_img.at<cv::Vec3b>(cv::Point(w,h))[c] = static_cast<uint8_t>(img_data.at(c*width*height+h*width+w));
		      }
		  }
	      }
	    Blob<Dtype> transformed_data;
	    transformed_data.Reshape(1,3,height,width);
	    this->data_transformer_->Transform(cv_img, &transformed_data);
	    caffe_copy(data_dim, transformed_data.mutable_cpu_data(), &top[j]->mutable_gpu_data()[i * data_dim]);
	  }
	  else { //label
	    caffe_copy(data_dim,
		       &hdf_blobs_[j]->cpu_data()[data_permutation_[current_row_]
						  * data_dim], &top[j]->mutable_gpu_data()[i * data_dim]);
	  }
	}
    }
    else {
#endif
    for (int j = 0; j < this->layer_param_.top_size(); ++j) {
      int data_dim = top[j]->count() / top[j]->shape(0);
      caffe_copy(data_dim,
          &hdf_blobs_[j]->cpu_data()[data_permutation_[current_row_]
            * data_dim], &top[j]->mutable_gpu_data()[i * data_dim]);
    }
#ifdef USE_OPENCV
    }
#endif
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(HDF5DataLayer);

}  // namespace caffe
