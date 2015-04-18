// Copyright 2014 Julien Martel

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {

template<typename Dtype>
DataRandTransformLayer<Dtype>::~DataRandTransformLayer<Dtype>() {
}

template<typename Dtype>
void DataRandTransformLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1)<< "Data Rand Transform Layer takes a single blob as input.";
  CHECK_EQ(top.size(), 1) << "Data Rand Transform Layer takes a single blob as output.";

  cv::namedWindow("Test",0);

  // Bottom[0] cause there is only one input blob
  NUM_ = bottom[0]->num();
  CHANNELS_ = bottom[0]->channels();
  HEIGHT_ = bottom[0]->height();
  WIDTH_ = bottom[0]->width();

  // Announce for the top blob layer
  top[0]->Reshape(NUM_,
      CHANNELS_,
      HEIGHT_,
      WIDTH_, this->device_context_
  );

  // Read the layer parameters
  apply_normalization_ = this->layer_param_.apply_normalization();

  apply_mirroring_ = this->layer_param_.apply_mirroring();
  prob_mirroring_ = this->layer_param_.prob_mirroring();

  apply_rot_ = this->layer_param_.apply_rot();
  rot_min_ = this->layer_param_.rot_min();
  rot_max_ = this->layer_param_.rot_max();

  apply_blur_ = this->layer_param_.apply_blur();
  blur_size_ = this->layer_param_.blur_size();
  blur_max_var_ = this->layer_param_.blur_max_var();

  apply_contrast_brightness_ = this->layer_param_.apply_contrast_brightness();
  alpha_ = this->layer_param_.alpha_c();
  beta_ = this->layer_param_.beta_c();

  /*
   LOG(ERROR) << "\nRotation: " << apply_rot_ << ", min: " << rot_min_ << ", max: " << rot_max_
   << "\nBlur: " << apply_blur_ << ", size: " << blur_size_ << ", var: " << blur_max_var_
   << "\nContrast/Brightness: " << apply_contrast_brightness_ << ", alpha: " << alpha_ << ", beta: " << beta_;
   */
  return;
}

template<typename Dtype>
void DataRandTransformLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  //TODO??
}

template<typename Dtype>
void DataRandTransformLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Transform the blob data in an opencv image
  cv::Mat img(bottom[0]->height(), bottom[0]->width(), CV_32FC3);
  cv::Mat imgTransformed(img.rows, img.cols, CV_32FC3);

  // Normalization
  float val;
  std::vector<float> mean(CHANNELS_);
  std::vector<float> std(CHANNELS_);

  // Center of rotation
  cv::Point2f center = cv::Point2f(img.cols / 2, img.rows / 2);

  // Mirroring
  cv::Mat map_x(HEIGHT_, WIDTH_, CV_32FC1);
  cv::Mat map_y(HEIGHT_, WIDTH_, CV_32FC1);
  for (int j = 0; j < HEIGHT_; j++) {
    for (int i = 0; i < WIDTH_; i++) {
      map_x.at<float>(j, i) = WIDTH_ - i;
      map_y.at<float>(j, i) = HEIGHT_ - j;
    }
  }

  for (int n = 0; n < NUM_; n++) {
    // Reinit for normalization
    for (int c = 0; c < CHANNELS_; c++) {
      mean[c] = 0;
      std[c] = 0;
    }

    // Transform the data into an opencv structure to apply transformations
    for (int c = 0; c < CHANNELS_; c++) {
      for (int h = 0; h < HEIGHT_; h++) {
        for (int w = 0; w < WIDTH_; w++) {
          val = (bottom[0]->data_at(n, c, h, w));

          img.at<cv::Vec3f>(h, w)[c] = val;
          mean[c] += val;
        }
      }
    }
    for (int c = 0; c < CHANNELS_; c++) {
      mean[c] = mean[c] / (img.rows * img.cols);
      //LOG(ERROR) << "Mean" << c << "="<< mean[c];
    }

    // Normalize patch-wise
    if (apply_normalization_) {
      for (int h = 0; h < HEIGHT_; h++) {
        for (int w = 0; w < WIDTH_; w++) {
          for (int c = 0; c < CHANNELS_; c++) {
            val = img.at<cv::Vec3f>(h, w)[c];

            std[c] += (mean[c] - val) * (mean[c] - val);
          }
        }
      }
      for (int c = 0; c < CHANNELS_; c++) {
        std[c] = sqrtf(std[c] / (img.rows * img.cols));
        //LOG(ERROR) << "Std" << c << "="<< std[c];
      }

      for (int h = 0; h < HEIGHT_; h++) {
        for (int w = 0; w < WIDTH_; w++) {
          for (int c = 0; c < CHANNELS_; c++) {
            img.at<cv::Vec3f>(h, w)[c] = (img.at<cv::Vec3f>(h, w)[c] - mean[c])
                / std[c];
          }
        }
      }
    }

    // Double mirroring
    if (apply_mirroring_) {
      cv::Scalar color;
      if (apply_normalization_)
        color = cv::Scalar(0, 0, 0);
      else
        color = cv::Scalar(0.5, 0.5, 0.5);

      if (float(rand()) / RAND_MAX < prob_mirroring_) {
        cv::remap(img, imgTransformed, map_x, map_y, CV_INTER_LINEAR,
                  cv::BORDER_CONSTANT, color);
        imgTransformed.copyTo(img);
      }
    }

    // Rotate image
    if (apply_rot_) {
      float angle = rot_min_ + (rot_max_ - rot_min_) * float(rand()) / RAND_MAX;  // [-rot_min ; rot_max]
      cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);

      cv::Scalar color;
      if (apply_normalization_)
        color = cv::Scalar(0, 0, 0);
      else
        color = cv::Scalar(0.5, 0.5, 0.5);

      cv::warpAffine(img, imgTransformed, rot_mat, img.size(),
                     cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT, color);
      imgTransformed.copyTo(img);
    }

    // Blur image
    if (apply_blur_) {
      float s = blur_max_var_ * float(rand()) / RAND_MAX;  // [0.0 ; max_var]
      cv::GaussianBlur(img, img, cv::Size(blur_size_, blur_size_), s);
    }

    // Contrast enhancement
    if (apply_contrast_brightness_) {
      float alpha = (1.0 - alpha_) + (2 * alpha_) * float(rand()) / RAND_MAX;  //[1.0-alpha ; 1.0+alpha]
      float beta = 2 * beta_ * float(rand()) / RAND_MAX - beta_;	//[-beta ; +beta]
      for (int y = 0; y < HEIGHT_; y++) {
        for (int x = 0; x < WIDTH_; x++) {
          for (int c = 0; c < CHANNELS_; c++) {
            img.at<cv::Vec3f>(y, x)[c] = alpha * (img.at<cv::Vec3f>(y, x)[c])
                + beta;
          }
        }
      }
    }

    // === DEBUG
    //cv::imshow("Test",0.5+0.5*img);
    //cv::waitKey(1);

    //Fill back to the blob
    Dtype* data = top[0]->mutable_cpu_data();
    for (int c = 0; c < CHANNELS_; c++) {
      for (int h = 0; h < HEIGHT_; h++) {
        for (int w = 0; w < WIDTH_; w++) {
          *(data + top[0]->offset(n, c, h, w)) = img.at<cv::Vec3f>(h, w)[c];
        }
      }
    }
  }
}

template<typename Dtype>
void DataRandTransformLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // No GPU implementation for now, just apply the CPU transformations
  Forward_cpu(bottom, top);
}

// The backward operations are dummy - they do not carry any computation.
template<typename Dtype>
void DataRandTransformLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void DataRandTransformLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(DataRandTransformLayer);
REGISTER_LAYER_CLASS(DataRandTransform);

}  // namespace caffe
