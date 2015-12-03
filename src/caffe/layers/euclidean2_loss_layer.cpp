#include <vector>

#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

namespace caffe {

template <typename Dtype>
void Euclidean2LossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void Euclidean2LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // if(0){
  //   //add some visualization code here
  //   int num = bottom[1]->num();
  //   int nchannel = bottom[1]->channels();
  //   int height = bottom[1]->height();
  //   int width = bottom[1]->width();
  //   LOG(INFO) << "Label shape is " << num << " x " << nchannel << " x " << height << " x " << width;

  //   static int counter = 0;
  //   int stride = 8;
  //   if(nchannel == 15){
  //     for(int n=0; n<num; n++){
  //       for(int c=0; c<nchannel; c++){
  //         int offset = n*nchannel*height*width + c*height*width;
          
  //         Mat label_map = Mat::zeros(height, width, CV_8UC1);
  //         for(int h=0; h<height; h++){
  //           for(int w=0; w<width; w++){
  //             label_map.at<uchar>(h,w) = (int)(bottom[1]->cpu_data()[offset + h*width + w]*255);
  //           }
  //         }
  //         char filename[100];
  //         sprintf(filename, "from_euc_loss2_%04d_part%02d.jpg", counter, c);
  //         //resize(label_map, label_map, Size(), stride, stride, INTER_LINEAR);
  //         applyColorMap(label_map, label_map, COLORMAP_JET);
  //         //addWeighted(label_map, 0.5, img_aug, 0.5, 0.0, label_map);
  //         imwrite(filename, label_map);
  //       }
  //       counter++;
  //     }
  //   }
  //   else {
  //     nchannel = 15;
  //     height = 46;
  //     width = 46;
  //     for(int n=0; n<num; n++){
  //       for(int c=0; c<nchannel; c++){
  //         int offset = n*nchannel*height*width + c*height*width;
          
  //         Mat label_map = Mat::zeros(height, width, CV_8UC1);
  //         for(int h=0; h<height; h++){
  //           for(int w=0; w<width; w++){
  //             label_map.at<uchar>(h,w) = (int)(bottom[1]->cpu_data()[offset + h*width + w]*255);
  //           }
  //         }
  //         char filename[100];
  //         sprintf(filename, "from_euc_loss_%04d_part%02d.jpg", counter, c);
  //         //resize(label_map, label_map, Size(), stride, stride, INTER_LINEAR);
  //         applyColorMap(label_map, label_map, COLORMAP_JET);
  //         //addWeighted(label_map, 0.5, img_aug, 0.5, 0.0, label_map);
  //         imwrite(filename, label_map);
  //       }
  //       counter++;
  //     }
  //   }
  // }

  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void Euclidean2LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(Euclidean2LossLayer);
#endif

INSTANTIATE_CLASS(Euclidean2LossLayer);
REGISTER_LAYER_CLASS(Euclidean2Loss);

}  // namespace caffe
