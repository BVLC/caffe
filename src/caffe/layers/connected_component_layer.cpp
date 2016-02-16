#ifdef USE_OPENCV
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/connected_component_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

// Derived from
// http://nghiaho.com/uploads/code/opencv_connected_component/blob.cpp
template<typename Dtype>
cv::Mat ConnectedComponentLayer<Dtype>::FindBlobs(int maxlabel,
                                                  const cv::Mat &input) {
  // Fill the label_image with the blobs
  cv::Mat label_image;
  input.convertTo(label_image, CV_32SC1);

  int label_count = maxlabel + 1;

  // Segment into label numbers higher than the original label numbers
  for (int y = 0; y < label_image.rows; y++) {
    int *row = reinterpret_cast<int*>(label_image.ptr(y));
    for (int x = 0; x < label_image.cols; x++) {
      // Skip background and already labeled areas
      if (row[x] > maxlabel || row[x] == 0) {
        continue;
      }
      cv::Rect rect;
      cv::floodFill(label_image, cv::Point(x, y), label_count, &rect, 0, 0, 4);
      label_count++;
    }
  }
  return label_image;
}

template<typename Dtype>
void ConnectedComponentLayer<Dtype>::LayerSetUp(
              const vector<Blob<Dtype>*>& bottom,
              const vector<Blob<Dtype>*>& top) {
}

template<typename Dtype>
void ConnectedComponentLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template<typename Dtype>
void ConnectedComponentLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  cv::Mat img(bottom[0]->height(), bottom[0]->width(), CV_8SC1);

  for (int_tp nc = 0; nc < bottom[0]->num() * bottom[0]->channels(); ++nc) {
    int maxlabel = 0;
    for (int_tp y = 0; y < bottom[0]->height(); ++y) {
      for (int_tp x = 0; x < bottom[0]->width(); ++x) {
        int val = bottom_data[nc * bottom[0]->width() * bottom[0]->height()
                                          + bottom[0]->width() * y + x];
        if (val > maxlabel) {
          maxlabel = val;
        }
        img.at<unsigned char>(y, x) = val;
      }
    }
    cv::Mat seg = FindBlobs(maxlabel, img);
#pragma omp parallel for
    for (int_tp y = 0; y < seg.rows; ++y) {
      for (int_tp x = 0; x < seg.cols; ++x) {
        top_data[nc * bottom[0]->width() * bottom[0]->height()
            + bottom[0]->width() * y + x] = seg.at<int>(y, x);
      }
    }
  }
}

template<typename Dtype>
void ConnectedComponentLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // Nothing to do, don't backpropagate to labels
  return;
}

INSTANTIATE_CLASS(ConnectedComponentLayer);
REGISTER_LAYER_CLASS(ConnectedComponent);

}  // namespace caffe
#endif  // USE_OPENCV
