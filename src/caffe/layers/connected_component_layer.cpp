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
template<typename Dtype, typename MItype, typename MOtype>
cv::Mat ConnectedComponentLayer<Dtype, MItype, MOtype>::FindBlobs(int maxlabel,
                                                  const cv::Mat &input) {
  // Fill the label_image with the blobs
  cv::Mat label_image;
  input.convertTo(label_image, CV_32SC1);

  int label_count = maxlabel + 1;

  // Segment into label numbers higher than the original label numbers
  for (int Y = 0; Y < label_image.rows; Y++) {
    int *row = reinterpret_cast<int*>(label_image.ptr(Y));
    for (int X = 0; X < label_image.cols; X++) {
      // Skip background and already labeled areas
      if (row[X] > maxlabel || row[X] == 0) {
        continue;
      }
      cv::Rect rect;
      cv::floodFill(label_image, cv::Point(X, Y), label_count, &rect, 0, 0, 4);
      label_count++;
    }
  }
  return label_image;
}

template<typename Dtype, typename MItype, typename MOtype>
void ConnectedComponentLayer<Dtype, MItype, MOtype>::LayerSetUp(
              const vector<Blob<MItype>*>& bottom,
              const vector<Blob<MOtype>*>& top) {
  this->InitializeQuantizers(bottom, top);
}

template<typename Dtype, typename MItype, typename MOtype>
void ConnectedComponentLayer<Dtype, MItype, MOtype>::Reshape(
    const vector<Blob<MItype>*>& bottom, const vector<Blob<MOtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template<typename Dtype, typename MItype, typename MOtype>
void ConnectedComponentLayer<Dtype, MItype, MOtype>::Forward_cpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  cv::Mat img(bottom[0]->height(), bottom[0]->width(), CV_8SC1);

  for (int_tp nc = 0; nc < bottom[0]->num() * bottom[0]->channels(); ++nc) {
    int maxlabel = 0;
    for (int_tp Y = 0; Y < bottom[0]->height(); ++Y) {
      for (int_tp X = 0; X < bottom[0]->width(); ++X) {
        int val = bottom_data[nc * bottom[0]->width() * bottom[0]->height()
                                          + bottom[0]->width() * Y + X];
        if (val > maxlabel) {
          maxlabel = val;
        }
        img.at<unsigned char>(Y, X) = val;
      }
    }
    cv::Mat seg = FindBlobs(maxlabel, img);
    for (int_tp Y = 0; Y < seg.rows; ++Y) {
      for (int_tp X = 0; X < seg.cols; ++X) {
        top_data[nc * bottom[0]->width() * bottom[0]->height()
            + bottom[0]->width() * Y + X] = seg.at<int>(Y, X);
      }
    }
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void ConnectedComponentLayer<Dtype, MItype, MOtype>::Backward_cpu(
    const vector<Blob<MOtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  // Nothing to do, don't backpropagate to labels
  return;
}

INSTANTIATE_CLASS_3T_GUARDED(ConnectedComponentLayer,
                             (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(ConnectedComponentLayer,
                             (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(ConnectedComponentLayer,
                             (double), (double), (double));

REGISTER_LAYER_CLASS(ConnectedComponent);
REGISTER_LAYER_CLASS_INST(ConnectedComponent, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(ConnectedComponent, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(ConnectedComponent, (double), (double), (double));

}  // namespace caffe
#endif  // USE_OPENCV
