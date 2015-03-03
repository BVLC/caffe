#ifndef IM_TRANSFORMS_HPP
#define IM_TRANSFORMS_HPP
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>

#if CV_VERSION_MAJOR == 3
#include <opencv2/imgcodecs/imgcodecs.hpp>
#define CV_GRAY2BGR cv::COLOR_GRAY2BGR
#define CV_BGR2GRAY cv::COLOR_BGR2GRAY
#define CV_BGR2YCrCb cv::COLOR_BGR2YCrCb
#define CV_YCrCb2BGR cv::COLOR_YCrCb2BGR
#define CV_IMWRITE_JPEG_QUALITY cv::IMWRITE_JPEG_QUALITY
#define CV_LOAD_IMAGE_COLOR cv::IMREAD_COLOR
#endif

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/rng.hpp"


namespace caffe {
  template <typename T>
  bool is_border(cv::Mat& edge, T color) {
    cv::Mat im = edge.clone().reshape(0, 1);
    bool res = true;
    for (int i = 0; i < im.cols; ++i)
      res &= (color == im.at<T>(0, i));

    return res;
  }

  /**
 * Function to auto-cropping image
 *
 * Parameters:
 *   src   The source image
 *   point any point - to define matrix type
 */

  template <typename T>
  cv::Rect CropMask(cv::Mat& src, T point, int padding = 2) {
    cv::Rect win(0, 0, src.cols, src.rows);

    std::vector<cv::Rect> edges;
    edges.push_back(cv::Rect(0, 0, src.cols, 1));
    edges.push_back(cv::Rect(src.cols-2, 0, 1, src.rows));
    edges.push_back(cv::Rect(0, src.rows-2, src.cols, 1));
    edges.push_back(cv::Rect(0, 0, 1, src.rows));

    cv::Mat edge;
    int nborder = 0;
    T color = src.at<T>(0, 0);

    for (unsigned int i = 0; i < edges.size(); ++i) {
        edge = src(edges[i]);
        nborder += is_border(edge, color);
      }

    if (nborder < 4) {
        return win;
      }

    bool next;

    do {
        edge = src(cv::Rect(win.x, win.height-2, win.width, 1));
        next = is_border(edge, color);
        if (next)
          win.height--;
      }  while (next && (win.height > 0));

    do {
        edge = src(cv::Rect(win.width-2, win.y, 1, win.height));
        next = is_border(edge, color);
        if (next)
          win.width--;
      }  while (next && (win.width > 0));

    do {
        edge = src(cv::Rect(win.x, win.y, win.width, 1));
        next = is_border(edge, color);
        if (next)
          win.y++, win.height--;
      }  while (next && (win.y <= src.rows));

    do {
        edge = src(cv::Rect(win.x, win.y, 1, win.height));
        next = is_border(edge, color);
        if (next)
          win.x++, win.width--;
      }  while (next && (win.x <= src.cols));

    // add padding
    if (win.x > padding)
      win.x -= padding;

    if (win.y > padding)
      win.y -= padding;

    if ((win.width+win.x+padding) < src.cols)
      win.width +=padding;

    if ((win.height+win.y+padding) < src.rows)
      win.height +=padding;

    return win;
  }

  void fillEdgeImage(cv::Mat edgesIn, cv::Mat& filledEdgesOut);

  void CenterObjectAndFillBg(const cv::Mat & in_img,
      cv::Mat & out_img, const bool fill_bg = false);

  cv::Mat AspectKeepingResize(const cv::Mat &in_img,
      const int new_width, const int new_height,
      const int pad_type = cv::BORDER_CONSTANT,
      const cv::Scalar pad_val = cv::Scalar(0, 0, 0),
      const int interp_mode = cv::INTER_LINEAR);

  cv::Mat AspectKeepingResizeBySmall(const cv::Mat &in_img,
      const int new_width, const int new_height,
      const int interp_mode = cv::INTER_LINEAR);


  cv::Mat ApplyResize(const cv::Mat &in_img, const ResizeParameter param);

  void constantNoise(cv::Mat &image, const int n, const std::vector<uchar> val);

  cv::Mat ApplyNoise(const cv::Mat &in_img, const NoiseParameter param);

}  // namespace caffe
#endif  // IM_TRANSFORMS_HPP
