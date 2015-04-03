#include <boost/thread/tss.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im_transforms.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#if CV_VERSION_MAJOR == 3
#define CV_THRESH_BINARY_INV cv::THRESH_BINARY_INV
#define CV_THRESH_OTSU cv::THRESH_OTSU
#endif

namespace caffe {
  void unsharpMask(cv::Mat& im) {
    cv::Mat tmp;
    cv::GaussianBlur(im, tmp, cv::Size(5, 5), 5);
    cv::addWeighted(im, 1.5, tmp, -0.5, 0, im);
  }

  cv::Mat colorReduce(const cv::Mat& image, int div = 64) {
    cv::Mat out_img;
    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.data;
    const int div_2 = div / 2;
    for ( int i = 0; i < 256; ++i ) {
      p[i] = i / div * div + div_2;
    }
    cv::LUT(image, lookUpTable, out_img);
    return out_img;
  }

  void fillEdgeImage(cv::Mat edgesIn, cv::Mat& filledEdgesOut) {
    cv::Mat edgesNeg = edgesIn.clone();
    cv::floodFill(edgesNeg, cv::Point(0, 0), cv::Scalar(255, 255, 255));
    cv::floodFill(edgesNeg,
        cv::Point(edgesIn.cols-1, edgesIn.rows - 1),
        cv::Scalar(255, 255, 255));
    cv::floodFill(edgesNeg, cv::Point(0, edgesIn.rows - 1),
        cv::Scalar(255, 255, 255));
    cv::floodFill(edgesNeg, cv::Point(edgesIn.cols - 1, 0),
        cv::Scalar(255, 255, 255));
    cv::bitwise_not(edgesNeg, edgesNeg);
    filledEdgesOut = (edgesNeg | edgesIn);
    return;
  }

  void CenterObjectAndFillBg(const cv::Mat & in_img,
     cv::Mat & out_img, const bool fill_bg) {
    cv::Mat mask, crop_mask;
    if (in_img.channels() > 1) {
      cv::Mat in_img_gray;
      cv::cvtColor(in_img, in_img_gray, CV_BGR2GRAY);
      cv::threshold(in_img_gray, mask, 0, 255,
          CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
    } else {
      cv::threshold(in_img, mask, 0, 255,
          CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
    }
    cv::Rect crop_rect = CropMask(mask, mask.at<uchar>(0, 0), 2);

    if (fill_bg) {
      cv::Mat temp_img = in_img(crop_rect);
      fillEdgeImage(mask, mask);
      crop_mask = mask(crop_rect).clone();
      out_img = cv::Mat::zeros(crop_rect.size(), in_img.type());
      temp_img.copyTo(out_img, crop_mask);
    } else {
      out_img = in_img(crop_rect).clone();
    }
  }

  cv::Mat AspectKeepingResize(const cv::Mat &in_img,
      const int new_width, const int new_height,
      const int pad_type,  const cv::Scalar pad_val,
      const int interp_mode) {
    cv::Mat img_resized;
    double orig_aspect = static_cast<double> (in_img.cols) /
        static_cast<double> (in_img.rows);

    double new_aspect = static_cast<double> (new_width) /
        static_cast<double> (new_height);

    if (orig_aspect > new_aspect) {
      cv::resize(in_img, img_resized,
          cv::Size(new_width, floor((double)new_width/orig_aspect)),
          0, 0, interp_mode);
      cv::Size resSize = img_resized.size();
      int padding = floor((new_height - resSize.height) / 2.0);
      cv::copyMakeBorder(img_resized, img_resized, padding,
          new_height - resSize.height - padding, 0, 0,  pad_type, pad_val);

    } else {
      cv::resize(in_img, img_resized,
          cv::Size(floor(orig_aspect*new_height), new_height),
          0, 0, interp_mode);
      cv::Size resSize = img_resized.size();
      int padding = floor((new_width - resSize.width) / 2.0);
      cv::copyMakeBorder(img_resized, img_resized, 0, 0, padding,
          new_width - resSize.width - padding, pad_type, pad_val);
    }
    return img_resized;
  }
  cv::Mat AspectKeepingResizeBySmall(const cv::Mat &in_img,
      const int new_width,
      const int new_height,
      const int interp_mode) {
    cv::Mat img_resized;
    double orig_aspect = static_cast<double> (in_img.cols) /
        static_cast<double> (in_img.rows);

    double new_aspect = static_cast<double> (new_width) /
        static_cast<double> (new_height);
    if (orig_aspect < new_aspect) {
      cv::resize(in_img, img_resized,
          cv::Size(new_width, floor((double) new_width/orig_aspect)),
          0, 0, interp_mode);

    } else {
      cv::resize(in_img, img_resized,
          cv::Size(floor(orig_aspect*new_height), new_height),
          0, 0, interp_mode);
    }
    return img_resized;
  }

  void constantNoise(cv::Mat &image, const int n,
      const std::vector<uchar> val) {
    const int cols = image.cols;
    const int rows = image.rows;

    if (image.channels() == 1) {
      for (int k = 0; k < n; ++k) {
        const int i = caffe_rng_rand()%cols;
        const int j = caffe_rng_rand()%rows;
        uchar* ptr = image.ptr<uchar>(j);
        ptr[i]= val[0];
      }
    } else if (image.channels() == 3) {  // color image
      for (int k = 0; k < n; ++k) {
        const int i = caffe_rng_rand()%cols;
        const int j = caffe_rng_rand()%rows;
        cv::Vec3b* ptr = image.ptr<cv::Vec3b>(j);
        (ptr[i])[0] = val[0];
        (ptr[i])[1] = val[1];
        (ptr[i])[2] = val[2];
      }
    }
  }

  // don't overflow uchar
  uchar inline applyNormalizeNoise(uchar pixel, float noise) {
    return (noise == 0) * pixel
        + (noise > 0) * (pixel + noise * (255.0 - pixel) /128.0)
        + (noise < 0) * (pixel + noise * pixel/128.0);
  }

  boost::thread_specific_ptr<cv::Mat> noise_buf;
  void gaussianNoise(const cv::Mat &image, const float fraction,
      const float stddev) {
    const int cols = image.cols;
    const int rows = image.rows;
    if (!noise_buf.get()) {
      noise_buf.reset(new cv::Mat());
    }
    noise_buf->create(cv::Size(cols, rows), image.type());
    cv::randn(*noise_buf, 0, stddev);
    if (noise_buf->channels() == 1) {
      for (int i = 0; i < rows; i++) {
        uchar* ptr = noise_buf->ptr<uchar>(i);
        uchar* image_ptr = image.ptr<uchar>(i);
        for (int j = 0; j < cols; j++) {
          int chosen = (caffe_rng_rand() % 1000) / 1000.0 < fraction;
          image_ptr[j] = applyNormalizeNoise(image_ptr[j], ptr[j] * chosen);
        }
      }
    } else if (noise_buf->channels() == 3) {
      for (int i = 0; i < rows; i++) {
        cv::Vec3b* ptr = noise_buf->ptr<cv::Vec3b>(i);
        cv::Vec3b* image_ptr = image.ptr<cv::Vec3b>(i);
        for (int j = 0; j < cols; j++) {
          int chosen = (caffe_rng_rand() % 1000) / 1000.0 < fraction;
          image_ptr[j][0] = applyNormalizeNoise(
              image_ptr[j][0], ptr[j][0] * chosen);
          image_ptr[j][1] = applyNormalizeNoise(
              image_ptr[j][1], ptr[j][1] * chosen);
          image_ptr[j][2] = applyNormalizeNoise(
              image_ptr[j][1], ptr[j][1] * chosen);
        }
      }
    }
//    image += *noise_buf;
  }

  cv::Mat ApplyResize(const cv::Mat &in_img, const ResizeParameter param) {
    cv::Mat out_img;

    // Reading parameters
    const int new_height = param.height();
    const int new_width = param.width();

    int pad_mode;
    switch (param.pad_mode()) {
      case ResizeParameter_Pad_mode_CONSTANT:
      {
        pad_mode = cv::BORDER_CONSTANT;
        break;
      }
      case ResizeParameter_Pad_mode_MIRRORED:
      {
        pad_mode = cv::BORDER_REFLECT101;
        break;
      }
      case ResizeParameter_Pad_mode_REPEAT_NEAREST:
      {
        pad_mode = cv::BORDER_REPLICATE;
        break;
      }
    }

    int interp_mode;
    switch (param.interp_mode()) {
      case ResizeParameter_Interp_mode_AREA:
      {
        interp_mode = cv::INTER_AREA;
        break;
      }
      case ResizeParameter_Interp_mode_CUBIC:
      {
        interp_mode = cv::INTER_CUBIC;
        break;
      }
      case ResizeParameter_Interp_mode_LINEAR:
      {
        interp_mode = cv::INTER_LINEAR;
        break;
      }
      case ResizeParameter_Interp_mode_NEAREST:
      {
        interp_mode = cv::INTER_NEAREST;
        break;
      }
      case ResizeParameter_Interp_mode_LANCZOS4:
      {
        interp_mode = cv::INTER_LANCZOS4;
        break;
      }
    }

    std::vector<double> pad_values;
    const int img_channels = in_img.channels();
    if (param.pad_value_size() > 0) {
      CHECK(param.pad_value_size() == 1 ||
          param.pad_value_size() == img_channels )
      << "Specify either 1 pad_value or as many as channels: " << img_channels;

      for (int i = 0; i < param.pad_value_size(); i++) {
        pad_values.push_back(param.pad_value(i));
      }
      if (img_channels > 1 && param.pad_value_size() == 1) {
        // Replicate the pad_value for simplicity
        for (int c = 1; c < img_channels; ++c) {
          pad_values.push_back(pad_values[0]);
        }
      }
    }
    // Centering object if on monotonic background
    if (param.center_object()) {
      CenterObjectAndFillBg(in_img, out_img, 0);
    } else {
      out_img = in_img;
    }

    const float max_angle = param.max_angle();
    if (max_angle > 0) {
      float angle = 0.0;
      caffe_rng_uniform(1, -max_angle, max_angle, &angle);
      cv::Mat matRotation = cv::getRotationMatrix2D(
          cv::Point(out_img.cols / 2, out_img.rows / 2), (angle), 1.0);
      cv::Mat temp_img;
      cv::warpAffine(out_img, temp_img, matRotation, out_img.size(),
          interp_mode, pad_mode,
          cv::Scalar(pad_values[0], pad_values[1], pad_values[2]));
      out_img = temp_img;
    }

    switch (param.resize_mode()) {
      case ResizeParameter_Resize_mode_WARP:
      {
        cv::resize(out_img, out_img,
            cv::Size(new_width, new_height), 0, 0, interp_mode);
        break;
      }
      case ResizeParameter_Resize_mode_FIT_LARGE_SIZE_AND_PAD:
      {
        out_img = AspectKeepingResize(out_img, new_width, new_height, pad_mode,
            cv::Scalar(pad_values[0], pad_values[1], pad_values[2]),
            interp_mode);
        break;
      }
      case ResizeParameter_Resize_mode_FIT_SMALL_SIZE:
      {
        out_img = AspectKeepingResizeBySmall(out_img,
            new_width, new_height, interp_mode);
        break;
      }
    }
    return  out_img;
  }

  cv::Mat ApplyNoise(const cv::Mat &in_img, const NoiseParameter param) {
    cv::Mat out_img;

    if (param.decolorize()) {
      cv::Mat grayscale_img;
      cv::cvtColor(in_img, grayscale_img, CV_BGR2GRAY);
      cv::cvtColor(grayscale_img, out_img,  CV_GRAY2BGR);
    } else {
      out_img = in_img;
    }

    if (param.gauss_blur()) {
      cv::GaussianBlur(out_img, out_img, cv::Size(7, 7), 1.5);
    }

    if (param.hist_eq()) {
      if (out_img.channels() > 1) {
        cv::Mat ycrcb_image;
        cv::cvtColor(out_img, ycrcb_image, CV_BGR2YCrCb);
        // Extract the L channel
        std::vector<cv::Mat> ycrcb_planes(3);
        cv::split(ycrcb_image, ycrcb_planes);
        // now we have the L image in ycrcb_planes[0]
        cv::Mat dst;
        cv::equalizeHist(ycrcb_planes[0], dst);
        ycrcb_planes[0] = dst;
        cv::merge(ycrcb_planes, ycrcb_image);
        // convert back to RGB
        cv::cvtColor(ycrcb_image, out_img, CV_YCrCb2BGR);

      } else {
        cv::Mat temp_img;
        cv::equalizeHist(out_img, temp_img);
        out_img = temp_img;
      }
    }

    if (param.clahe()) {
      cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
      clahe->setClipLimit(4);

      if (out_img.channels() > 1) {
        cv::Mat ycrcb_image;
        cv::cvtColor(out_img, ycrcb_image, CV_BGR2YCrCb);
        // Extract the L channel
        std::vector<cv::Mat> ycrcb_planes(3);
        cv::split(ycrcb_image, ycrcb_planes);
        // now we have the L image in ycrcb_planes[0]
        cv::Mat dst;
        clahe->apply(ycrcb_planes[0], dst);
        ycrcb_planes[0] = dst;
        cv::merge(ycrcb_planes, ycrcb_image);
        // convert back to RGB
        cv::cvtColor(ycrcb_image, out_img, CV_YCrCb2BGR);

      } else {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(4);
        cv::Mat temp_img;
        clahe->apply(out_img, temp_img);
        out_img = temp_img;
      }
    }

    if (param.jpeg() > 0) {
      std::vector<uchar> buf;
      std::vector<int> params;
      params.push_back(CV_IMWRITE_JPEG_QUALITY);
      params.push_back(param.jpeg());
      cv::imencode(".jpg", out_img, buf, params);
      out_img = cv::imdecode(buf, CV_LOAD_IMAGE_COLOR);
    }

    if (param.erode()) {
      cv::Mat element = cv::getStructuringElement(
          2, cv::Size(3, 3), cv::Point(1, 1));
      cv::erode(out_img, out_img, element);
    }

    if (param.posterize()) {
      cv::Mat tmp_img;
      tmp_img = colorReduce(out_img);
      out_img = tmp_img;
    }

    if (param.inverse()) {
      cv::Mat tmp_img;
      cv::bitwise_not(out_img, tmp_img);
      out_img = tmp_img;
    }

    std::vector<uchar> noise_values;
    if (param.saltpepper_param().value_size() > 0) {
      CHECK(param.saltpepper_param().value_size() == 1
          || param.saltpepper_param().value_size() == out_img.channels())
       << "Specify either 1 pad_value or as many as channels: "
       << out_img.channels();

      for (int i = 0; i < param.saltpepper_param().value_size(); i++) {
        noise_values.push_back(uchar(param.saltpepper_param().value(i)));
      }
      if (out_img.channels()  > 1
          && param.saltpepper_param().value_size() == 1) {
        // Replicate the pad_value for simplicity
        for (int c = 1; c < out_img.channels(); ++c) {
          noise_values.push_back(uchar(noise_values[0]));
        }
      }
    }
    if (param.saltpepper()) {
      const int noise_pixels_num =
          floor(param.saltpepper_param().fraction()
              * out_img.cols * out_img.rows);
      switch (param.saltpepper_param().type()) {
        case SaltPepperParameter_SaltType_absolute:
          constantNoise(out_img, noise_pixels_num, noise_values);
          break;
        case SaltPepperParameter_SaltType_relative:
          gaussianNoise(out_img,
              param.saltpepper_param().fraction(), noise_values[0]);
      }
    }
    return  out_img;
  }
}  // namespace caffe
