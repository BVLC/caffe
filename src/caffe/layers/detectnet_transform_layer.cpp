#ifdef USE_OPENCV

#include "caffe/layers/detectnet_transform_layer.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <boost/array.hpp>
#include <boost/foreach.hpp>
#include <boost/static_assert.hpp>

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "caffe/util/detectnet_coverage.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

using namespace cv;  // NOLINT(build/namespaces)
using boost::array;
#define foreach_ BOOST_FOREACH


namespace caffe {


struct AugmentSelection {
    bool flip;
    float degree;
    Point crop;
    float scale;
    float hue_rotation;
    float saturation;
};


template<typename Dtype>
DetectNetTransformationLayer<Dtype>::DetectNetTransformationLayer(
    const LayerParameter& param) :
    Layer<Dtype>(param),
    a_param_(param.detectnet_augmentation_param()),
    g_param_(param.detectnet_groundtruth_param()),
    t_param_(param.transform_param()),
    coverage_(CoverageGenerator<Dtype>::create(
        param.detectnet_groundtruth_param())),
    phase_(param.phase()),
    rng_(new Caffe::RNG(caffe_rng_rand())) {}


template <typename Dtype>
void DetectNetTransformationLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // retrieve mean image or values:
  if (t_param_.has_mean_file()) {
    size_t image_x = bottom[0]->width();
    size_t image_y = bottom[0]->height();

    retrieveMeanImage(Size(image_x, image_y));

  } else if (t_param_.mean_value_size() != 0) {
    retrieveMeanChannels();
  }
}


template<typename Dtype>
void DetectNetTransformationLayer<Dtype>::retrieveMeanImage(Size dimensions) {
  CHECK(t_param_.has_mean_file());

  const string& mean_file = t_param_.mean_file();
  BlobProto blob_proto;
  Blob<Dtype> data_mean;

  ReadProtoFromBinaryFileOrDie(mean_file, &blob_proto);
  data_mean.FromProto(blob_proto);
  data_mean_ = blobToMats(data_mean).at(0);

  // resize, if dimensions were defined:
  if (dimensions.area() > 0) {
    resize(data_mean_, data_mean_, dimensions, cv::INTER_CUBIC);
  }
  // scale from 0..255 to 0..1:
  data_mean_ /= Dtype(UINT8_MAX);
}


template<typename Dtype>
void DetectNetTransformationLayer<Dtype>::retrieveMeanChannels() {
  switch (t_param_.mean_value_size()) {
    case 1:
      mean_values_.fill(t_param_.mean_value(0) / Dtype(UINT8_MAX));
      break;
    case 3:
      for (size_t iChannel = 0; iChannel != 3; ++iChannel) {
        mean_values_[iChannel] =
            t_param_.mean_value(iChannel) / Dtype(UINT8_MAX);
      }
      break;
    case 0:
    default:
      break;
  }
}


template<typename Dtype>
void DetectNetTransformationLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top
) {
  // accept only three channel images:
  CHECK_EQ(bottom[0]->channels(), 3);
  // accept only equal numbers of labels and images:
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  // resize mean image if it exists:
  if (t_param_.has_mean_file()) {
    size_t image_x = bottom[0]->width();
    size_t image_y = bottom[0]->height();
    retrieveMeanImage(Size(image_x, image_y));
  }
  // resize image output layer: (never changes /wrt input blobs)
  top[0]->Reshape(
      bottom[0]->num(),
      bottom[0]->channels(),
      g_param_.image_size_y(),
      g_param_.image_size_x());
  // resize tensor output layer: (never changes /wrt input blobs)
  Vec3i tensorDimensions = coverage_->dimensions();
  top[1]->Reshape(
      bottom[1]->num(),
      tensorDimensions(0),
      tensorDimensions(1),
      tensorDimensions(2));
}


template<typename Dtype>
typename DetectNetTransformationLayer<Dtype>::Mat3v
DetectNetTransformationLayer<Dtype>::dataToMat(
    const Dtype* _data,
    Size dimensions
) const {
  // NOLINT_NEXT_LINE(whitespace/line_length)
  // CODE FROM https://github.com/BVLC/caffe/blob/4874c01487/examples/cpp_classification/classification.cpp#L125-137
  // The format of the mean file is planar 32-bit float BGR or grayscale.
  vector<Mat> channels; channels.reserve(3);

  Dtype* data = const_cast<Dtype*>(_data);
  for (size_t iChannel = 0; iChannel != 3; ++iChannel) {
    // Extract an individual channel. This does not perform a datacopy, so doing
    //  this in a performance critical location should be fine.
    Mat channel(dimensions, cv::DataType<Dtype>::type, data);
    channels.push_back(channel);
    data += dimensions.area();
  }

  // Merge the separate channels into a single image.
  Mat3v result;
  merge(channels, result);

  return result;
}


template<typename Dtype>
vector<typename DetectNetTransformationLayer<Dtype>::Mat3v>
DetectNetTransformationLayer<Dtype>::blobToMats(
    const Blob<Dtype>& images
) const {
  CHECK_EQ(images.channels(), 3);
  vector<Mat3v> result; result.reserve(images.num());

  for (size_t iImage = 0; iImage != images.num(); ++iImage) {
    const Dtype* image_data = &images.cpu_data()[
        images.offset(iImage, 0, 0, 0)];

    result.push_back(dataToMat(
        image_data,
        Size(images.width(), images.height())));
  }

  return result;
}


template<typename Dtype>
vector<vector<typename DetectNetTransformationLayer<Dtype>::BboxLabel> >
DetectNetTransformationLayer<Dtype>::blobToLabels(
    const Blob<Dtype>& labels
) const {
  vector<vector<BboxLabel > > result; result.reserve(labels.num());

  for (size_t iLabel = 0; iLabel != labels.num(); ++iLabel) {
    const Dtype* source = &labels.cpu_data()[
        labels.offset(iLabel, 0, 0, 0)
    ];

    size_t numOfBbox = static_cast<size_t>(source[0]);
    size_t bboxLen = static_cast<size_t>(source[1]);
    CHECK_EQ(bboxLen, sizeof(BboxLabel) / sizeof(Dtype));
    CHECK_LE(numOfBbox, labels.height());
    // exclude header
    source += bboxLen;
    // convert label into typed struct:
    result.push_back(vector<BboxLabel>(
        reinterpret_cast<const BboxLabel*>(source),
        reinterpret_cast<const BboxLabel*>(source + bboxLen * numOfBbox)));
  }

  return result;
}


template<typename Dtype>
struct toDtype : public std::unary_function<float, Dtype> {
  Dtype operator() (const Vec<Dtype, 1>& value) { return value(0); }
};
template<typename Dtype>
void DetectNetTransformationLayer<Dtype>::matToBlob(
    const Mat3v& source,
    Dtype* destination
) const {
  std::vector<Mat1v> channels;
  split(source, channels);

  size_t offset = 0;
  for (size_t iChannel = 0; iChannel != channels.size(); ++iChannel) {
    const Mat1v& channel = channels[iChannel];

    std::transform(
        channel.begin(),
        channel.end(),
        &destination[offset],
        toDtype<Dtype>());

    offset += channel.total();
  }
}


template<typename Dtype>
void DetectNetTransformationLayer<Dtype>::matsToBlob(
    const vector<Mat3v>& _source,
    Blob<Dtype>* _dest
) const {
  for (size_t iImage = 0; iImage != _source.size(); ++iImage) {
    Dtype* destination = &_dest->mutable_cpu_data()[
        _dest->offset(iImage, 0, 0, 0)
    ];
    const Mat3v& source = _source[iImage];
    matToBlob(source, destination);
  }
}


template<typename Dtype>
void DetectNetTransformationLayer<Dtype>::pixelMeanSubtraction(
    Mat3v* source
) const {
  *source -= data_mean_;
}


template<typename Dtype>
void DetectNetTransformationLayer<Dtype>::channelMeanSubtraction(
    Mat3v* source
) const {
  vector<Mat1f> channels;
  split(*source, channels);
  for (size_t iChannel = 0; iChannel != channels.size(); ++iChannel) {
    channels[iChannel] -= mean_values_[iChannel];
  }
  merge(channels, *source);
}


template<typename Dtype>
void DetectNetTransformationLayer<Dtype>::meanSubtract(Mat3v* source) const {
  if (t_param_.has_mean_file()) {
    pixelMeanSubtraction(source);
  } else if (t_param_.mean_value_size() != 0) {
    channelMeanSubtraction(source);
  }
}


template<typename Dtype>
Dtype DetectNetTransformationLayer<Dtype>::randDouble() {
  rng_t* rng =
      static_cast<rng_t*>(rng_->generator());
  uint64_t randval = (*rng)();

  return (Dtype(randval) / Dtype(rng_t::max()));
}


template<typename Dtype>
void DetectNetTransformationLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top
) {
  // verify image parameters:
  const int image_count = bottom[0]->num();
  // verify label parameters:
  const int label_count = bottom[1]->num();
  CHECK_EQ(image_count, label_count);

  const vector<Mat3v> inputImages = blobToMats(*bottom[0]);
  const vector<vector<BboxLabel > > labels = blobToLabels(*bottom[1]);

  vector<Mat3v> outputImages(inputImages.size());
  Blob<Dtype>& outputLabels = *top[1];

  for (size_t iImage = 0; iImage != inputImages.size(); ++iImage) {
    const Mat3v& inputImage = inputImages[iImage];
    const vector<BboxLabel >& inputLabel = labels[iImage];

    Mat3v& outputImage = outputImages[iImage];
    Dtype* outputLabel = &outputLabels.mutable_cpu_data()[
        outputLabels.offset(iImage, 0, 0, 0)
    ];

    transform(inputImage, inputLabel, &outputImage, outputLabel);
  }
  // emplace images in output image blob:
  matsToBlob(outputImages, top[0]);
}

template<typename Dtype>
void DetectNetTransformationLayer<Dtype>::transform(
    const Mat3v& img,
    const vector<BboxLabel >& bboxlist,
    Mat3v* img_aug,
    Dtype* transformed_label
) {
  uint32_t image_x = g_param_.image_size_x();
  uint32_t image_y = g_param_.image_size_y();

  // Perform mean subtraction on un-augmented image:
  Mat3v img_temp = img.clone();  // size determined by scale
  // incoming float images have values from 0..255, but OpenCV expects these
  //  values to be from 0..1:
  img_temp /= Dtype(UINT8_MAX);
  *img_aug = Mat::zeros(image_y, image_x, img.type());
  vector<BboxLabel > bboxlist_aug;
  AugmentSelection as;
  // We only do random transform as augmentation when training.
  if (this->phase_ == TRAIN) {
    // TODO: combine hueRotation and desaturation if performance is a concern,
    //  as we must redundantly convert to and from HSV colorspace for each
    //  operation
    as.hue_rotation = augmentation_hueRotation(img_temp, &img_temp);
    as.saturation = augmentation_desaturation(img_temp, &img_temp);
    // mean subtraction must occur after color augmentations as colorshift
    //  outside of 0..1 invalidates scale
    meanSubtract(&img_temp);
    // images now bounded from -0.5...0.5
    as.scale = augmentation_scale(img_temp, img_aug, bboxlist, &bboxlist_aug);
    as.degree =
        augmentation_rotate(*img_aug, &img_temp, bboxlist_aug, &bboxlist_aug);
    as.flip = augmentation_flip(img_temp, img_aug, bboxlist_aug, &bboxlist_aug);
    as.crop = augmentation_crop(
        *img_aug,
        &img_temp,
        bboxlist_aug,
        &bboxlist_aug);
    *img_aug = img_temp;
  } else {
    // perform mean subtraction:
    meanSubtract(&img_temp);

    // deterministically scale the image and ground-truth, if requested:
    transform_scale(
        img_temp,
        img_aug,
        bboxlist,
        &bboxlist_aug,
        Size(image_x, image_y));
  }

  CHECK_EQ(img_aug->cols, image_x);
  CHECK_EQ(img_aug->rows, image_y);

  // generate transformed_label based on bboxlist_aug
  coverage_->generate(transformed_label, bboxlist_aug);

  // networks expect floats bounded from -255..255, so rescale to this range:
  *img_aug *= Dtype(UINT8_MAX);
}


template<typename Dtype>
float DetectNetTransformationLayer<Dtype>::augmentation_scale(
    const Mat3v& img_src,
    Mat3v* img_dst,
    const vector<BboxLabel >& bboxList,
    vector<BboxLabel >* bboxList_aug
) {
  bool doScale = randDouble() <= a_param_.scale_prob();
  // linear shear into [scale_min, scale_max]
  float scale =
      a_param_.scale_min() +
      (a_param_.scale_max() - a_param_.scale_min()) *
      randDouble();

  if (doScale) {
    // scale uniformly across both axes by some random value:
    Size scaleSize(round(img_src.cols * scale), round(img_src.rows * scale));
    transform_scale(
        img_src,
        img_dst,
        bboxList,
        bboxList_aug,
        scaleSize);
  } else {
    *img_dst = img_src.clone();
    *bboxList_aug = bboxList;
    scale = 1;
  }

  return scale;
}


template<typename Dtype>
void DetectNetTransformationLayer<Dtype>::transform_scale(
    const Mat3v& img,
    Mat3v* img_temp,
    const vector<BboxLabel >& bboxList,
    vector<BboxLabel >* _bboxList_aug,
    const Size& size
) {
  // perform scaling if desired size and image size are non-equal:
  if (size.height != img.rows || size.width != img.cols) {
    Dtype scale_x = (Dtype)size.width / img.cols;
    Dtype scale_y = (Dtype)size.height / img.rows;

    resize(img, *img_temp, size, cv::INTER_CUBIC);

    vector<BboxLabel > bboxList_aug;
    foreach_(BboxLabel label, bboxList) {  // for every bbox:
      // resize by scale
      label.bbox = Rectv(
          label.bbox.x * scale_x,
          label.bbox.y * scale_y,
          label.bbox.width * scale_x,
          label.bbox.height * scale_y);
      bboxList_aug.push_back(label);
    }

    _bboxList_aug->swap(bboxList_aug);

  } else {
    *img_temp = img.clone();
    *_bboxList_aug = bboxList;
  }
}


template<typename Dtype>
Point DetectNetTransformationLayer<Dtype>::augmentation_crop(
    const Mat3v& img_src,
    Mat3v* img_dst,
    const vector<BboxLabel >& bboxList,
    vector<BboxLabel >* bboxList_aug
) {
  bool doCrop = randDouble() <= a_param_.crop_prob();
  Size2i crop(g_param_.image_size_x(), g_param_.image_size_y());
  Size2i shift(a_param_.shift_x(), a_param_.shift_y());
  Size2i imgSize(img_src.cols, img_src.rows);
  Point2i offset, inner, outer;
  if (doCrop) {
    // perform a random crop, bounded by the difference between the network's
    //  input and the size of the incoming image. Add a user-defined shift to
    //  the max range of the crop offset.
    offset = Point2i(
        round(randDouble() * (imgSize.width  - crop.width  + shift.width)),
        round(randDouble() * (imgSize.height - crop.height + shift.height)));
    offset -= Point2i(shift.width / 2, shift.height / 2);
  } else {
    // perform a deterministic crop, placing the image in the middle of the
    //  network's input region:
    offset = Point2i(
        (imgSize.width  - crop.width) / 2,
        (imgSize.height - crop.height) / 2);
  }
  inner = Point2i(std::max(0,  1 * offset.x), std::max(0,  1 * offset.y));
  outer = Point2i(std::max(0, -1 * offset.x), std::max(0, -1 * offset.y));

  // crop / grow to size:
  transform_crop(
      img_src,
      img_dst,
      bboxList,
      bboxList_aug,
      Rect(inner, crop),
      crop,
      outer);
  return offset;
}


template<typename Dtype>
void DetectNetTransformationLayer<Dtype>::transform_crop(
    const Mat3v& img_src,
    Mat3v* img_dst,
    const vector<BboxLabel>& bboxlist,
    vector<BboxLabel>* _bboxlist_aug,
    Rect inner,
    Size2i dst_size,
    Point2i outer
) const {
  // ensure src_rect fits within img_src:
  Rect src_rect = inner & Rect(Point(0, 0), Size2i(img_src.cols, img_src.rows));
  // dst_rect has a size the same as src_rect:
  Rect dst_rect(outer, src_rect.size());
  // ensure dst_rect fits within img_dst:
  dst_rect &= Rect(Point(0, 0), dst_size);
  // assert src_rect and dst_rect have the same size:
  src_rect = Rect(src_rect.tl(), dst_rect.size());
  // Fail with an Opencv exception if any of these are negative

  // no operation is needed in the case of zero transformations:
  if (src_rect == dst_rect
    && src_rect.tl() == Point2i(0, 0)
    && src_rect.size() == dst_size
    && dst_size == Size2i(img_src.cols, img_src.rows)) {
    // no crop is needed:
    *img_dst = img_src.clone();
    *_bboxlist_aug = bboxlist;
  } else {
    // construct a destination matrix:
    *img_dst = Mat3v(dst_size);
    // and fill with black:
    img_dst->setTo(Scalar(-0.5, -0.5, -0.5));

    // define destinationROI inside of destination mat:
    Mat3v destinationROI = (*img_dst)(dst_rect);
    // sourceROI inside of source mat:
    Mat3v sourceROI = img_src(src_rect);
    // copy sourceROI into destinationROI:
    sourceROI.copyTo(destinationROI);

    // translate all bounding boxes by the new offset, -inner + outer:
    Point2v bboxOffset = dst_rect.tl() - src_rect.tl();
    vector<BboxLabel> bboxlist_aug;
    foreach_(const BboxLabel& label, bboxlist) {  // for every bbox
      BboxLabel cropped = label;
      // shift topLeft by offset
      cropped.bbox += bboxOffset;

      // Add to list of bboxes.
      bboxlist_aug.push_back(cropped);
    }
    _bboxlist_aug->swap(bboxlist_aug);
  }
}


template<typename Dtype>
bool DetectNetTransformationLayer<Dtype>::augmentation_flip(
    const Mat3v& img_src,
    Mat3v* img_aug,
    const vector<BboxLabel >& bboxlist,
    vector<BboxLabel >* _bboxlist_aug) {
  bool doflip = randDouble() <= a_param_.flip_prob();
  if (doflip) {
    flip(img_src, *img_aug, 1);
    float w = img_src.cols;

    vector <BboxLabel > bboxlist_aug;
    foreach_(BboxLabel label, bboxlist) {  // for every bbox
      // flip X across the width of the image:
      label.bbox.x = w - label.bbox.x - 1.0f - label.bbox.width;
      // Y, width, height stay the same

      bboxlist_aug.push_back(label);
    }
    _bboxlist_aug->swap(bboxlist_aug);
  } else {
    *img_aug = img_src.clone();
    *_bboxlist_aug = bboxlist;
  }
  return doflip;
}


template<typename Dtype>
typename DetectNetTransformationLayer<Dtype>::Mat1v
DetectNetTransformationLayer<Dtype>::getTransformationMatrix(
    Rect region,
    Dtype rotation
) const {
  Size2v size(region.width, region.height);
  Point2v center = size * (Dtype)0.5;
  array<Point2f, 4> srcTri, dstTri;

  // Define a rotated rectangle for our initial position, retrieving points
  //  for each of our 4 corners:
  RotatedRect initialRect(center, size, 0.0);
  initialRect.points(srcTri.c_array());
  // Another rotated rectangle for our eventual position.
  RotatedRect rotatedRect(center, size, rotation);
  // retrieve boundingRect, whose top-left will be below zero:
  Rectv boundingRect = rotatedRect.boundingRect();
  // push all points up by the the topleft boundingRect's delta from
  //  the origin:
  rotatedRect = RotatedRect(center - boundingRect.tl(), size, rotation);
  // retrieve points for each of the rotated rectangle's 4 corners:
  rotatedRect.points(dstTri.c_array());

  // compute the affine transformation of this operation:
  Mat1v result(2, 3);
  result = getAffineTransform(srcTri.c_array(), dstTri.c_array());
  // return the transformation matrix
  return result;
}


template<typename Dtype>
Rect DetectNetTransformationLayer<Dtype>::getBoundingRect(
    Rect region,
    Dtype rotation
) const {
  Size2v size(region.width, region.height);
  Point2v center = size * (Dtype)0.5;

  return RotatedRect(center, size, rotation).boundingRect();
}


template<typename Dtype>
float DetectNetTransformationLayer<Dtype>::augmentation_rotate(
    const Mat3v& img_src,
    Mat3v* img_aug,
    const vector<BboxLabel >& bboxList,
    vector<BboxLabel >* _bboxlist_aug) {
  bool doRotate = randDouble() <= a_param_.rotation_prob();
  float degree = (randDouble() - 0.5) * 2 * a_param_.max_rotate_degree();

  if (doRotate && std::abs(degree) > FLT_EPSILON) {
    Rect roi(0, 0, img_src.cols, img_src.rows);
    // determine new bounding rect:
    Size2i boundingSize = getBoundingRect(roi, degree).size();
    // determine rotation matrix:
    Mat1v transformationMatrix = getTransformationMatrix(roi, degree);

    // construct a destination matrix large enough to contain the rotated image:
    *img_aug = Mat3v(boundingSize);
    // and fill with black:
    img_aug->setTo(Scalar(-0.5, -0.5, -0.5));
    // warp old image into new buffer, maintaining the background:
    warpAffine(
        img_src,
        *img_aug,
        transformationMatrix,
        boundingSize,
        INTER_LINEAR,
        BORDER_TRANSPARENT);

    vector<BboxLabel > bboxlist_aug;
    foreach_(BboxLabel label, bboxList) {  // for each bbox:
      Mat1v center(3, 1);
      // center of bbox as midpoint between topLeft and bottomRight:
      Point2v center_point = (label.bbox.tl() + label.bbox.br()) * 0.5;
      center.template at<Dtype>(0, 0) = center_point.x;
      center.template at<Dtype>(1, 0) = center_point.y;
      center.template at<Dtype>(2, 0) = 1.0;

      // rotate around the origin:
      Mat1v new_center(3, 1);
      new_center = transformationMatrix * center;

      // rotated bbox with topleft as midpoint - size/2 and prior size
      Point2v new_center_point(
          new_center.template at<Dtype>(0, 0),
          new_center.template at<Dtype>(1, 0));
      Rectv rotated(
          new_center_point - (label.bbox.br() - center_point),
          label.bbox.size());
      label.bbox = rotated;

      // Add to list of bboxes. Note that it's possible for the bounding box
      // to be out of the display area at this point
      bboxlist_aug.push_back(label);
    }
    _bboxlist_aug->swap(bboxlist_aug);
  } else {
    *img_aug = img_src.clone();
    *_bboxlist_aug = bboxList;
    degree = 0.0f;
  }

  return degree;
}


template<typename Dtype>
float DetectNetTransformationLayer<Dtype>::augmentation_hueRotation(
    const Mat3v& img,
    Mat3v* result) {
  bool doHueRotate = randDouble() <= a_param_.hue_rotation_prob();
  // rotate hue by this amount in degrees
  float rotation =
      (randDouble()                       // range: 0..1
      * (2.0 * a_param_.hue_rotation()))  // range: 0..2*rot
      - a_param_.hue_rotation();          // range: -rot..rot
  // clamp to -180d..180d
  rotation = std::max(std::min(rotation, 180.0f), -180.0f);

  // if we're actually rotating:
  if (doHueRotate && std::abs(rotation) > FLT_EPSILON) {
    // convert to HSV colorspace
    cvtColor(img, *result, COLOR_BGR2HSV);

    // retrieve the hue channel:
    static const array<int, 2> from_mix = {{0, 0}};
    Mat1v hueChannel(result->rows, result->cols);
    mixChannels(
        result, 1,
        &hueChannel, 1,
        from_mix.data(), from_mix.size()/2);

    // shift the hue's value by some amount:
    hueChannel += rotation;

    // place hue-rotated channel back in result matrix:
    // NOLINT_NEXT_LINE(whitespace/comma)
    static const array<int, 6> to_mix = {{3,0, 1,1, 2,2}};
    const array<Mat, 2> to_channels = {{*result, hueChannel}};
    mixChannels(
        to_channels.data(), 2,
        result, 1,
        to_mix.data(), to_mix.size()/2);

    // back to BGR colorspace
    cvtColor(*result, *result, COLOR_HSV2BGR);
  } else {
    *result = img;
    rotation = 0.0f;
  }

  return rotation;
}


template<typename Dtype>
float DetectNetTransformationLayer<Dtype>::augmentation_desaturation(
    const Mat3v& img,
    Mat3v* result) {
  bool doDesaturate = randDouble() <= a_param_.desaturation_prob();
  // scale saturation by this amount:
  float saturation =
      1.0                           // inverse
    - randDouble()                  // range: 0..1
    * a_param_.desaturation_max();  // range: 0..max

  // if our random value is large enough to produce noticeable desaturation:
  if (doDesaturate && (saturation < 1.0 - 1.0/UINT8_MAX)) {
    // convert to HSV colorspace
    cvtColor(img, *result, COLOR_BGR2HSV);

    // retrieve the saturation channel:
    static const array<int, 2> from_mix = {{1, 0}};
    Mat1v saturationChannel(result->rows, result->cols);
    mixChannels(
        result, 1,
        &saturationChannel, 1,
        from_mix.data(), from_mix.size()/2);
    // de-saturate the channel by an amount:
    saturationChannel *= saturation;

    // place de-saturated channel back in result matrix:
    // NOLINT_NEXT_LINE(whitespace/comma)
    static const array<int, 6> to_mix = {{0,0, 3,1, 2,2}};
    const array<Mat, 2> to_channels = {{*result, saturationChannel}};
    mixChannels(
        to_channels.data(), 2,
        result, 1,
        to_mix.data(), to_mix.size()/2);

    // convert back to BGR colorspace:
    cvtColor(*result, *result, COLOR_HSV2BGR);
  } else {
    *result = img;
    saturation = 1.0;
  }

  return saturation;
}


INSTANTIATE_CLASS(DetectNetTransformationLayer);
REGISTER_LAYER_CLASS(DetectNetTransformation);

}  // namespace caffe

#endif  // USE_OPENCV
