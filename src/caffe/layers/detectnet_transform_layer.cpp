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
    cv::resize(data_mean_, data_mean_, dimensions, 0, 0, cv::INTER_CUBIC);
  }

#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    mean_blob_.Reshape(1, data_mean_.channels(),
        data_mean_.size().height, data_mean_.size().width);
    matToBlob(data_mean_, mean_blob_.mutable_cpu_data());
  }
#endif

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

#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    const size_t aug_data_sz = sizeof(AugmentSelection) * bottom[0]->num();
    gpu_workspace_augmentations_.reserve(aug_data_sz);
    const size_t tmp_data_sz = sizeof(Dtype) * bottom[0]->count();
    gpu_workspace_tmpdata_.reserve(tmp_data_sz);
  }
#endif
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

    AugmentSelection as = get_augmentations(inputImage.size());

    outputImage = transform_image_cpu(inputImage, as);
    transform_label_cpu(inputLabel, outputLabel, as, inputImage.size());
  }
  // emplace images in output image blob:
  matsToBlob(outputImages, top[0]);
}

template<typename Dtype>
AugmentSelection DetectNetTransformationLayer<Dtype>::get_augmentations(
    cv::Size original_size) {
  const int image_size_x = g_param_.image_size_x();
  const int image_size_y = g_param_.image_size_y();
  AugmentSelection as;
  // Turn off all augmentations
  as.hue_rotation = 0;
  as.saturation = 1;
  as.flip = false;
  as.scale = original_size;
  as.rotation = 0;
  as.crop_offset = cv::Point(0, 0);

  if (this->phase_ == TRAIN) {  // Randomly enable some augmentations
    if (randDouble() <= a_param_.hue_rotation_prob()) {
      // [-hue_rotation, hue_rotation]
      as.hue_rotation = (randDouble() - 0.5) * 2 * a_param_.hue_rotation();
      // clamp to -180d..180d
      as.hue_rotation = std::max(std::min(as.hue_rotation, 180.0f), -180.0f);
    }
    if (randDouble() <= a_param_.desaturation_prob()) {
      // [1-desaturation_max, 1]
      as.saturation = 1.0 - randDouble() * a_param_.desaturation_max();
    }
    as.flip = randDouble() <= a_param_.flip_prob();
    if (randDouble() <= a_param_.scale_prob()) {
      // [scale_min, scale_max]
      float scale = a_param_.scale_min() +
        (a_param_.scale_max() - a_param_.scale_min()) * randDouble();
      as.scale = cv::Size(round(original_size.width * scale),
          round(original_size.height * scale));
    }
    if (randDouble() <= a_param_.rotation_prob()) {
      // [-max_rotate_degree, max_rotate_degree]
      as.rotation = (randDouble() - 0.5) * 2 * a_param_.max_rotate_degree();
    }

    // crop calculations
    cv::Size before_crop = as.scale;
    if (as.doRotation()) {
      before_crop = getRotatedSize(as.scale, as.rotation);
    }
    int shift_x, shift_y;
    double pct_x, pct_y;
    if (randDouble() <= a_param_.crop_prob()) {
      // Random crop
      pct_x = randDouble();
      pct_y = randDouble();
      shift_x = a_param_.shift_x();
      shift_y = a_param_.shift_y();
    } else {
      // Deterministic crop
      pct_x = 0.5;
      pct_y = 0.5;
      shift_x = 0;
      shift_y = 0;
    }
    as.crop_offset = cv::Point(
        round(pct_x * (before_crop.width - image_size_x + shift_x) -
          shift_x/2),
        round(pct_y * (before_crop.height - image_size_y + shift_y) -
          shift_y/2));
  } else {  // phase == TEST
    as.scale = cv::Size(image_size_x, image_size_y);
  }
  return as;
}


template<typename Dtype>
typename DetectNetTransformationLayer<Dtype>::Mat3v
DetectNetTransformationLayer<Dtype>::transform_image_cpu(
    const Mat3v& src_img, const AugmentSelection& as
) {
  // Scale from [0,255] to [0,1] for HSV augmentations
  Mat3v img = src_img.clone() / UINT8_MAX;

  // Do HSV transformations before mean subtraction while image still in [0,1]
  if (as.doHueRotation() || as.doDesaturation()) {
    img = transform_hsv_cpu(img, as);
  }

  meanSubtract(&img);
  if (as.flip) {
    cv::flip(img, img, 1);
  }
  if (as.doScale(img.size())) {
    cv::resize(img, img, as.scale, 0, 0, cv::INTER_CUBIC);
  }
  if (as.doRotation()) {
    img = rotate_image_cpu(img, as.rotation);
  }
  img = crop_image_cpu(img,
      cv::Size(
        g_param_.image_size_x(),
        g_param_.image_size_y()),
      as.crop_offset);

  // Scale from [-1,1] into [-255,255]
  img *= Dtype(UINT8_MAX);
  return img;
}


template<typename Dtype>
void DetectNetTransformationLayer<Dtype>::transform_label_cpu(
    vector<BboxLabel> bboxes, Dtype* transformed_label,
    const AugmentSelection& as, const cv::Size& orig_size
) {
  if (as.flip) {
    bboxes = flip_label_cpu(bboxes, orig_size);
  }
  if (as.doScale(orig_size)) {
    bboxes = scale_label_cpu(bboxes, orig_size, as.scale);
  }
  if (as.doRotation()) {
    bboxes = rotate_label_cpu(bboxes, as.scale, as.rotation);
  }
  bboxes = crop_label_cpu(bboxes, as.crop_offset);

  coverage_->generate(transformed_label, bboxes);
}


template<typename Dtype>
typename DetectNetTransformationLayer<Dtype>::Mat3v
DetectNetTransformationLayer<Dtype>::transform_hsv_cpu(
    const Mat3v& orig, const AugmentSelection& as) {
  // Use CV_32F since cvtColor doesn't support CV_64F
  cv::Mat_<cv::Vec<float, 3> > result = orig;
  cvtColor(result, result, COLOR_BGR2HSV);
  vector<Mat1v> channels(3);
  cv::split(result, channels);
  // Perform transformations
  if (as.doHueRotation()) {
    channels[0] += as.hue_rotation;
  }
  if (as.doDesaturation()) {
    channels[1] *= as.saturation;
  }
  cv::merge(channels, result);
  // Convert back to BGR
  cvtColor(result, result, COLOR_HSV2BGR);
  return result;
}


template<typename Dtype>
vector<typename DetectNetTransformationLayer<Dtype>::BboxLabel>
DetectNetTransformationLayer<Dtype>::flip_label_cpu(
    const vector<BboxLabel>& orig, const cv::Size& size) {
  vector<BboxLabel> result;
  foreach_(BboxLabel label, orig) {
    // flip X across the width of the image:
    label.bbox.x = size.width - label.bbox.x - label.bbox.width - 1;
    result.push_back(label);
  }
  return result;
}


template<typename Dtype>
vector<typename DetectNetTransformationLayer<Dtype>::BboxLabel>
DetectNetTransformationLayer<Dtype>::scale_label_cpu(
    const vector<BboxLabel>& orig,
    const cv::Size& old_size, const cv::Size& new_size) {
  vector<BboxLabel> result;
  Dtype scale_x = (Dtype)new_size.width / old_size.width;
  Dtype scale_y = (Dtype)new_size.height / old_size.height;
  foreach_(BboxLabel label, orig) {
    // resize by scale
    label.bbox = Rectv(
        label.bbox.x * scale_x,
        label.bbox.y * scale_y,
        label.bbox.width * scale_x,
        label.bbox.height * scale_y);
    result.push_back(label);
  }
  return result;
}


template<typename Dtype>
typename DetectNetTransformationLayer<Dtype>::Mat3v
DetectNetTransformationLayer<Dtype>::rotate_image_cpu(
    const Mat3v& orig, const float rotation) {
  // Calculate new size
  Rect roi(0, 0, orig.cols, orig.rows);
  Size2i boundingSize = getRotatedSize(roi.size(), rotation);
  Mat3v result(boundingSize, Dtype(0));
  // Rotate image
  Mat1v transformationMatrix = getTransformationMatrix(roi, rotation);
  cv::warpAffine(orig, result, transformationMatrix, boundingSize,
      INTER_LINEAR, BORDER_TRANSPARENT);
  return result;
}


template<typename Dtype>
vector<typename DetectNetTransformationLayer<Dtype>::BboxLabel>
DetectNetTransformationLayer<Dtype>::rotate_label_cpu(
    const vector<BboxLabel>& orig, const cv::Size& orig_size,
    const float rotation) {
  // Get transformation matrix
  Rect roi(0, 0, orig_size.width, orig_size.height);
  Mat1v transformationMatrix = getTransformationMatrix(roi, rotation);
  // Rotate labels
  vector<BboxLabel> result;
  foreach_(BboxLabel label, orig) {
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
    result.push_back(label);
  }
  return result;
}

template<typename Dtype>
typename DetectNetTransformationLayer<Dtype>::Mat3v
DetectNetTransformationLayer<Dtype>::crop_image_cpu(
    const Mat3v& orig, const cv::Size& new_size, const cv::Point& crop_offset
) {
  Mat3v result(new_size, Dtype(0));

  // Create rects for ROIs
  cv::Rect src_rect = cv::Rect(crop_offset, new_size) &
    cv::Rect(cv::Point(0, 0), orig.size());
  cv::Rect dst_rect = cv::Rect(-crop_offset, orig.size()) &
    cv::Rect(cv::Point(0, 0), new_size);
  CHECK_EQ(src_rect.size(), dst_rect.size());

  // Copy
  Mat3v sourceROI = orig(src_rect);
  Mat3v destinationROI = result(dst_rect);
  sourceROI.copyTo(destinationROI);

  return result;
}

template<typename Dtype>
vector<typename DetectNetTransformationLayer<Dtype>::BboxLabel>
DetectNetTransformationLayer<Dtype>::crop_label_cpu(
    const vector<BboxLabel>& orig, const cv::Point& crop_offset
) {
  vector<BboxLabel> result;
  Point2v offset = crop_offset;  // convert to Dtype
  foreach_(BboxLabel label, orig) {
    label.bbox -= offset;
    result.push_back(label);
  }
  return result;
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
cv::Size DetectNetTransformationLayer<Dtype>::getRotatedSize(
    cv::Size size, float rotation) const {
  cv::Point center(0.5 * size.width, 0.5 * size.height);
  return cv::RotatedRect(center, size, rotation).boundingRect().size();
}


#ifdef CPU_ONLY
STUB_GPU_FORWARD(DetectNetTransformationLayer, Forward);
#endif

INSTANTIATE_CLASS(DetectNetTransformationLayer);
REGISTER_LAYER_CLASS(DetectNetTransformation);

}  // namespace caffe

#endif  // USE_OPENCV
