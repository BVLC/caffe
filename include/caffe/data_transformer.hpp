#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#include <vector>
#include <random>

#include "google/protobuf/repeated_field.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

using google::protobuf::RepeatedPtrField;

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class DataTransformer {
 public:
  explicit DataTransformer(const TransformationParameter& param, Phase phase);
  virtual ~DataTransformer() {}

  /**
   * @brief Initialize the Random number generations if needed by the
   *    transformation.
   */
  void InitRand();

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See data_layer.cpp for an example.
   */
  void Transform(const Datum& datum, Blob<Dtype>* transformed_blob,
                 const bool preserve_pixel_vals = false,
                 const bool preserve_annotations = false);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Datum.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  void Transform(const vector<Datum> & datum_vector,
                Blob<Dtype>* transformed_blob);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the annotated data.
   *
   * @param anno_datum
   *    AnnotatedDatum containing the data and annotation to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See annotated_data_layer.cpp for an example.
   * @param transformed_anno_vec
   *    This is destination annotation.
   */
  void Transform(const AnnotatedDatum& anno_datum,
                 Blob<Dtype>* transformed_blob,
                 RepeatedPtrField<AnnotationGroup>* transformed_anno_vec);
  void Transform(const AnnotatedDatum& anno_datum,
                 Blob<Dtype>* transformed_blob,
                 RepeatedPtrField<AnnotationGroup>* transformed_anno_vec,
                 bool* do_mirror);
  void Transform(const AnnotatedDatum& anno_datum,
                 Blob<Dtype>* transformed_blob,
                 vector<AnnotationGroup>* transformed_anno_vec,
                 bool* do_mirror);
  void Transform(const AnnotatedDatum& anno_datum,
                 Blob<Dtype>* transformed_blob,
                 vector<AnnotationGroup>* transformed_anno_vec);

  /**
   * @brief Transform the annotation according to the transformation applied
   * to the datum.
   *
   * @param anno_datum
   *    AnnotatedDatum containing the data and annotation to be transformed.
   * @param do_resize
   *    If true, resize the annotation accordingly before crop.
   * @param crop_bbox
   *    The cropped region applied to anno_datum.datum()
   * @param do_mirror
   *    If true, meaning the datum has mirrored.
   * @param transformed_anno_group_all
   *    Stores all transformed AnnotationGroup.
   * @param rangle
   *    The rotation angle, if any
   */
  void TransformAnnotation(
      const AnnotatedDatum& anno_datum, const bool do_resize,
      const NormalizedBBox& crop_bbox, const bool do_mirror,
      const bool do_crop,
      RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all,
      const int rangle=0);

  /**
   * @brief Crops the datum according to bbox.
   */
  void CropImage(const Datum& datum, const NormalizedBBox& bbox,
                 Datum* crop_datum);

  /**
   * @brief Crops the datum and AnnotationGroup according to bbox.
   */
  void CropImage(const AnnotatedDatum& anno_datum, const NormalizedBBox& bbox,
                 AnnotatedDatum* cropped_anno_datum);

  /**
   * @brief Expand the datum.
   */
  void ExpandImage(const Datum& datum, const float expand_ratio,
                   NormalizedBBox* expand_bbox, Datum* expanded_datum);

  /**
   * @brief Expand the datum and adjust AnnotationGroup.
   */
  void ExpandImage(const AnnotatedDatum& anno_datum,
                   AnnotatedDatum* expanded_anno_datum);

  /**
   * @brief Apply distortion to the datum.
   */
  void DistortImage(const Datum& datum, Datum* distort_datum);

#ifdef USE_OPENCV
  /**
   * @brief Rotate image.
   */
  void RotateImage(cv::Mat &cv_img,
		   int &r);
  
  /**
   * @brief Rotate datum
   */
  void RotateImage(const Datum& datum,
		   Datum* rotate_datum,
		   int& r);

  /**
   * @brief Rotate annotated datum and annotations (bbox)
   */
  void RotateImage(const AnnotatedDatum& anno_datum,
		   AnnotatedDatum* rotated_anno_datum);
  
  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Mat.
   *
   * @param mat_vector
   *    A vector of Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  void Transform(const vector<cv::Mat> & mat_vector,
                Blob<Dtype>* transformed_blob);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a cv::Mat
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See image_data_layer.cpp for an example.
   * @param preserve_pixel_vals
   *    Use with dense label images to preserve the input pixel values
   *    which would be labels (and thus cannot be interpolated or scaled).
   * @param preserve_annotations
   *    Use with bounding boxes annotations to avoid rotations for instance
   *    (that are computed from within the annotated data layer instead)
   */
  void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob,
                 NormalizedBBox* crop_bbox, bool* do_mirror,
                 bool preserve_pixel_vals = false, bool preserve_annotations = false,
                 bool use_previous_mirror_value = false);
  void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob,
		 bool preserve_pixel_vals = false, bool preserve_annotations = false);

  /**
   * @brief Crops img according to bbox.
   */
  void CropImage(const cv::Mat& img, const NormalizedBBox& bbox,
                 cv::Mat* crop_img);

  /**
   * @brief Expand img to include mean value as background.
   */
  void ExpandImage(const cv::Mat& img, const float expand_ratio,
                   NormalizedBBox* expand_bbox, cv::Mat* expand_img);

  void TransformInv(const Blob<Dtype>* blob, vector<cv::Mat>* cv_imgs);
  void TransformInv(const Dtype* data, cv::Mat* cv_img, const int height,
                    const int width, const int channels);
#endif  // USE_OPENCV

  /**
   * @brief Applies the same transformation defined in the data layer's
   * transform_param block to all the num images in a input_blob.
   *
   * @param input_blob
   *    A Blob containing the data to be transformed. It applies the same
   *    transformation to all the num images in the blob.
   * @param transformed_blob
   *    This is destination blob, it will contain as many images as the
   *    input blob. It can be part of top blob's data.
   */
  void Transform(Blob<Dtype>* input_blob, Blob<Dtype>* transformed_blob);

  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   */
  vector<int> InferBlobShape(const Datum& datum);
  vector<int> InferBlobShape(const SparseDatum &datum);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *    It uses the first element to infer the shape of the blob.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   */
  vector<int> InferBlobShape(const vector<Datum> & datum_vector);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *    It uses the first element to infer the shape of the blob.
   *
   * @param mat_vector
   *    A vector of Mat containing the data to be transformed.
   */
  vector<int> InferBlobShape(const vector<cv::Mat> & mat_vector);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   */
  vector<int> InferBlobShape(const cv::Mat& cv_img);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data and return transform information.
   */
  void Transform(const Datum& datum, Blob<Dtype>* transformed_blob,
                 NormalizedBBox* crop_bbox, bool* do_mirror,
                 const bool preserve_pixel_vals = false,
                 const bool preserve_annotations = false,
                 const bool use_previous_mirror_value = false);


 protected:
   /**
   * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
   *
   * @param n
   *    The upperbound (exclusive) value of the random number.
   * @return
   *    A uniformly random integer value from ({0, 1, ..., n-1}).
   */
  virtual int Rand(int n);

  // Transform and return the transformation information.
  void Transform(const Datum& datum, Dtype* transformed_data,
                 NormalizedBBox* crop_bbox, bool* do_mirror);
  void Transform(const Datum& datum, Dtype* transformed_data);


  // Tranformation parameters
  TransformationParameter param_;


  shared_ptr<Caffe::RNG> rng_;
  Phase phase_;
  Blob<Dtype> data_mean_;
  vector<Dtype> mean_values_;
  std::mt19937 rg_;
  std::uniform_int_distribution<int> rd_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_

