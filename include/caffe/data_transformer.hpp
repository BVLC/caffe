/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#include <queue>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

using google::protobuf::RepeatedPtrField;

namespace caffe {

class DataReader;

class BoxLabel {
 public:
  float class_label_;
  float difficult_;
  float box_[4];
};

class RandNumbers {
 public:
   /**
   * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
   *
   * @param n
   *    The upperbound (exclusive) value of the random number.
   * @return
   *    A uniformly random integer value from ({0, 1, ..., n-1}).
   */
  int operator()(int n) {
    CHECK_GT(n, 0);
    return GetNextNumber() % n;
  }

  virtual uint32_t GetNextNumber() = 0;
};

class GenRandNumbers: public RandNumbers {
 public:
  void Init() {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  }

  void Reset() { rng_.reset(); }
  bool IsEmpty() { return (rng_.get() == nullptr); }

  virtual uint32_t GetNextNumber() {
    CHECK(rng_);
    caffe::rng_t* rng = static_cast<caffe::rng_t*>(rng_->generator());
    return (*rng)();
  }
 private:
  shared_ptr<Caffe::RNG> rng_;
};


class PreclcRandomNumbers: public RandNumbers {
 public:
  void FillRandomNumbers(int num_count, RandNumbers& rand_gen) {
    for (int i = 0; i < num_count; i++)
      random_numbers.push(rand_gen.GetNextNumber());
  }

  virtual uint32_t GetNextNumber() {
    CHECK(!random_numbers.empty());
    uint32_t num = random_numbers.front();
    random_numbers.pop();
    return num;
  }
 private:
  std::queue<uint32_t> random_numbers;
};

namespace {
  // We use these type and value to distinguish when
  // annotation handler should be used.
  struct EmptyType { };
  EmptyType empty_value;
}

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
  void ReinitRand();

  void GenerateRandNumbers(PreclcRandomNumbers& rn, bool sample_bboxes = false);
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

  void Transform(const Datum& datum, Blob<Dtype>* transformed_blob)
                               {Transform(datum, transformed_blob, rand_num_);}
  void Transform(const Datum& datum, Blob<Dtype>* transformed_blob,
                                                       RandNumbers& rand_num);

  void Transform(Blob<Dtype>* input_blob,
                 Blob<Dtype>* transformed_blob,
                 RandNumbers& rand_num);

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
		 RandNumbers& rand_num);
  void Transform(const AnnotatedDatum& anno_datum,
                 Blob<Dtype>* transformed_blob,
                 vector<AnnotationGroup>* transformed_anno_vec);
  void Transform(const AnnotatedDatum& anno_datum,
		 Blob<Dtype>* transformed_blob,
		 vector<AnnotationGroup>* transformed_anno_vec,
		 RandNumbers& rand_num);
  void Transform(const Datum& datum, Blob<Dtype>* transformed_blob, vector<BoxLabel>* box_labels); 
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
   */
  void TransformAnnotation(
      const AnnotatedDatum& anno_datum, const bool do_resize,
      const NormalizedBBox& crop_bbox, const bool do_mirror,
      RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all);

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
   */
  template<typename AnnotationHandler = EmptyType>
  void Transform(const cv::Mat& cv_img,
                 Blob<Dtype>* transformed_blob,
                 NormalizedBBox* crop_bbox,
                 RandNumbers& rand_num,
                 AnnotationHandler anno_handler = empty_value);
  void Transform(const cv::Mat& cv_img,
                 Blob<Dtype>* transformed_blob,
                 RandNumbers& rand_num);
  void Transform(const cv::Mat& cv_img,
                 Blob<Dtype>* transformed_blob) {
    Transform(cv_img, transformed_blob, rand_num_);
  }


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
#ifdef USE_OPENCV
  vector<int> InferBlobShape(const vector<cv::Mat> & mat_vector);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   */
  vector<int> InferBlobShape(const cv::Mat& cv_img);
#endif  // USE_OPENCV

 protected:
  GenRandNumbers rand_num_;

  void Transform(const Datum& datum, Dtype* transformed_data);

  // Transform and return the transformation information.
  template<typename AnnotationHandler = EmptyType>
  void Transform(const Datum& datum, Dtype* transformed_data,
                 NormalizedBBox* crop_bbox, RandNumbers& rand_num,
                 AnnotationHandler annotation_handler = empty_value);

  void Transform(const Datum& datum, Dtype* transformed_data,
                RandNumbers& rand_num);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data and return transform information.
   */
  template<typename AnnotationHandler = EmptyType>
  void Transform(const Datum& datum, Blob<Dtype>* transformed_blob,
                 NormalizedBBox* crop_bbox, RandNumbers& rand_num,
                 AnnotationHandler annotation_handler = empty_value);

  // Tranformation parameters
  TransformationParameter param_;

  Phase phase_;
  Blob<Dtype> data_mean_;
  vector<Dtype> mean_values_;

  // Data reader used if any to get data
  DataReader* data_reader_used;


 private:
  void Transform(const Datum& datum, Dtype* transformed_data,
                 NormalizedBBox* crop_bbox, RandNumbers& rand_num,
                 const bool do_mirror, const bool has_uint8,
                 const bool has_mean_file, const bool has_mean_values);

  template<bool do_mirror, bool has_mean_file, bool has_mean_values>
  void Transform(const cv::Mat& cv_img,
                 Blob<Dtype>* transformed_blob,
                 NormalizedBBox* crop_bbox,
                 RandNumbers& rand_num);

  template<bool has_uint8,  bool do_mirror, bool has_mean_file,
          bool has_mean_values>
  void Transform(const Datum& datum, Dtype* transformed_data,
                 NormalizedBBox* crop_bbox, RandNumbers& rand_num);

#ifdef USE_OPENCV
  void RandomResizeImage(const Datum& datum, Datum *resized_datum);
  void RandomResizeImage(const cv::Mat& img, cv::Mat *resized_img);
  void RandomAlterAspectRatio(const Datum& datum, Datum *resized_datum);
  void RandomAlterAspectRatio(const cv::Mat& img, cv::Mat *resized_img);
#endif
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_
